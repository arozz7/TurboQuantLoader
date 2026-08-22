//! HTTP utilities for the llama-server subprocess backend.
//!
//! - [`proxy_request`] — transparent byte-stream proxy (OpenAI route pass-through)
//! - [`spawn_event_reader`] — converts an OpenAI SSE response into a
//!   [`GenerateEvent`] channel consumed by the Anthropic translation layer

use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::metrics::{MetricsCollector, RequestMetrics};
use crate::model::backend::{GenerateEvent, GenerateStream, GenerateSummary};
use crate::server::error::ApiError;

/// Forward `body` to `{base_url}{path}` via POST and stream the response back.
///
/// The caller is responsible for any request/response translation (e.g. the
/// Anthropic route converts to OpenAI format before calling this function).
///
/// `extra_headers` are merged into the upstream request; `Content-Type:
/// application/json` is always set.
#[allow(dead_code)]
pub async fn proxy_request(
    client: &reqwest::Client,
    base_url: &str,
    path: &str,
    body: Vec<u8>,
    extra_headers: Option<&HeaderMap>,
) -> Result<Response, ApiError> {
    let url = format!("{base_url}{path}");

    let mut req = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body);

    // Merge caller-supplied headers (e.g. Authorization).
    if let Some(headers) = extra_headers {
        for (name, value) in headers {
            // Skip hop-by-hop and content headers — we set those ourselves.
            let n = name.as_str().to_lowercase();
            if n == "content-length" || n == "transfer-encoding" || n == "connection" {
                continue;
            }
            req = req.header(name.as_str(), value.as_bytes());
        }
    }

    let upstream = req
        .send()
        .await
        .with_context(|| format!("proxy request to {url} failed"))
        .map_err(ApiError::from)?;

    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Copy upstream Content-Type so the client knows what it's receiving.
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| HeaderValue::from_bytes(v.as_bytes()).ok())
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));

    // Stream the response body — works for both JSON blobs and SSE streams.
    let byte_stream = upstream
        .bytes_stream()
        .map(|result| result.map_err(std::io::Error::other));
    let body = Body::from_stream(byte_stream);

    let response = axum::response::Response::builder()
        .status(status)
        .header(axum::http::header::CONTENT_TYPE, content_type)
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());

    Ok(response)
}

// ── OpenAI SSE → GenerateEvent adapter ───────────────────────────────────────

/// Send a POST to `url` with the given JSON `body` and spawn a task that reads
/// the OpenAI SSE response, converting each delta into a [`GenerateEvent`].
///
/// Returns a [`GenerateStream`] (channel receiver) that the existing Anthropic
/// streaming handler can consume without modification.
pub async fn spawn_event_reader(
    client: &reqwest::Client,
    url: &str,
    body: Vec<u8>,
) -> Result<GenerateStream, ApiError> {
    let response = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .with_context(|| format!("upstream request to {url} failed"))
        .map_err(ApiError::from)?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(ApiError::from(anyhow::anyhow!(
            "llama-server returned {status}: {text}"
        )));
    }

    let (tx, rx) = mpsc::channel::<GenerateEvent>(256);

    tokio::spawn(async move {
        let mut buf = String::new();
        let mut stream = response.bytes_stream();
        let mut completion_tokens: u32 = 0;
        let mut prompt_tokens: u32 = 0;
        let mut cached_tokens: u32 = 0;
        let mut finish_reason_str = String::from("stop");

        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(GenerateEvent::Error(e.to_string())).await;
                    return;
                }
            };

            buf.push_str(&String::from_utf8_lossy(&bytes));

            // Drain complete SSE events (each event ends with \n\n).
            while let Some(pos) = buf.find("\n\n") {
                let event_str = buf[..pos].to_string();
                buf.drain(..pos + 2);

                for line in event_str.lines() {
                    let data = match line.strip_prefix("data: ") {
                        Some(d) => d,
                        None => continue,
                    };
                    if data == "[DONE]" {
                        continue;
                    }
                    let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                        continue;
                    };

                    // Capture usage when llama-server includes it.
                    if let Some(usage) = v.get("usage").filter(|u| !u.is_null()) {
                        prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                        completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0) as u32;
                        // llama.cpp ≥ b3900 reports cached tokens under prompt_tokens_details;
                        // older builds may use the top-level tokens_cached field.
                        cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
                            .as_u64()
                            .or_else(|| usage["tokens_cached"].as_u64())
                            .unwrap_or(0) as u32;
                    }

                    if let Some(fr) = v["choices"][0]["finish_reason"].as_str() {
                        if !fr.is_empty() && fr != "null" {
                            finish_reason_str = fr.to_string();
                        }
                    }

                    if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
                        if !content.is_empty() {
                            if completion_tokens == 0 {
                                completion_tokens += 1;
                            }
                            if tx
                                .send(GenerateEvent::Token(content.to_string()))
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }

        let _ = tx
            .send(GenerateEvent::Done(GenerateSummary {
                tokens_generated: completion_tokens,
                context_tokens: prompt_tokens + completion_tokens,
                tokens_per_second: 0.0,
                finish_reason: finish_reason_str,
                cached_tokens,
            }))
            .await;
    });

    Ok(rx)
}

// ── Tracked reader (with metrics) ────────────────────────────────────────────

/// Like [`spawn_event_reader`] but wraps the channel with a timing layer that
/// records TTFT, TPS, and token counts into [`MetricsCollector`] and emits a
/// structured completion log line.
pub async fn spawn_tracked_reader(
    client: &reqwest::Client,
    url: &str,
    body: Vec<u8>,
    metrics: Arc<MetricsCollector>,
    model: String,
) -> Result<GenerateStream, ApiError> {
    let start = Instant::now();
    let inner_rx = spawn_event_reader(client, url, body).await?;

    let (tx, rx) = mpsc::channel::<GenerateEvent>(256);

    tokio::spawn(async move {
        metrics
            .active_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut first_token = true;
        let mut ttft_ms = 0u64;
        let mut counted_tokens = 0u32;

        let mut stream = tokio_stream::wrappers::ReceiverStream::new(inner_rx);

        while let Some(event) = stream.next().await {
            match &event {
                GenerateEvent::Token(_) => {
                    if first_token {
                        ttft_ms = start.elapsed().as_millis() as u64;
                        first_token = false;
                    }
                    counted_tokens += 1;
                }
                GenerateEvent::Done(summary) => {
                    let generation_ms = start.elapsed().as_millis() as u64;
                    let tokens = if summary.tokens_generated > 0 {
                        summary.tokens_generated
                    } else {
                        counted_tokens
                    };
                    let tps = if generation_ms > 0 {
                        tokens as f32 / (generation_ms as f32 / 1000.0)
                    } else {
                        0.0
                    };
                    let prompt = summary.context_tokens.saturating_sub(tokens);

                    tracing::info!(
                        model = %model,
                        prompt_tokens = prompt,
                        completion_tokens = tokens,
                        ttft_ms,
                        generation_ms,
                        tps = format!("{tps:.1}"),
                        finish_reason = %summary.finish_reason,
                        "request complete"
                    );

                    metrics.inc_requests();
                    metrics
                        .record(RequestMetrics {
                            ttft_ms,
                            generation_ms,
                            tokens_per_second: tps,
                            prompt_tokens: prompt,
                            completion_tokens: tokens,
                            finish_reason: summary.finish_reason.clone(),
                            cached_tokens: summary.cached_tokens,
                        })
                        .await;
                }
                GenerateEvent::Error(_) => {
                    metrics.inc_errors();
                }
            }

            if tx.send(event).await.is_err() {
                break;
            }
        }

        metrics
            .active_requests
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    });

    Ok(rx)
}

// ── Request body builder ──────────────────────────────────────────────────────

/// Build a JSON body for llama-server's `/v1/chat/completions` from pre-processed
/// messages and sampler parameters.
pub fn build_chat_body(
    messages: &[(&str, String)],
    stream: bool,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: Option<f32>,
) -> Vec<u8> {
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
        .collect();

    let mut body = serde_json::json!({
        "model": "local",
        "messages": msgs,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream_options": {"include_usage": true},
    });

    if let Some(min_p) = min_p {
        body["min_p"] = serde_json::json!(min_p);
    }

    serde_json::to_vec(&body).unwrap_or_default()
}
