//! POST /v1/chat/completions — OpenAI Chat Completions API.
//!
//! Translates the incoming request (tool injection into system prompt for Qwen3),
//! proxies to llama-server, and streams/collects the response.

use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::conversation_log::{now_iso8601, ConversationEntry, ConversationLogger, LogMessage};
use crate::metrics::{MetricsCollector, RequestMetrics};
use crate::model::backend::GenerateEvent;
use crate::model::registry::ModelRegistry;
use crate::server::error::ApiError;
use crate::server::proxy::{build_chat_body, spawn_tracked_reader};
use crate::server::sse::{data_event, sse_response};
use crate::server::stream_parser::{ParsedEvent, StreamParser};
use crate::server::types::openai::{
    ChatCompletionChunk, ChatCompletionRequest, DeltaMessage, StreamChoice, ToolCallFunction,
    ToolCallInfo, Usage,
};
use crate::server::AppState;

/// `POST /v1/chat/completions`
pub async fn chat_completions(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    // Gate: return 503 while a model switch is in progress.
    if state.switching.load(Ordering::SeqCst) {
        return Ok(switching_503());
    }

    let req: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            let body_str = String::from_utf8_lossy(&body);
            tracing::error!(
                "Failed to parse ChatCompletionRequest: {}\nRaw Body: {}",
                e,
                body_str
            );
            return Err(anyhow::anyhow!("Invalid request: {}", e).into());
        }
    };

    // If the client requested a specific model that differs from the loaded one,
    // trigger a switch and return 503 so the client retries after load.
    let current = state.current_model_name().await;
    if !ModelRegistry::matches_current(&req.model, &current)
        && state.trigger_model_switch(&req.model).await
    {
        return Ok(switching_503());
    }
    // resolve returned None (unknown model) — fall through and serve with current model

    state.touch_last_request();

    let proc = state.process_snapshot().await;
    let cfg = state.config_snapshot().await;
    let model_name = model_name_from_config(&cfg);

    let messages = prepare_messages(&req);
    let url = format!("{}/v1/chat/completions", proc.base_url());
    let temperature = req.temperature.or(cfg.backend.temperature).unwrap_or(0.6);
    let top_p = req.top_p.or(cfg.backend.top_p).unwrap_or(0.95);
    let top_k = req.top_k.unwrap_or(20);
    let min_p = req.min_p.or(cfg.backend.min_p);

    let request_id = new_id("chatcmpl");

    tracing::info!(
        id = %request_id,
        model = %model_name,
        stream = req.stream,
        messages = req.messages.len(),
        tools = req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
        max_tokens = req.max_tokens,
        "OpenAI chat request"
    );

    // Build log-friendly message list from the prepared (tool-injected) messages.
    let log_messages: Vec<LogMessage> = messages
        .iter()
        .map(|(role, content)| LogMessage {
            role: role.to_string(),
            content: content.clone(),
        })
        .collect();

    if req.stream {
        let body = build_chat_body(
            &messages,
            true,
            req.max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
        );
        let rx = spawn_tracked_reader(
            proc.http_client(),
            &url,
            body,
            state.metrics.clone(),
            model_name.clone(),
        )
        .await?;
        let include_usage = req
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);
        Ok(streaming_response(
            rx,
            model_name,
            request_id,
            log_messages,
            state.conv_logger.clone(),
            include_usage,
        ))
    } else {
        let body = build_chat_body(
            &messages,
            false,
            req.max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
        );
        non_streaming_response(
            proc.http_client(),
            &url,
            body,
            NonStreamingArgs {
                request_id,
                model_name,
                log_messages,
                conv_logger: state.conv_logger.clone(),
                metrics: state.metrics.clone(),
            },
        )
        .await
    }
}

// ── Message preparation ───────────────────────────────────────────────────────

fn prepare_messages(req: &ChatCompletionRequest) -> Vec<(&'static str, String)> {
    let mut messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| {
            let mut text = m
                .content
                .as_ref()
                .map(|c| c.clone().into_text())
                .unwrap_or_default();

            if let Some(tool_calls) = &m.tool_calls {
                for tc in tool_calls {
                    if let (Some(name), Some(args)) = (&tc.function.name, &tc.function.arguments) {
                        text.push_str(&format!(
                            "\n<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>",
                            name, args
                        ));
                    }
                }
            }

            (m.role.clone(), text)
        })
        .collect();

    if let Some(tools) = &req.tools {
        if !tools.is_empty() {
            let tools_json = serde_json::to_string_pretty(tools).unwrap_or_default();
            let injection = format!(
                "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
                 You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\
                 {tools_json}\n</tools>\n\n\
                 For each function call, you MUST return a json object with function name and arguments \
                 within <tool_call></tool_call> XML tags.\n\
                 Format:\n<tool_call>\n{{\"name\": \"tool_name_here\", \"arguments\": {{\"arg_1\": \"value_1\"}}}}\n</tool_call>"
            );
            if let Some(sys) = messages.iter_mut().find(|(r, _)| r == "system") {
                sys.1.push_str(&injection);
            } else {
                messages.insert(
                    0,
                    (
                        "system".into(),
                        format!("You are a helpful assistant.{injection}"),
                    ),
                );
            }
        }
    }

    messages
        .into_iter()
        .map(|(role, content)| {
            let r: &'static str = match role.as_str() {
                "system" => "system",
                "assistant" => "assistant",
                "tool" => "tool",
                _ => "user",
            };
            (r, content)
        })
        .collect()
}

// ── Streaming response ────────────────────────────────────────────────────────

fn streaming_response(
    mut rx: crate::model::backend::GenerateStream,
    model: String,
    id: String,
    log_messages: Vec<LogMessage>,
    conv_logger: Arc<ConversationLogger>,
    include_usage: bool,
) -> Response {
    let created = unix_now();
    let model_clone = model.clone();

    let (tx, sse_rx) = tokio::sync::mpsc::unbounded_channel();

    let first = data_event(&ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model_clone.clone(),
        choices: vec![StreamChoice {
            index: 0,
            delta: DeltaMessage {
                role: Some("assistant"),
                content: Some(String::new()),
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
        usage: None,
    });
    let _ = tx.send(Ok::<_, Infallible>(first));

    tokio::spawn(async move {
        let mut parser = StreamParser::new();
        let mut tool_called = false;
        let mut response_buf = String::new();

        let emit_events = |events: Vec<ParsedEvent>, tx: &tokio::sync::mpsc::UnboundedSender<_>| {
            for evt in events {
                match evt {
                    ParsedEvent::TextToken(text) => {
                        let _ = tx.send(Ok::<_, Infallible>(data_event(&ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_clone.clone(),
                            choices: vec![StreamChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(text),
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        })));
                    }
                    ParsedEvent::ThinkingToken(text) => {
                        let _ = tx.send(Ok::<_, Infallible>(data_event(&ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_clone.clone(),
                            choices: vec![StreamChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: None,
                                    reasoning_content: Some(text),
                                    tool_calls: None,
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        })));
                    }
                    ParsedEvent::ThinkingEnd => {}
                    ParsedEvent::ToolCallReady { name, arguments } => {
                        let _ = tx.send(Ok::<_, Infallible>(data_event(&ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_clone.clone(),
                            choices: vec![StreamChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: Some(vec![ToolCallInfo {
                                        index: 0,
                                        id: Some(new_id("call")),
                                        r#type: Some("function".to_string()),
                                        function: ToolCallFunction {
                                            name: Some(name),
                                            arguments: Some(
                                                serde_json::to_string(&arguments)
                                                    .unwrap_or_else(|_| "{}".to_string()),
                                            ),
                                        },
                                    }]),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        })));
                    }
                }
            }
        };

        while let Some(event) = rx.recv().await {
            match event {
                GenerateEvent::Token(text) => {
                    response_buf.push_str(&text);
                    let evts = parser.push(&text);
                    if evts
                        .iter()
                        .any(|e| matches!(e, ParsedEvent::ToolCallReady { .. }))
                    {
                        tool_called = true;
                    }
                    emit_events(evts, &tx);
                }
                GenerateEvent::Done(summary) => {
                    let evts = parser.flush();
                    if evts
                        .iter()
                        .any(|e| matches!(e, ParsedEvent::ToolCallReady { .. }))
                    {
                        tool_called = true;
                    }
                    emit_events(evts, &tx);

                    let finish_reason = if tool_called {
                        "tool_calls"
                    } else {
                        match summary.finish_reason.as_str() {
                            "length" => "length",
                            "tool_calls" => "tool_calls",
                            _ => "stop",
                        }
                    };

                    let prompt_tokens = summary
                        .context_tokens
                        .saturating_sub(summary.tokens_generated);
                    conv_logger.log(&ConversationEntry {
                        ts: now_iso8601(),
                        id: id.clone(),
                        model: model_clone.clone(),
                        protocol: "openai",
                        stream: true,
                        messages: log_messages,
                        response: response_buf,
                        prompt_tokens,
                        completion_tokens: summary.tokens_generated,
                        tps: summary.tokens_per_second,
                        finish_reason: finish_reason.to_string(),
                    });

                    let _ = tx.send(Ok::<_, Infallible>(data_event(&ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_clone.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: DeltaMessage {
                                role: None,
                                content: None,
                                reasoning_content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    })));

                    if include_usage {
                        let _ = tx.send(Ok::<_, Infallible>(data_event(&ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_clone.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens,
                                completion_tokens: summary.tokens_generated,
                                total_tokens: summary.context_tokens,
                            }),
                        })));
                    }

                    let _ = tx.send(Ok::<_, Infallible>(
                        axum::response::sse::Event::default().data("[DONE]"),
                    ));
                    break;
                }
                GenerateEvent::Error(e) => {
                    tracing::warn!(error = %e, "generation error during OpenAI stream");
                    break;
                }
            }
        }
    });

    sse_response(UnboundedReceiverStream::new(sse_rx))
}

// ── Non-streaming response ────────────────────────────────────────────────────

struct NonStreamingArgs {
    request_id: String,
    model_name: String,
    log_messages: Vec<LogMessage>,
    conv_logger: Arc<ConversationLogger>,
    metrics: Arc<MetricsCollector>,
}

async fn non_streaming_response(
    client: &reqwest::Client,
    url: &str,
    body: Vec<u8>,
    args: NonStreamingArgs,
) -> Result<Response, ApiError> {
    let NonStreamingArgs {
        request_id,
        model_name,
        log_messages,
        conv_logger,
        metrics,
    } = args;
    let start = Instant::now();
    metrics
        .active_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let upstream = client
        .post(url)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .map_err(|e| ApiError::from(anyhow::anyhow!("upstream request failed: {e}")))?;

    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let content_type = upstream
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    let bytes = upstream
        .bytes()
        .await
        .map_err(|e| ApiError::from(anyhow::anyhow!("failed to read upstream body: {e}")))?;

    // Default to the raw upstream bytes; replaced below if we successfully
    // split out a reasoning block, so any parse failure just falls back to
    // proxying llama-server's response verbatim (today's behavior).
    let mut response_bytes = bytes.clone();

    if status.is_success() {
        if let Ok(mut v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            let raw_content = v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            // llama-server doesn't separate reasoning from content for this
            // chat template (confirmed empirically — <think> tags land
            // unparsed in `content`), so split it out ourselves the same way
            // the streaming path's StreamParser does, keeping non-streaming
            // and streaming clients consistent on the reasoning_content
            // convention. Does not touch <tool_call> handling — non-streaming
            // tool-call parsing is unchanged from today.
            let (visible_content, reasoning_content) = split_reasoning(&raw_content);
            if let Some(message) = v["choices"][0]["message"].as_object_mut() {
                message.insert(
                    "content".to_string(),
                    serde_json::Value::String(visible_content.clone()),
                );
                if let Some(reasoning) = reasoning_content {
                    message.insert(
                        "reasoning_content".to_string(),
                        serde_json::Value::String(reasoning),
                    );
                }
            }
            if let Ok(rewritten) = serde_json::to_vec(&v) {
                response_bytes = axum::body::Bytes::from(rewritten);
            }

            let response_text = raw_content;
            let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
            let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
            let cached_tokens = v["usage"]["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .or_else(|| v["usage"]["tokens_cached"].as_u64())
                .unwrap_or(0) as u32;
            let finish_reason = v["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or("stop")
                .to_string();
            let generation_ms = start.elapsed().as_millis() as u64;
            let tps = if generation_ms > 0 {
                completion_tokens as f32 / (generation_ms as f32 / 1000.0)
            } else {
                0.0
            };

            conv_logger.log(&ConversationEntry {
                ts: now_iso8601(),
                id: request_id,
                model: model_name,
                protocol: "openai",
                stream: false,
                messages: log_messages,
                response: response_text,
                prompt_tokens,
                completion_tokens,
                tps,
                finish_reason: finish_reason.clone(),
            });

            metrics.inc_requests();
            metrics
                .record(RequestMetrics {
                    ttft_ms: generation_ms,
                    generation_ms,
                    tokens_per_second: tps,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    cached_tokens,
                })
                .await;
        }
    } else {
        metrics.inc_errors();
    }

    metrics
        .active_requests
        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

    Ok(axum::response::Response::builder()
        .status(status)
        .header(axum::http::header::CONTENT_TYPE, content_type)
        .body(axum::body::Body::from(response_bytes))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()))
}

/// Split a `<think>...</think>` block out of a complete (non-streamed)
/// response string, mirroring the tag [`StreamParser`] recognizes on the
/// streaming path. Returns `(visible_content, reasoning_content)` — the
/// second element is `None` when no thinking block is present or it's empty.
fn split_reasoning(content: &str) -> (String, Option<String>) {
    if let (Some(start), Some(end)) = (content.find("<think>"), content.find("</think>")) {
        if end > start {
            let reasoning = content[start + "<think>".len()..end].trim().to_string();
            let mut visible = String::with_capacity(content.len());
            visible.push_str(&content[..start]);
            visible.push_str(&content[end + "</think>".len()..]);
            let visible = visible.trim().to_string();
            let reasoning = if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            };
            return (visible, reasoning);
        }
    }
    (content.to_string(), None)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn switching_503() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "model switching in progress — retry in a few seconds",
            "type": "service_unavailable",
        }
    });
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [("Retry-After", "10")],
        Json(body),
    )
        .into_response()
}

fn model_name_from_config(config: &crate::config::AppConfig) -> String {
    config
        .model
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("local-model")
        .to_string()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn new_id(prefix: &str) -> String {
    format!("{}-{}", prefix, uuid::Uuid::new_v4())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_reasoning_extracts_think_block() {
        let (visible, reasoning) = split_reasoning("<think>pondering</think>the answer");
        assert_eq!(visible, "the answer");
        assert_eq!(reasoning.as_deref(), Some("pondering"));
    }

    #[test]
    fn split_reasoning_no_think_block_passes_through() {
        let (visible, reasoning) = split_reasoning("just an answer");
        assert_eq!(visible, "just an answer");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn split_reasoning_empty_think_block_yields_no_reasoning() {
        let (visible, reasoning) = split_reasoning("<think></think>the answer");
        assert_eq!(visible, "the answer");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn split_reasoning_preserves_tool_call_tags_in_visible_content() {
        let (visible, reasoning) = split_reasoning(
            "<think>need to read the file</think>\n<tool_call>\n{\"name\": \"Read\"}\n</tool_call>",
        );
        assert_eq!(reasoning.as_deref(), Some("need to read the file"));
        assert!(visible.contains("<tool_call>"));
    }
}
