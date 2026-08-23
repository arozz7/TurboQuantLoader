//! POST /v1/messages — Anthropic Messages API.
//!
//! SSE event sequence for a response with thinking:
//!
//! ```text
//! message_start
//! ping
//! content_block_start  (thinking, index=0)      ← only if model thinks
//! content_block_delta  (thinking_delta) × N
//! content_block_stop   (index=0)
//! content_block_start  (text, index=1)
//! content_block_delta  (text_delta) × M
//! content_block_stop   (index=1)
//! message_delta        (stop_reason=end_turn)
//! message_stop
//! ```
//!
//! For a response with no thinking the thinking block pair is omitted and the
//! text block runs at index 0.

use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;

use crate::conversation_log::{now_iso8601, ConversationEntry, ConversationLogger, LogMessage};
use crate::metrics::RequestMetrics;
use crate::model::backend::GenerateEvent;
use crate::model::registry::ModelRegistry;
use crate::server::error::ApiError;
use crate::server::proxy::{build_chat_body, spawn_tracked_reader};
use crate::server::sse::{
    input_json_delta_event, named_event, sse_response, tool_use_block_start_event,
};
use crate::server::stream_parser::{ParsedEvent, StreamParser};
use crate::server::types::anthropic::{
    ContentBlockDeltaEvent, ContentBlockStartData, ContentBlockStartEvent, ContentBlockStopEvent,
    InputUsage, MessageDeltaData, MessageDeltaEvent, MessageStartData, MessageStartEvent,
    MessageStopEvent, MessagesRequest, MessagesResponse, OutputUsage, PingEvent, TextBlock,
    TextDelta,
};
use crate::server::AppState;

/// `POST /v1/messages`
pub async fn create_message(
    State(state): State<AppState>,
    Json(req): Json<MessagesRequest>,
) -> Result<Response, ApiError> {
    // Gate: return 503 while a model switch is in progress.
    if state.switching.load(Ordering::SeqCst) {
        return Ok(switching_503());
    }

    // Trigger a model switch when the client requests a different model.
    let current = state.current_model_name().await;
    if !ModelRegistry::matches_current(&req.model, &current)
        && state.trigger_model_switch(&req.model).await
    {
        return Ok(switching_503());
    }

    state.touch_last_request();

    let proc = state.process_snapshot().await;
    let cfg = state.config_snapshot().await;

    let model_name = cfg
        .model
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("local-model")
        .to_string();

    let msg_count = req.messages.len();
    let has_tools = req.tools.is_some();
    let max_tokens = req.max_tokens;
    let request_id = format!("msg_{}", uuid::Uuid::new_v4());

    tracing::info!(
        id = %request_id,
        model = %model_name,
        stream = req.stream,
        messages = msg_count,
        tools = has_tools,
        max_tokens,
        "Anthropic messages request"
    );

    let messages = to_openai_messages(&req);
    let log_messages: Vec<LogMessage> = messages
        .iter()
        .map(|(role, content)| LogMessage {
            role: role.to_string(),
            content: content.clone(),
        })
        .collect();

    let temperature = req.temperature.or(cfg.backend.temperature).unwrap_or(0.6);
    let top_p = req.top_p.or(cfg.backend.top_p).unwrap_or(0.95);
    let top_k = req.top_k.unwrap_or(20);
    let min_p = req.min_p.or(cfg.backend.min_p);
    let base_url = proc.base_url();
    let http = proc.http_client().clone();

    if req.stream {
        let url = format!("{base_url}/v1/chat/completions");
        let body = build_chat_body(
            &messages,
            true,
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
        );
        let rx = spawn_tracked_reader(&http, &url, body, state.metrics.clone(), model_name.clone())
            .await?;
        Ok(streaming_response(
            rx,
            model_name,
            request_id,
            log_messages,
            state.conv_logger.clone(),
        ))
    } else {
        let url = format!("{base_url}/v1/chat/completions");
        let body = build_chat_body(
            &messages,
            false,
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
        );
        let start = Instant::now();
        state
            .metrics
            .active_requests
            .fetch_add(1, Ordering::Relaxed);

        let upstream = http
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| ApiError::from(anyhow::anyhow!("upstream request failed: {e}")))?;

        let json: serde_json::Value = upstream.json().await.map_err(|e| {
            ApiError::from(anyhow::anyhow!("failed to parse upstream response: {e}"))
        })?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
        let cached_tokens = json["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .or_else(|| json["usage"]["tokens_cached"].as_u64())
            .unwrap_or(0) as u32;
        let finish_reason = json["choices"][0]["finish_reason"]
            .as_str()
            .unwrap_or("stop")
            .to_string();
        let generation_ms = start.elapsed().as_millis() as u64;
        let tps = if generation_ms > 0 {
            completion_tokens as f32 / (generation_ms as f32 / 1000.0)
        } else {
            0.0
        };

        tracing::info!(
            tokens = completion_tokens,
            finish_reason = %finish_reason,
            "non-streaming response complete"
        );

        state.conv_logger.log(&ConversationEntry {
            ts: now_iso8601(),
            id: request_id,
            model: model_name.clone(),
            protocol: "anthropic",
            stream: false,
            messages: log_messages,
            response: text.clone(),
            prompt_tokens,
            completion_tokens,
            tps,
            finish_reason: finish_reason.clone(),
        });

        state.metrics.inc_requests();
        state
            .metrics
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
        state
            .metrics
            .active_requests
            .fetch_sub(1, Ordering::Relaxed);

        Ok(full_response(
            text,
            prompt_tokens,
            completion_tokens,
            &model_name,
        ))
    }
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

// ── Request translation ───────────────────────────────────────────────────────

/// Convert an Anthropic `MessagesRequest` to a list of `(role, content)` pairs
/// suitable for [`build_chat_body`], including Qwen3 tool injection in the
/// system prompt.
fn to_openai_messages(req: &MessagesRequest) -> Vec<(&'static str, String)> {
    let system_text = req
        .system
        .clone()
        .map(|s| s.into_text())
        .unwrap_or_default();
    let system_content = if let Some(tools) = &req.tools {
        if tools.is_empty() {
            system_text
        } else {
            let tools_json = tools_to_qwen3_json(tools);
            format!(
                "{system_text}\n\n# Tools\n\nYou may call one or more functions to assist with \
                 the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools_json}\n</tools>\n\n\
                 For each function call, you MUST return a json object with function name and arguments within <tool_call></tool_call> XML tags.\n\
                 Format:\n<tool_call>\n{{\"name\": \"tool_name_here\", \"arguments\": {{\"arg_1\": \"value_1\"}}}}\n</tool_call>"
            )
        }
    } else {
        system_text
    };

    let mut out: Vec<(&'static str, String)> = Vec::new();

    if !system_content.is_empty() {
        out.push(("system", system_content));
    }

    for m in &req.messages {
        let role: &'static str = match m.role.as_str() {
            "assistant" => "assistant",
            "tool" => "tool",
            _ => "user",
        };
        out.push((role, m.content.clone().into_qwen3_text()));
    }

    out
}

/// Convert Anthropic tools array to the JSON string Qwen3 expects inside
/// `<tools>…</tools>`.
///
/// Anthropic format: `{ name, description, input_schema: { type, properties, required } }`
/// Qwen3 format:   `[{ "type": "function", "function": { name, description, parameters } }]`
fn tools_to_qwen3_json(tools: &[serde_json::Value]) -> String {
    let converted: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            let name = t.get("name").cloned().unwrap_or(serde_json::Value::Null);
            let description = t
                .get("description")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let parameters = t
                .get("input_schema")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            })
        })
        .collect();
    serde_json::to_string_pretty(&converted).unwrap_or_default()
}

// ── Non-streaming response ────────────────────────────────────────────────────

fn full_response(text: String, prompt_tokens: u32, output_tokens: u32, model: &str) -> Response {
    let resp = MessagesResponse {
        id: new_id("msg"),
        r#type: "message",
        role: "assistant",
        content: vec![TextBlock {
            r#type: "text",
            text,
        }],
        model: model.to_string(),
        stop_reason: "end_turn",
        stop_sequence: None,
        usage: InputUsage {
            input_tokens: prompt_tokens,
            output_tokens,
        },
    };
    Json(resp).into_response()
}

// ── Streaming response ────────────────────────────────────────────────────────

fn streaming_response(
    rx: crate::model::backend::GenerateStream,
    model: String,
    request_id: String,
    log_messages: Vec<LogMessage>,
    conv_logger: Arc<ConversationLogger>,
) -> Response {
    let (tx, output_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);
    let model_clone = model.clone();

    tokio::spawn(async move {
        // ── Preamble ──────────────────────────────────────────────────────────
        let ok = tx
            .send(Ok(named_event(
                "message_start",
                &MessageStartEvent {
                    r#type: "message_start",
                    message: MessageStartData {
                        id: request_id.clone(),
                        r#type: "message",
                        role: "assistant",
                        content: vec![],
                        model,
                        stop_reason: None,
                        stop_sequence: None,
                        usage: InputUsage {
                            input_tokens: 0,
                            output_tokens: 1,
                        },
                    },
                },
            )))
            .await;
        if ok.is_err() {
            return;
        }
        if tx
            .send(Ok(named_event("ping", &PingEvent { r#type: "ping" })))
            .await
            .is_err()
        {
            return;
        }

        // ── Token processing ──────────────────────────────────────────────────
        let mut parser = StreamParser::new();
        let mut block_index: u32 = 0;
        let mut text_open = false;
        let mut tool_called = false;
        let mut response_buf = String::new();

        let mut model_stream = ReceiverStream::new(rx);
        while let Some(model_event) = model_stream.next().await {
            match model_event {
                GenerateEvent::Token(text) => {
                    response_buf.push_str(&text);
                    let events = parser.push(&text);
                    if emit_parsed(
                        &tx,
                        events,
                        &mut block_index,
                        &mut text_open,
                        &mut tool_called,
                    )
                    .await
                    .is_err()
                    {
                        return;
                    }
                }
                GenerateEvent::Done(summary) => {
                    let events = parser.flush();
                    let _ = emit_parsed(
                        &tx,
                        events,
                        &mut block_index,
                        &mut text_open,
                        &mut tool_called,
                    )
                    .await;

                    if text_open {
                        let _ = tx
                            .send(Ok(named_event(
                                "content_block_stop",
                                &ContentBlockStopEvent {
                                    r#type: "content_block_stop",
                                    index: block_index,
                                },
                            )))
                            .await;
                    }

                    let stop_reason = if tool_called { "tool_use" } else { "end_turn" };

                    let prompt_tokens = summary
                        .context_tokens
                        .saturating_sub(summary.tokens_generated);
                    conv_logger.log(&ConversationEntry {
                        ts: now_iso8601(),
                        id: request_id.clone(),
                        model: model_clone.clone(),
                        protocol: "anthropic",
                        stream: true,
                        messages: log_messages,
                        response: response_buf,
                        prompt_tokens,
                        completion_tokens: summary.tokens_generated,
                        tps: summary.tokens_per_second,
                        finish_reason: stop_reason.to_string(),
                    });

                    let _ = tx
                        .send(Ok(named_event(
                            "message_delta",
                            &MessageDeltaEvent {
                                r#type: "message_delta",
                                delta: MessageDeltaData {
                                    stop_reason,
                                    stop_sequence: None,
                                },
                                usage: OutputUsage {
                                    output_tokens: summary.tokens_generated,
                                },
                            },
                        )))
                        .await;
                    let _ = tx
                        .send(Ok(named_event(
                            "message_stop",
                            &MessageStopEvent {
                                r#type: "message_stop",
                            },
                        )))
                        .await;
                    break;
                }
                GenerateEvent::Error(e) => {
                    tracing::warn!(error = %e, "generation error during Anthropic stream");
                    break;
                }
            }
        }
    });

    sse_response(ReceiverStream::new(output_rx))
}

// ── Parsed-event → SSE dispatch ───────────────────────────────────────────────

async fn emit_parsed(
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    events: Vec<ParsedEvent>,
    block_index: &mut u32,
    text_open: &mut bool,
    tool_called: &mut bool,
) -> Result<(), ()> {
    for parsed in events {
        match parsed {
            ParsedEvent::ThinkingToken(t) => {
                // Emit thinking tokens as standard text blocks to prevent SDK crashes.
                if !*text_open {
                    tx.send(Ok(named_event(
                        "content_block_start",
                        &ContentBlockStartEvent {
                            r#type: "content_block_start",
                            index: *block_index,
                            content_block: ContentBlockStartData {
                                r#type: "text",
                                text: "",
                            },
                        },
                    )))
                    .await
                    .map_err(|_| ())?;
                    *text_open = true;
                }
                tx.send(Ok(named_event(
                    "content_block_delta",
                    &ContentBlockDeltaEvent {
                        r#type: "content_block_delta",
                        index: *block_index,
                        delta: TextDelta {
                            r#type: "text_delta",
                            text: t,
                        },
                    },
                )))
                .await
                .map_err(|_| ())?;
            }

            ParsedEvent::ThinkingEnd => {
                // Model continues into text block immediately after thinking.
            }

            ParsedEvent::TextToken(t) => {
                if !*text_open {
                    tx.send(Ok(named_event(
                        "content_block_start",
                        &ContentBlockStartEvent {
                            r#type: "content_block_start",
                            index: *block_index,
                            content_block: ContentBlockStartData {
                                r#type: "text",
                                text: "",
                            },
                        },
                    )))
                    .await
                    .map_err(|_| ())?;
                    *text_open = true;
                }
                tx.send(Ok(named_event(
                    "content_block_delta",
                    &ContentBlockDeltaEvent {
                        r#type: "content_block_delta",
                        index: *block_index,
                        delta: TextDelta {
                            r#type: "text_delta",
                            text: t,
                        },
                    },
                )))
                .await
                .map_err(|_| ())?;
            }

            ParsedEvent::ToolCallReady { name, arguments } => {
                if *text_open {
                    tx.send(Ok(named_event(
                        "content_block_stop",
                        &ContentBlockStopEvent {
                            r#type: "content_block_stop",
                            index: *block_index,
                        },
                    )))
                    .await
                    .map_err(|_| ())?;
                    *text_open = false;
                    *block_index += 1;
                }

                let tool_id = new_id("toolu");
                tx.send(Ok(tool_use_block_start_event(*block_index, tool_id, name)))
                    .await
                    .map_err(|_| ())?;

                let json = serde_json::to_string(&arguments).unwrap_or_default();
                tx.send(Ok(input_json_delta_event(*block_index, json)))
                    .await
                    .map_err(|_| ())?;

                tx.send(Ok(named_event(
                    "content_block_stop",
                    &ContentBlockStopEvent {
                        r#type: "content_block_stop",
                        index: *block_index,
                    },
                )))
                .await
                .map_err(|_| ())?;

                *block_index += 1;
                *tool_called = true;
            }
        }
    }
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn new_id(prefix: &str) -> String {
    format!("{}_{}", prefix, uuid::Uuid::new_v4().simple())
}
