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

use axum::extract::State;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;

use crate::inference::engine::{ChatMessage, ChatRequest};
use crate::model::backend::{GenerateEvent, GenerateStream, SamplerParams};
use crate::server::error::ApiError;
use crate::server::sse::{
    input_json_delta_event, named_event, sse_response, thinking_block_start_event,
    thinking_delta_event, tool_use_block_start_event,
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
    let msg_count = req.messages.len();
    let has_tools = req.tools.is_some();
    // Claude Code requests up to 32 000 tokens but actual responses are short.
    // Cap here to prevent runaway generation; raise once tool use is implemented.
    let max_tokens = req.max_tokens.min(4096);
    tracing::info!(
        stream = req.stream,
        messages = msg_count,
        has_tools,
        max_tokens,
        requested_max_tokens = req.max_tokens,
        "POST /v1/messages"
    );

    let chat_req = to_chat_request(&req, max_tokens);

    if req.stream {
        let stream = tokio::task::block_in_place(|| state.engine.chat(chat_req))?;
        Ok(streaming_response(stream.into_inner(), state.engine.model_name().to_string()))
    } else {
        let stream = tokio::task::block_in_place(|| state.engine.chat(chat_req))?;
        let (text, summary) =
            stream.collect_full().await.map_err(|e| anyhow::anyhow!(e))?;
        tracing::info!(
            tokens = summary.tokens_generated,
            tps = summary.tokens_per_second,
            "non-streaming response complete"
        );
        Ok(full_response(
            text,
            summary.context_tokens,
            summary.tokens_generated,
            state.engine.model_name(),
        ))
    }
}

// ── Request translation ───────────────────────────────────────────────────────

fn to_chat_request(req: &MessagesRequest, max_tokens: u32) -> ChatRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // Build the system message, appending a <tools> block when tools are present.
    let system_text = req.system.clone().map(|s| s.into_text()).unwrap_or_default();
    let system_content = if let Some(tools) = &req.tools {
        if tools.is_empty() {
            system_text
        } else {
            let tools_json = tools_to_qwen3_json(tools);
            format!(
                "{system_text}\n\n# Tools\n\nYou may call one or more functions to assist with \
                 the user query.\n\n<tools>\n{tools_json}\n</tools>"
            )
        }
    } else {
        system_text
    };

    if !system_content.is_empty() {
        messages.push(ChatMessage { role: "system".into(), content: system_content });
    }

    for m in &req.messages {
        messages.push(ChatMessage {
            role: m.role.clone(),
            content: m.content.clone().into_qwen3_text(),
        });
    }

    ChatRequest {
        messages,
        max_tokens,
        sampler: SamplerParams {
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            top_k: req.top_k.unwrap_or(40),
            ..SamplerParams::default()
        },
    }
}

/// Convert an Anthropic tools array into the JSON string Qwen3.5 expects inside
/// `<tools>…</tools>`.
///
/// Anthropic format: `{ name, description, input_schema: { type, properties, required } }`
/// Qwen3.5 format:   `[{ "type": "function", "function": { name, description, parameters } }]`
fn tools_to_qwen3_json(tools: &[serde_json::Value]) -> String {
    let converted: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            let name = t.get("name").cloned().unwrap_or(serde_json::Value::Null);
            let description = t.get("description").cloned().unwrap_or(serde_json::Value::Null);
            // Anthropic calls the schema "input_schema"; Qwen3.5 expects "parameters".
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

fn full_response(
    text: String,
    context_tokens: u32,
    output_tokens: u32,
    model: &str,
) -> Response {
    let prompt_tokens = context_tokens.saturating_sub(output_tokens);
    let resp = MessagesResponse {
        id: new_id("msg"),
        r#type: "message",
        role: "assistant",
        content: vec![TextBlock { r#type: "text", text }],
        model: model.to_string(),
        stop_reason: "end_turn",
        stop_sequence: None,
        usage: InputUsage { input_tokens: prompt_tokens, output_tokens },
    };
    Json(resp).into_response()
}

// ── Streaming response ────────────────────────────────────────────────────────

/// Build a streaming SSE [`Response`] that routes model tokens through the
/// [`StreamParser`], emitting Anthropic thinking and tool_use content blocks
/// as well as plain text deltas.
fn streaming_response(rx: GenerateStream, model: String) -> Response {
    let msg_id = new_id("msg");
    let (tx, output_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);

    tokio::spawn(async move {
        // ── Preamble ──────────────────────────────────────────────────────────
        // content_block_start is deferred — we don't know yet whether the first
        // block will be thinking or text.
        let ok = tx
            .send(Ok(named_event(
                "message_start",
                &MessageStartEvent {
                    r#type: "message_start",
                    message: MessageStartData {
                        id: msg_id,
                        r#type: "message",
                        role: "assistant",
                        content: vec![],
                        model,
                        stop_reason: None,
                        stop_sequence: None,
                        usage: InputUsage { input_tokens: 0, output_tokens: 1 },
                    },
                },
            )))
            .await;
        if ok.is_err() {
            return;
        }
        if tx.send(Ok(named_event("ping", &PingEvent { r#type: "ping" }))).await.is_err() {
            return;
        }

        // ── Token processing ──────────────────────────────────────────────────
        let mut parser = StreamParser::new();
        let mut block_index: u32 = 0;
        let mut thinking_open = false;
        let mut text_open = false;
        let mut tool_called = false;

        let mut model_stream = ReceiverStream::new(rx);
        while let Some(model_event) = model_stream.next().await {
            match model_event {
                GenerateEvent::Token(text) => {
                    let events = parser.push(&text);
                    if emit_parsed(
                        &tx,
                        events,
                        &mut block_index,
                        &mut thinking_open,
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
                    // Flush any buffered partial tag content.
                    let events = parser.flush();
                    let _ = emit_parsed(
                        &tx,
                        events,
                        &mut block_index,
                        &mut thinking_open,
                        &mut text_open,
                        &mut tool_called,
                    )
                    .await;

                    // Close any open content block.
                    if thinking_open || text_open {
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

                    // Final events.
                    // Use stop_reason "tool_use" when the model invoked a tool so that
                    // Claude Code knows to run the tool and send a follow-up request.
                    let stop_reason = if tool_called { "tool_use" } else { "end_turn" };
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
                            &MessageStopEvent { r#type: "message_stop" },
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

/// Translate a batch of [`ParsedEvent`]s into SSE events and send them.
///
/// Returns `Err(())` if the receiver has been dropped (client disconnected).
async fn emit_parsed(
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    events: Vec<ParsedEvent>,
    block_index: &mut u32,
    thinking_open: &mut bool,
    text_open: &mut bool,
    tool_called: &mut bool,
) -> Result<(), ()> {
    for parsed in events {
        match parsed {
            ParsedEvent::ThinkingToken(t) => {
                // Emit thinking tokens as standard text blocks to prevent Claude SDK crashing
                if !*text_open {
                    tx.send(Ok(named_event(
                        "content_block_start",
                        &ContentBlockStartEvent {
                            r#type: "content_block_start",
                            index: *block_index,
                            content_block: ContentBlockStartData { r#type: "text", text: "" },
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
                        delta: TextDelta { r#type: "text_delta", text: t },
                    },
                )))
                .await
                .map_err(|_| ())?;
            }

            ParsedEvent::ThinkingEnd => {
                // Do not close the block. The model continues emitting text smoothly
                // into the same text block right after thinking!
            }

            ParsedEvent::TextToken(t) => {
                if !*text_open {
                    tx.send(Ok(named_event(
                        "content_block_start",
                        &ContentBlockStartEvent {
                            r#type: "content_block_start",
                            index: *block_index,
                            content_block: ContentBlockStartData { r#type: "text", text: "" },
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
                        delta: TextDelta { r#type: "text_delta", text: t },
                    },
                )))
                .await
                .map_err(|_| ())?;
            }

            ParsedEvent::ToolCallReady { name, arguments } => {
                // Close any open text block before the tool_use block.
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
                // text_open stays false — model continues after tool call
            }
        }
    }
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn new_id(prefix: &str) -> String {
    // Anthropic strict API format requires underscores (e.g. toolu_01A09q...)
    format!("{}_{}", prefix, uuid::Uuid::new_v4().simple())
}
