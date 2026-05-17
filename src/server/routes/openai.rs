//! POST /v1/chat/completions — OpenAI Chat Completions API.
//!
//! Translates the incoming request (tool injection into system prompt for Qwen3),
//! proxies to llama-server, and streams/collects the response.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use std::sync::atomic::Ordering;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::model::backend::GenerateEvent;
use crate::model::registry::ModelRegistry;
use crate::server::error::ApiError;
use crate::server::proxy::{build_chat_body, proxy_request, spawn_tracked_reader};
use crate::server::sse::{data_event, sse_response};
use crate::server::stream_parser::{ParsedEvent, StreamParser};
use crate::server::types::openai::{
    ChatCompletionChunk, ChatCompletionRequest, DeltaMessage, StreamChoice, ToolCallFunction,
    ToolCallInfo,
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
            tracing::error!("Failed to parse ChatCompletionRequest: {}\nRaw Body: {}", e, body_str);
            return Err(anyhow::anyhow!("Invalid request: {}", e).into());
        }
    };

    // If the client requested a specific model that differs from the loaded one,
    // trigger a switch and return 503 so the client retries after load.
    let current = state.current_model_name().await;
    if !ModelRegistry::matches_current(&req.model, &current) {
        if state.trigger_model_switch(&req.model).await {
            return Ok(switching_503());
        }
        // resolve returned None (unknown model) — fall through and serve with current model
    }

    state.touch_last_request();

    let proc = state.process_snapshot().await;
    let cfg = state.config_snapshot().await;
    let model_name = model_name_from_config(&cfg);

    let messages = prepare_messages(&req);
    let url = format!("{}/v1/chat/completions", proc.base_url());
    let temperature = req.temperature.unwrap_or(0.6);
    let top_p = req.top_p.unwrap_or(0.95);
    let top_k = req.top_k.unwrap_or(20);

    if req.stream {
        let body = build_chat_body(&messages, true, req.max_tokens, temperature, top_p, top_k);
        let rx = spawn_tracked_reader(
            proc.http_client(),
            &url,
            body,
            state.metrics.clone(),
        )
        .await?;
        Ok(streaming_response(rx, model_name))
    } else {
        let body = build_chat_body(&messages, false, req.max_tokens, temperature, top_p, top_k);
        // Non-streaming: transparent proxy — llama-server returns OpenAI JSON directly.
        proxy_request(proc.http_client(), &proc.base_url(), "/v1/chat/completions", body, None)
            .await
    }
}

// ── Message preparation ───────────────────────────────────────────────────────

fn prepare_messages(req: &ChatCompletionRequest) -> Vec<(&'static str, String)> {
    let mut messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| {
            let mut text = m.content.as_ref().map(|c| c.clone().into_text()).unwrap_or_default();
            
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
                    ("system".into(), format!("You are a helpful assistant.{injection}")),
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

fn streaming_response(mut rx: crate::model::backend::GenerateStream, model: String) -> Response {
    let id = new_id("chatcmpl");
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
                tool_calls: None,
            },
            finish_reason: None,
        }],
    });
    let _ = tx.send(Ok::<_, Infallible>(first));

    tokio::spawn(async move {
        let mut parser = StreamParser::new();
        let mut tool_called = false;

        let emit_events = |events: Vec<ParsedEvent>,
                           tx: &tokio::sync::mpsc::UnboundedSender<_>| {
            for evt in events {
                match evt {
                    ParsedEvent::TextToken(text) | ParsedEvent::ThinkingToken(text) => {
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
                                    tool_calls: None,
                                },
                                finish_reason: None,
                            }],
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
                        })));
                    }
                }
            }
        };

        while let Some(event) = rx.recv().await {
            match event {
                GenerateEvent::Token(text) => {
                    let evts = parser.push(&text);
                    if evts.iter().any(|e| matches!(e, ParsedEvent::ToolCallReady { .. })) {
                        tool_called = true;
                    }
                    emit_events(evts, &tx);
                }
                GenerateEvent::Done(summary) => {
                    let evts = parser.flush();
                    if evts.iter().any(|e| matches!(e, ParsedEvent::ToolCallReady { .. })) {
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
                    
                    let _ = tx.send(Ok::<_, Infallible>(data_event(&ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_clone.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: DeltaMessage { role: None, content: None, tool_calls: None },
                            finish_reason: Some(finish_reason),
                        }],
                    })));
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
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn new_id(prefix: &str) -> String {
    format!("{}-{}", prefix, uuid::Uuid::new_v4())
}
