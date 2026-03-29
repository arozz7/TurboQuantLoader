//! POST /v1/chat/completions — OpenAI Chat Completions API.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;

use crate::inference::engine::{ChatMessage, ChatRequest};
use crate::model::backend::{GenerateEvent, SamplerParams};
use crate::server::error::ApiError;
use crate::server::sse::{data_event, sse_response};
use crate::server::types::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, DeltaMessage,
    ResponseMessage, StreamChoice, Usage,
};
use crate::server::AppState;

/// `POST /v1/chat/completions`
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let chat_req = to_chat_request(&req);

    if req.stream {
        let stream = tokio::task::block_in_place(|| state.engine.chat(chat_req))?;
        Ok(streaming_response(stream.into_inner(), state.engine.model_name().to_string()))
    } else {
        let stream = tokio::task::block_in_place(|| state.engine.chat(chat_req))?;
        let (text, summary) = stream
            .collect_full()
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(full_response(text, summary.context_tokens, summary.tokens_generated, state.engine.model_name()))
    }
}

// ── Request translation ───────────────────────────────────────────────────────

fn to_chat_request(req: &ChatCompletionRequest) -> ChatRequest {
    let messages = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone().into_text(),
        })
        .collect();

    ChatRequest {
        messages,
        max_tokens: req.max_tokens,
        sampler: SamplerParams {
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            top_k: req.top_k.unwrap_or(40),
            seed: req.seed,
            ..SamplerParams::default()
        },
    }
}

// ── Non-streaming response ────────────────────────────────────────────────────

fn full_response(text: String, context_tokens: u32, output_tokens: u32, model: &str) -> Response {
    let prompt_tokens = context_tokens.saturating_sub(output_tokens);
    let resp = ChatCompletionResponse {
        id: new_id("chatcmpl"),
        object: "chat.completion",
        created: unix_now(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage { role: "assistant", content: text },
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: output_tokens,
            total_tokens: context_tokens,
        },
    };
    Json(resp).into_response()
}

// ── Streaming response ────────────────────────────────────────────────────────

fn streaming_response(
    rx: crate::model::backend::GenerateStream,
    model: String,
) -> Response {
    let id = new_id("chatcmpl");
    let created = unix_now();

    // First chunk announces the assistant role.
    let first = data_event(&ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model.clone(),
        choices: vec![StreamChoice {
            index: 0,
            delta: DeltaMessage { role: Some("assistant"), content: Some(String::new()) },
            finish_reason: None,
        }],
    });

    let token_stream = ReceiverStream::new(rx).filter_map(move |event| match event {
        GenerateEvent::Token(text) => Some(Ok::<_, Infallible>(data_event(&ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: DeltaMessage { role: None, content: Some(text) },
                finish_reason: None,
            }],
        }))),
        GenerateEvent::Done(_) => {
            // [DONE] sentinel expected by OpenAI clients.
            Some(Ok(axum::response::sse::Event::default().data("[DONE]")))
        }
        GenerateEvent::Error(e) => {
            tracing::warn!(error = %e, "generation error during OpenAI stream");
            None
        }
    });

    let full_stream =
        tokio_stream::iter([Ok::<_, Infallible>(first)]).chain(token_stream);

    sse_response(full_stream)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn new_id(prefix: &str) -> String {
    format!("{}-{}", prefix, uuid::Uuid::new_v4())
}
