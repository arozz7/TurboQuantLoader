//! Shared SSE response builder for OpenAI and Anthropic streaming handlers.

use std::convert::Infallible;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::Stream;

/// Wrap any `Stream<Item = Result<Event, Infallible>>` into a keep-alive SSE `Response`.
///
/// A 15-second keep-alive ping is added automatically to prevent proxy
/// timeouts on long-running generations.
pub fn sse_response(
    stream: impl Stream<Item = Result<Event, Infallible>> + Send + 'static,
) -> Response {
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Serialise `value` to a JSON string and wrap it in an SSE `data:` line.
///
/// Panics if serialisation fails (only possible for non-serialisable types,
/// which never appear here).
pub fn data_event(value: &impl serde::Serialize) -> Event {
    Event::default().data(
        serde_json::to_string(value).expect("serialisation of SSE payload must not fail"),
    )
}

/// Same as [`data_event`] but also sets the SSE `event:` name field.
///
/// Used by the Anthropic streaming handler which names every event type.
pub fn named_event(name: &'static str, value: &impl serde::Serialize) -> Event {
    Event::default()
        .event(name)
        .data(serde_json::to_string(value).expect("serialisation of SSE payload must not fail"))
}

/// Build a `content_block_start` event for a thinking block at `index`.
pub fn thinking_block_start_event(index: u32) -> Event {
    // Emit raw JSON directly — we need a thinking variant of content_block_start
    // which doesn't match the existing text ContentBlockStartData struct.
    let json = serde_json::json!({
        "type": "content_block_start",
        "index": index,
        "content_block": { "type": "thinking", "thinking": "" }
    });
    Event::default()
        .event("content_block_start")
        .data(json.to_string())
}

/// Build a `content_block_delta` event carrying a thinking token.
pub fn thinking_delta_event(index: u32, thinking: String) -> Event {
    use crate::server::types::anthropic::{ContentBlockDeltaThinkingEvent, ThinkingDelta};
    named_event(
        "content_block_delta",
        &ContentBlockDeltaThinkingEvent {
            r#type: "content_block_delta",
            index,
            delta: ThinkingDelta { r#type: "thinking_delta", thinking },
        },
    )
}

/// Build a `content_block_start` event for a tool_use block.
pub fn tool_use_block_start_event(index: u32, id: String, name: String) -> Event {
    let json = serde_json::json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": {}
        }
    });
    Event::default()
        .event("content_block_start")
        .data(json.to_string())
}

/// Build a `content_block_delta` event carrying an input_json_delta for tool use.
pub fn input_json_delta_event(index: u32, partial_json: String) -> Event {
    use crate::server::types::anthropic::{ContentBlockDeltaInputJsonEvent, InputJsonDelta};
    named_event(
        "content_block_delta",
        &ContentBlockDeltaInputJsonEvent {
            r#type: "content_block_delta",
            index,
            delta: InputJsonDelta { r#type: "input_json_delta", partial_json },
        },
    )
}
