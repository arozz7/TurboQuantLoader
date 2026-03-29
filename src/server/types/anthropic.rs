//! Anthropic Messages API — request and response types.
//!
//! Reference: <https://docs.anthropic.com/en/api/messages>

use serde::{Deserialize, Serialize};

// ── Request ───────────────────────────────────────────────────────────────────

/// Content of an Anthropic message — plain string or list of content blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl AnthropicContent {
    /// Flatten to a plain string, joining all text blocks in order.
    ///
    /// Non-text blocks are silently skipped. Use [`into_qwen3_text`] when
    /// the message may contain `tool_use` or `tool_result` blocks.
    pub fn into_text(self) -> String {
        match self {
            Self::Text(s) => s,
            Self::Blocks(blocks) => blocks
                .into_iter()
                .filter_map(|b| if b.r#type == "text" { Some(b.text) } else { None })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Flatten to a Qwen3.5-formatted string, converting `tool_use` blocks to
    /// `<tool_call>` markup and `tool_result` blocks to `<tool_response>` markup.
    pub fn into_qwen3_text(self) -> String {
        match self {
            Self::Text(s) => s,
            Self::Blocks(blocks) => {
                let mut parts = Vec::new();
                for b in blocks {
                    match b.r#type.as_str() {
                        "text" => {
                            if !b.text.is_empty() {
                                parts.push(b.text);
                            }
                        }
                        "tool_use" => {
                            let call = serde_json::json!({
                                "name": b.name,
                                "arguments": b.input,
                            });
                            let json = serde_json::to_string(&call).unwrap_or_default();
                            parts.push(format!("<tool_call>\n{json}\n</tool_call>"));
                        }
                        "tool_result" => {
                            let text = match &b.content {
                                Some(serde_json::Value::String(s)) => s.clone(),
                                Some(serde_json::Value::Array(arr)) => arr
                                    .iter()
                                    .filter_map(|v| {
                                        if v.get("type").and_then(|t| t.as_str()) == Some("text") {
                                            v.get("text").and_then(|t| t.as_str()).map(str::to_string)
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join(""),
                                Some(other) => other.to_string(),
                                None => b.text.clone(),
                            };
                            parts.push(format!("<tool_response>\n{text}\n</tool_response>"));
                        }
                        _ => {}
                    }
                }
                parts.join("\n")
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlock {
    pub r#type: String,
    /// Present on `text` blocks (and as fallback content in `tool_result`).
    #[serde(default)]
    pub text: String,
    /// Present on `tool_use` blocks — tool call identifier.
    #[serde(default)]
    pub id: String,
    /// Present on `tool_use` blocks — the tool name.
    #[serde(default)]
    pub name: String,
    /// Present on `tool_use` blocks — tool input arguments.
    #[serde(default)]
    pub input: serde_json::Value,
    /// Present on `tool_result` blocks — the id of the `tool_use` this answers.
    #[serde(default)]
    pub tool_use_id: String,
    /// Present on `tool_result` blocks — the result content (string or block array).
    #[serde(default)]
    pub content: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    /// Ignored — model is always the one loaded in config.
    #[allow(dead_code)]
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    /// Optional system prompt — string or content-block array.
    ///
    /// Claude Code v2.1+ sends this as `[{"type":"text","text":"..."}]`.
    /// Older clients send a plain string. Both are accepted.
    pub system: Option<AnthropicContent>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    /// Tool definitions sent by Claude Code — not yet forwarded to the model.
    /// Captured here so the handler can log and inspect them.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
}

fn default_max_tokens() -> u32 {
    2048
}

// ── Non-streaming response ────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct InputUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct TextBlock {
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub content: Vec<TextBlock>,
    pub model: String,
    pub stop_reason: &'static str,
    pub stop_sequence: Option<String>,
    pub usage: InputUsage,
}

// ── Streaming event types ─────────────────────────────────────────────────────

// ── Thinking block types ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ThinkingBlockStartData {
    pub r#type: &'static str, // "thinking"
    /// Always empty string — content arrives via thinking_delta events.
    pub thinking: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ThinkingDelta {
    pub r#type: &'static str, // "thinking_delta"
    pub thinking: String,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockDeltaThinkingEvent {
    pub r#type: &'static str, // "content_block_delta"
    pub index: u32,
    pub delta: ThinkingDelta,
}

// ── Tool use block types ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ToolUseBlockStartData {
    pub r#type: &'static str, // "tool_use"
    pub id: String,
    pub name: String,
    /// Always empty object — input arrives via input_json_delta events.
    pub input: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct InputJsonDelta {
    pub r#type: &'static str, // "input_json_delta"
    pub partial_json: String,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockDeltaInputJsonEvent {
    pub r#type: &'static str, // "content_block_delta"
    pub index: u32,
    pub delta: InputJsonDelta,
}

#[derive(Debug, Serialize)]
pub struct MessageStartEvent {
    pub r#type: &'static str,
    pub message: MessageStartData,
}

#[derive(Debug, Serialize)]
pub struct MessageStartData {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub content: Vec<serde_json::Value>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: InputUsage,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockStartEvent {
    pub r#type: &'static str,
    pub index: u32,
    pub content_block: ContentBlockStartData,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockStartData {
    pub r#type: &'static str,
    pub text: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockDeltaEvent {
    pub r#type: &'static str,
    pub index: u32,
    pub delta: TextDelta,
}

#[derive(Debug, Serialize)]
pub struct TextDelta {
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockStopEvent {
    pub r#type: &'static str,
    pub index: u32,
}

#[derive(Debug, Serialize)]
pub struct MessageDeltaEvent {
    pub r#type: &'static str,
    pub delta: MessageDeltaData,
    pub usage: OutputUsage,
}

#[derive(Debug, Serialize)]
pub struct MessageDeltaData {
    pub stop_reason: &'static str,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OutputUsage {
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct MessageStopEvent {
    pub r#type: &'static str,
}

#[derive(Debug, Serialize)]
pub struct PingEvent {
    pub r#type: &'static str,
}
