//! OpenAI Chat Completions API — request and response types.
//!
//! Reference: <https://platform.openai.com/docs/api-reference/chat>

use serde::{Deserialize, Serialize};

// ── Request ───────────────────────────────────────────────────────────────────

/// Content of a chat message — plain string or list of content parts.
///
/// Both wire shapes are accepted; all text parts are concatenated for
/// inference. Non-text parts (images etc.) are silently dropped.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
    Unknown(serde_json::Value),
}

impl MessageContent {
    /// Flatten to a plain string, joining all text parts in order.
    pub fn into_text(self) -> String {
        match self {
            Self::Text(s) => s,
            Self::Parts(parts) => parts
                .into_iter()
                .filter_map(|p| if p.r#type == "text" { p.text } else { None })
                .collect::<Vec<_>>()
                .join(""),
            Self::Unknown(val) => {
                if val.is_null() {
                    String::new()
                } else if let Some(s) = val.as_str() {
                    s.to_string()
                } else {
                    serde_json::to_string(&val).unwrap_or_default()
                }
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentPart {
    pub r#type: String,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<MessageContent>,
    pub tool_calls: Option<Vec<ToolCallInfo>>,
    #[allow(dead_code)]
    pub tool_call_id: Option<String>,
    #[allow(dead_code)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Requested model name. Triggers a hot-swap when it differs from the loaded model.
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(alias = "max_completion_tokens")]
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    #[allow(dead_code)]
    pub seed: Option<u64>,
    pub tools: Option<Vec<serde_json::Value>>,
    /// OpenAI streaming convention: when `include_usage` is set, an extra
    /// final chunk with empty `choices` and a populated `usage` object is
    /// sent before `[DONE]`. Opt-in — clients that don't ask for it (like our
    /// own coding-agent) never see the extra chunk, since some naive parsers
    /// assume `choices` is never empty.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

fn default_max_tokens() -> u32 {
    32768
}

// ── Non-streaming response ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    pub content: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: &'static str,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

// ── Streaming chunk ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    #[serde(default)]
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Serialize)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Chain-of-thought text, streamed separately from `content` per the
    /// reasoning-model convention used by llama.cpp, vLLM, LM Studio, and
    /// OpenRouter/DeepSeek — lets clients that declare reasoning support
    /// (e.g. a `reasoning: true` model entry) distinguish "still thinking"
    /// from the actual answer instead of receiving both interleaved in
    /// `content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallInfo>>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: DeltaMessage,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    /// Only populated on the final usage-only chunk (empty `choices`) sent
    /// when the request set `stream_options.include_usage`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}
