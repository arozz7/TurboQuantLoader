use anyhow::Result;
use tracing::info;

use crate::config::{AppConfig, KvCacheConfig};
use crate::kv_cache::{create_kv_cache, KvCacheBackend};
use crate::model::backend::{GenerateRequest, ModelBackend, SamplerParams};

use super::stream::TokenStream;

// ── Chat types ────────────────────────────────────────────────────────────────

/// A single message in a conversation (role + content).
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Plain-text message body.
    pub content: String,
}

/// A request to generate the next assistant turn.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// Full conversation history including the new user message.
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate for this turn.
    pub max_tokens: u32,
    /// Sampling parameters for this request.
    pub sampler: SamplerParams,
}

// ── InferenceEngine ───────────────────────────────────────────────────────────

/// Orchestrates model loading, chat-template formatting, and streaming generation.
///
/// Callers depend on `InferenceEngine` rather than on a concrete backend type.
/// The backend is selected at construction time via [`create_backend`].
pub struct InferenceEngine {
    backend: Box<dyn ModelBackend>,
    /// KV cache metadata tracker — used by Phase 4 stats endpoints.
    _kv_cache: Box<dyn KvCacheBackend>,
}

impl InferenceEngine {
    /// Load the model described by `config` and return a ready engine.
    ///
    /// This is a **blocking** call (model weights are loaded from disk). Wrap in
    /// `tokio::task::spawn_blocking` when calling from an async context.
    pub fn new(config: AppConfig) -> Result<Self> {
        let kv_cache = create_kv_cache(&config.kv_cache);
        let backend = create_backend(&config)?;
        info!(model = backend.model_name(), "inference engine ready");

        Ok(Self { backend, _kv_cache: kv_cache })
    }

    /// Begin generating the next assistant turn for `req`.
    ///
    /// Returns a [`TokenStream`] that yields tokens as they are produced.
    pub fn chat(&self, req: ChatRequest) -> Result<TokenStream> {
        let prompt = format_messages(&req.messages);
        let gen_req = GenerateRequest {
            prompt,
            max_tokens: req.max_tokens,
            sampler: req.sampler,
            stop_strings: vec!["<|im_end|>".into()],
        };
        let rx = self.backend.generate(gen_req)?;
        Ok(TokenStream::new(rx))
    }

    /// Model name reported by the backend.
    pub fn model_name(&self) -> &str {
        self.backend.model_name()
    }

    /// Context window size in tokens.
    pub fn context_size(&self) -> u32 {
        self.backend.context_size()
    }

    /// Tear down and recreate the inference context with new parameters.
    ///
    /// Model weights remain loaded. Used by the `bench` command to sweep
    /// (context_size × kv_bits) combinations without reloading the model.
    pub fn reconfigure_context(&self, n_ctx: u32, kv_cfg: &KvCacheConfig) -> Result<()> {
        self.backend.reconfigure_context(n_ctx, kv_cfg)
    }
}

// ── Chat template ─────────────────────────────────────────────────────────────

/// Apply the ChatML template used by Qwen3.5 (and most instruction-tuned models).
///
/// Format:
/// ```text
/// <|im_start|>system
/// You are a helpful assistant.<|im_end|>
/// <|im_start|>user
/// Hello<|im_end|>
/// <|im_start|>assistant
/// ```
fn format_messages(messages: &[ChatMessage]) -> String {
    let mut out = String::new();

    // Prepend a default system prompt if the first message is not a system message.
    let has_system = messages.first().map(|m| m.role == "system").unwrap_or(false);
    if !has_system {
        out.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    }

    for msg in messages {
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }

    // Prompt the model to begin its response.
    // Thinking tokens (<think>...</think>) are parsed and streamed as Anthropic
    // thinking content blocks by the server layer — do not suppress them here.
    out.push_str("<|im_start|>assistant\n");
    out
}

// ── Backend factory ───────────────────────────────────────────────────────────

/// Instantiate the appropriate [`ModelBackend`] for the active feature set.
#[cfg(feature = "llama-backend")]
fn create_backend(config: &AppConfig) -> Result<Box<dyn ModelBackend>> {
    use crate::model::llama_cpp::LlamaCppBackend;
    Ok(Box::new(LlamaCppBackend::load_full(&config.model, &config.kv_cache)?))
}

#[cfg(not(feature = "llama-backend"))]
fn create_backend(_config: &AppConfig) -> Result<Box<dyn ModelBackend>> {
    anyhow::bail!(
        "No inference backend is enabled.\n\
         Build with one of:\n\
         \t--features llama-backend   (CPU-only inference)\n\
         \t--features cuda            (NVIDIA GPU)\n\
         \t--features metal           (Apple GPU)\n\
         \t--features vulkan          (cross-platform GPU)"
    )
}
