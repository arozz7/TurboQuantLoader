use anyhow::Result;
use tracing::info;

use crate::config::{AppConfig, KvCacheConfig};
use crate::config::ModelConfig;
use crate::kv_cache::{CacheStats, KvCacheBackend};
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

// ── NoopKvCache ───────────────────────────────────────────────────────────────

/// Phase 2 stub — does nothing. Replaced by `LlamaNativeCache` in Phase 3.
pub struct NoopKvCache;

impl KvCacheBackend for NoopKvCache {
    fn configure(&mut self, _cfg: &KvCacheConfig) -> Result<()> {
        Ok(())
    }

    fn stats(&self) -> CacheStats {
        CacheStats::default()
    }

    fn reset(&mut self) {}
}

// ── InferenceEngine ───────────────────────────────────────────────────────────

/// Orchestrates model loading, chat-template formatting, and streaming generation.
///
/// Callers depend on `InferenceEngine` rather than on a concrete backend type.
/// The backend is selected at construction time via [`create_backend`].
pub struct InferenceEngine {
    backend: Box<dyn ModelBackend>,
    _kv_cache: Box<dyn KvCacheBackend>, // Phase 3: replaced by LlamaNativeCache
}

impl InferenceEngine {
    /// Load the model described by `config` and return a ready engine.
    ///
    /// This is a **blocking** call (model weights are loaded from disk). Wrap in
    /// `tokio::task::spawn_blocking` when calling from an async context.
    pub fn new(config: AppConfig) -> Result<Self> {
        let backend = create_backend(&config.model)?;
        info!(model = backend.model_name(), "inference engine ready");

        Ok(Self {
            backend,
            _kv_cache: Box::new(NoopKvCache),
        })
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
    out.push_str("<|im_start|>assistant\n");
    out
}

// ── Backend factory ───────────────────────────────────────────────────────────

/// Instantiate the appropriate [`ModelBackend`] for the active feature set.
#[cfg(feature = "llama-backend")]
fn create_backend(config: &ModelConfig) -> Result<Box<dyn ModelBackend>> {
    use crate::model::llama_cpp::LlamaCppBackend;
    Ok(Box::new(LlamaCppBackend::load(config)?))
}

#[cfg(not(feature = "llama-backend"))]
fn create_backend(_config: &ModelConfig) -> Result<Box<dyn ModelBackend>> {
    anyhow::bail!(
        "No inference backend is enabled.\n\
         Build with one of:\n\
         \t--features llama-backend   (CPU-only inference)\n\
         \t--features cuda            (NVIDIA GPU)\n\
         \t--features metal           (Apple GPU)\n\
         \t--features vulkan          (cross-platform GPU)"
    )
}
