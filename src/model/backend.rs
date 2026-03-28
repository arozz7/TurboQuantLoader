use anyhow::Result;
use tokio::sync::mpsc;

use crate::config::{KvCacheConfig, ModelConfig};

/// Parameters controlling the sampling strategy during text generation.
///
/// Passed as part of [`GenerateRequest`]. Phase 2 moves this to
/// `inference/sampler.rs`; callers always import it from `model::backend`
/// until that refactor.
#[derive(Debug, Clone)]
pub struct SamplerParams {
    /// Softmax temperature — higher values increase randomness (default: `0.7`).
    pub temperature: f32,
    /// Nucleus sampling probability threshold (default: `0.9`).
    pub top_p: f32,
    /// Top-K sampling — keep only the K most likely tokens (default: `40`).
    pub top_k: u32,
    /// Min-P sampling floor (default: `0.05`).
    pub min_p: f32,
    /// Penalty applied to recently generated tokens to reduce repetition (default: `1.1`).
    pub repeat_penalty: f32,
    /// Number of recent tokens considered for the repeat penalty (default: `64`).
    pub repeat_last_n: u32,
    /// Fixed RNG seed for reproducible outputs. `None` uses a random seed.
    pub seed: Option<u64>,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: None,
        }
    }
}

/// A request to generate text from a prompt.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Fully formatted prompt string (chat template applied by the caller).
    pub prompt: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sampling configuration for this request.
    pub sampler: SamplerParams,
    /// Generation halts when any of these strings appears in the output.
    pub stop_strings: Vec<String>,
}

/// A single event emitted by a generation stream.
#[derive(Debug)]
pub enum GenerateEvent {
    /// A decoded text fragment (one or more unicode characters).
    Token(String),
    /// Generation finished successfully; carries timing and usage stats.
    Done(GenerateSummary),
    /// Generation stopped due to an error; the string is a human-readable message.
    Error(String),
}

/// End-of-generation statistics.
#[derive(Debug, Clone)]
pub struct GenerateSummary {
    /// Total number of tokens produced.
    pub tokens_generated: u32,
    /// Average generation throughput in tokens per second.
    pub tokens_per_second: f32,
    /// Number of tokens in the full context at completion (prompt + generated).
    pub context_tokens: u32,
}

/// Async channel receiver that delivers [`GenerateEvent`]s from the backend.
///
/// The channel is closed after a `Done` or `Error` variant is sent.
pub type GenerateStream = mpsc::Receiver<GenerateEvent>;

/// Core abstraction over inference backends.
///
/// Implemented by [`LlamaCppBackend`] in Phase 2. All callers depend on this
/// trait rather than on a concrete type, enabling backend swaps without
/// cascading changes.
pub trait ModelBackend: Send + Sync {
    /// Load the model described by `config` and return a ready-to-use backend.
    ///
    /// This is a blocking operation and should be called outside of async
    /// contexts (or wrapped in `tokio::task::spawn_blocking`).
    fn load(config: &ModelConfig) -> Result<Self>
    where
        Self: Sized;

    /// Encode `text` into a sequence of token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;

    /// Decode token IDs back to a UTF-8 string.
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;

    /// Maximum context window size supported by this model instance (in tokens).
    fn context_size(&self) -> u32;

    /// Human-readable model name, e.g. `"Qwen3.5-35B-A3B-UD-IQ3_XXS"`.
    fn model_name(&self) -> &str;

    /// Begin token generation for `req` and return a stream of [`GenerateEvent`]s.
    ///
    /// Generation runs on a background thread; tokens are forwarded through an
    /// `mpsc` channel. The stream is closed when a `Done` or `Error` event is sent.
    fn generate(&self, req: GenerateRequest) -> Result<GenerateStream>;

    /// Apply KV cache quantization settings.
    ///
    /// Must be called before the first [`generate`](Self::generate) invocation.
    fn apply_kv_cache_config(&mut self, cfg: &KvCacheConfig) -> Result<()>;
}
