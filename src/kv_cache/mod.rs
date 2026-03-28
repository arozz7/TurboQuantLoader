use anyhow::Result;

use crate::config::KvCacheConfig;

/// Live statistics from the active KV cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Memory currently consumed by the KV cache in megabytes.
    pub used_mb: f32,
    /// Compression ratio relative to an FP16 baseline (`1.0` = no compression).
    pub compression_ratio: f32,
    /// Number of tokens currently stored in the context.
    pub context_tokens: u32,
    /// Number of full-attention layers in the model.
    ///
    /// For Qwen3.5-35B-A3B this is `10` (the remaining 30 layers use linear
    /// attention and do not contribute to the KV cache).
    pub full_attention_layers: u32,
}

/// Abstraction over KV cache backends.
///
/// Both [`LlamaNativeCache`](crate::kv_cache::llama_native::LlamaNativeCache)
/// (Track A, Phase 3) and
/// [`TurboQuantCache`](crate::kv_cache::turbo_quant::TurboQuantCache)
/// (Track B, `turbo-kv` feature, Phase 3) implement this trait. The active
/// backend is selected at startup via [`create_kv_cache`](crate::kv_cache::create_kv_cache);
/// all callers depend only on this trait.
pub trait KvCacheBackend: Send + Sync {
    /// Apply quantization settings and memory budget from `cfg`.
    ///
    /// Must be called before the first inference request.
    fn configure(&mut self, cfg: &KvCacheConfig) -> Result<()>;

    /// Return a snapshot of current cache memory usage and compression stats.
    fn stats(&self) -> CacheStats;

    /// Evict all cached KV entries, equivalent to clearing the conversation context.
    fn reset(&mut self);
}
