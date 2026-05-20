use anyhow::Result;
use tracing::warn;

use crate::config::{KvCacheConfig, KvStrategy};

pub mod llama_native;
pub mod prefix_cache;
pub mod turbo_quant;

pub use llama_native::LlamaNativeCache;
#[allow(unused_imports)]
pub use prefix_cache::PrefixCache;

/// Live statistics from the active KV cache.
#[allow(dead_code)]
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
#[allow(dead_code)]
pub trait KvCacheBackend: Send + Sync {
    /// Apply quantization settings and memory budget from `cfg`.
    fn configure(&mut self, cfg: &KvCacheConfig) -> Result<()>;

    /// Return a snapshot of current cache memory usage and compression stats.
    fn stats(&self) -> CacheStats;

    /// Evict all cached KV entries, equivalent to clearing the conversation context.
    fn reset(&mut self);
}

/// Instantiate the appropriate [`KvCacheBackend`] for the configured strategy.
///
/// Falls back to [`LlamaNativeCache`] with a warning when the `turbo-kv` feature
/// is requested but not compiled in.
pub fn create_kv_cache(cfg: &KvCacheConfig) -> Box<dyn KvCacheBackend> {
    match cfg.strategy {
        KvStrategy::LlamaNative => Box::new(LlamaNativeCache::new()),
        KvStrategy::TurboQuant => {
            #[cfg(feature = "turbo-kv")]
            {
                use turbo_quant::turbo_quant_impl::TurboQuantCache;
                Box::new(TurboQuantCache::new())
            }
            #[cfg(not(feature = "turbo-kv"))]
            {
                warn!(
                    "KV strategy 'turbo_quant' requested but the `turbo-kv` feature is not \
                     compiled in — using LlamaNative instead. \
                     Rebuild with --features turbo-kv once Qwen3.5 MoE support lands."
                );
                Box::new(LlamaNativeCache::new())
            }
        }
    }
}
