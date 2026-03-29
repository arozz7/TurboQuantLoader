/// TurboQuant KV cache — Track B.
///
/// This stub is only compiled when the `turbo-kv` Cargo feature is active.
/// Until `tq-kv` adds Qwen3.5 MoE support, all calls delegate to
/// [`LlamaNativeCache`] with a one-time warning.
///
/// Swapping Track A → B requires zero changes to [`InferenceEngine`] or any
/// caller — the interface is identical.
#[cfg(feature = "turbo-kv")]
pub mod turbo_quant_impl {
    use anyhow::Result;
    use tracing::warn;

    use crate::config::KvCacheConfig;
    use crate::kv_cache::{CacheStats, KvCacheBackend};
    use crate::kv_cache::llama_native::LlamaNativeCache;

    pub struct TurboQuantCache {
        inner: LlamaNativeCache,
    }

    impl TurboQuantCache {
        pub fn new() -> Self {
            Self { inner: LlamaNativeCache::new() }
        }
    }

    impl KvCacheBackend for TurboQuantCache {
        fn configure(&mut self, cfg: &KvCacheConfig) -> Result<()> {
            warn!(
                "TurboQuant KV compression is not yet supported for Qwen3.5 MoE \
                 — falling back to LlamaNative"
            );
            self.inner.configure(cfg)
        }

        fn stats(&self) -> CacheStats {
            self.inner.stats()
        }

        fn reset(&mut self) {
            self.inner.reset()
        }
    }
}
