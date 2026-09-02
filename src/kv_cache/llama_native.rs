use anyhow::Result;

use crate::config::{KvCacheConfig, KvType};

use super::{CacheStats, KvCacheBackend};

/// FP16 KV cache baseline, in bytes per element — used as the reference point
/// for [`KvCacheBackend::stats`] compression-ratio reporting.
const FP16_BYTES_PER_ELEMENT: f32 = 2.0;

/// llama.cpp native KV cache — Track A.
///
/// The actual quantization type is applied inside the inference thread when
/// the [`LlamaContext`] is created. This struct holds the configured K/V
/// types so [`stats`](Self::stats) can compute the compression ratio, and is
/// updated via [`configure`](Self::configure) whenever the inference thread
/// is reconfigured.
pub struct LlamaNativeCache {
    #[allow(dead_code)]
    type_k: KvType,
    #[allow(dead_code)]
    type_v: KvType,
}

impl LlamaNativeCache {
    pub fn new() -> Self {
        Self {
            type_k: KvType::F16,
            type_v: KvType::F16,
        }
    }
}

impl KvCacheBackend for LlamaNativeCache {
    fn configure(&mut self, cfg: &KvCacheConfig) -> Result<()> {
        self.type_k = cfg.type_k;
        self.type_v = cfg.type_v;
        Ok(())
    }

    fn stats(&self) -> CacheStats {
        // compression_ratio = fp16_size / average(K type size, V type size).
        let avg_bytes_per_element =
            (self.type_k.bytes_per_element() + self.type_v.bytes_per_element()) / 2.0;
        let compression_ratio = FP16_BYTES_PER_ELEMENT / avg_bytes_per_element;

        CacheStats {
            // Phase 4 TODO: read actual KV buffer bytes from llama_state_size.
            used_mb: 0.0,
            compression_ratio,
            context_tokens: 0,
            // Qwen3.5-35B-A3B: every 4th layer is full-attention, 10 out of 40 total.
            full_attention_layers: 10,
        }
    }

    fn reset(&mut self) {
        // KV eviction is handled by the inference thread via ctx.clear_kv_cache().
    }
}
