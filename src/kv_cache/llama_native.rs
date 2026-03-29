use anyhow::Result;
use tracing::warn;

use crate::config::{KvBits, KvCacheConfig};

use super::{CacheStats, KvCacheBackend};

/// llama.cpp native KV cache — Track A.
///
/// The actual quantization type (`Q4_0` / `Q8_0`) is applied inside the
/// inference thread when the [`LlamaContext`] is created. This struct holds the
/// configured bit-width so [`stats`](Self::stats) can compute the compression
/// ratio, and is updated via [`configure`](Self::configure) whenever the
/// inference thread is reconfigured.
pub struct LlamaNativeCache {
    bits: KvBits,
}

impl LlamaNativeCache {
    pub fn new() -> Self {
        Self { bits: KvBits::Four }
    }
}

impl KvCacheBackend for LlamaNativeCache {
    fn configure(&mut self, cfg: &KvCacheConfig) -> Result<()> {
        match cfg.bits {
            KvBits::Two | KvBits::Three => {
                warn!(
                    bits = u8::from(cfg.bits),
                    "llama.cpp has no native {}-bit KV type — using Q4_0 (4-bit)",
                    u8::from(cfg.bits)
                );
            }
            _ => {}
        }
        self.bits = cfg.bits;
        Ok(())
    }

    fn stats(&self) -> CacheStats {
        // FP16 baseline = 2 bytes/element.
        // compression_ratio = fp16_size / quantized_size.
        let compression_ratio = match self.bits {
            KvBits::Two | KvBits::Three => 4.0, // falls back to Q4_0
            KvBits::Four => 4.0,                // Q4_0 = 0.5 bytes/element
            KvBits::Eight => 2.0,               // Q8_0 = 1 byte/element
        };

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
