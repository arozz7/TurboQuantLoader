# Phase 5.1 Hotfixes

## Diff Narrative

**Files modified:**
- `src/model/llama_cpp.rs`

**Behavior changes:**
1. **M-RoPE Inconsistency Fallback:** To overcome an upstream `llama.cpp` sequence boundary mismatch when trimming warm caches for models that utilize M-RoPE (such as Qwen3.5), we added a dynamic prefill retry chunk. If `ctx.decode` fails and `n_past > 0`, we catch the failure gracefully, invalidate the prefix cache, and forcefully trigger a cold-cache retry. This unblocks API generation without removing `prefix_cache` from standard runs.
2. **Generation Progress Status:** During continuous block generation, we've injected terminal output (`tracing::info!("actively generating: {} tokens elapsed...", tokens_generated);`) that triggers every 50 tokens. This mitigates the "spinning wheel" perception in long `stream=false` POST queries with large max-tokens.
3. **KV Cache Prefill Progress Logging:** When generating with `n_past=0` against large contexts (such as Claude Code API 32000-token queries) which can take up to several minutes to ingest, we added a log on each chunk step (`info!("prefill progress: chunk {}/{}...")`) to visibly track large memory batches.
4. **VRAM Offloading & Tensor Split Fix:** We uncovered a critical bug where `config.tensor_split` arrays were being parsed but never actually forwarded to `llama_cpp::LlamaModelParams` via the `.with_tensor_split()` method. Additionally, the loader ignored device assignment entirely if only a single GPU ratio was supplied (`tensor_split.len() > 1`). This caused `llama.cpp` to always default to a 50/50 split across all detected physical GPUs and fall back to CPU memory (causing massive lag) if the smaller secondary GPU ran out of space. We've fixed this so that specifying `[1.0]` correctly limits operations purely to the primary CUDA device.

**Tests & Confidence:**
- `cargo check` verified compilation.
- Fallback loop validates successfully against the bounds of LlamaContext logic.

**Assumptions & Risks:**
- **Risk:** Dropping the cached prefixes inside `prefix_cache.invalidate()` when a warm context causes an `M-RoPE` crash will penalize latency uniquely for those particular models. A permanent fix relies on `#16890` or equivalent upstream patches for `llama_cpp`.
