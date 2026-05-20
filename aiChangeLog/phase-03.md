# Phase 3 — KV Cache Wiring + Bench Command

## Files Created
| File | Purpose |
|------|---------|
| `src/gpu_stats.rs` | NVML-backed GPU telemetry (`query_all_gpus`); no-op stub on non-CUDA builds |
| `src/kv_cache/llama_native.rs` | `LlamaNativeCache` — maps KvBits to Q4_0/Q8_0, tracks compression ratio |
| `src/kv_cache/turbo_quant.rs` | `TurboQuantCache` stub (delegates to `LlamaNativeCache` behind `turbo-kv` feature) |
| `docs/bench_prompt.txt` | Fixed prompt used by the `bench` command |

## Files Modified
| File | Changes |
|------|---------|
| `src/model/backend.rs` | Added `reconfigure_context` to `ModelBackend` trait with default no-op |
| `src/model/llama_cpp.rs` | Added `LlamaCppBackend::load_full(model, kv)`, `rebuild_context()` helper, `ReconfigureContext` command; `context_size` → `AtomicU32`; `apply_kv_cache_config` fix |
| `src/kv_cache/mod.rs` | Rewrote: added `LlamaNativeCache`, `TurboQuantCache` modules, `create_kv_cache()` factory |
| `src/inference/engine.rs` | `NoopKvCache` → `create_kv_cache()`; `create_backend` takes `&AppConfig`; added `reconfigure_context` passthrough |
| `src/main.rs` | Added `mod gpu_stats`; implemented `cmd_bench` (async); imports `KvCacheConfig` |

## Behavior Changes

### `LlamaCppBackend::load_full`
- New constructor that takes both `ModelConfig` and `KvCacheConfig`
- Passes `kv_config` to the inference thread so the initial `LlamaContext` is created with the correct quantization type — no double-rebuild on startup
- `ModelBackend::load` now delegates to `load_full` with `KvCacheConfig::default()`

### `rebuild_context` helper
- Encapsulates `LlamaContext` creation: respects `n_ctx`, `batch_size`, `threads`, and KV type
- Used on startup and for every `ReconfigureContext` command

### `ReconfigureContext` backend command
- Tears down and recreates the `LlamaContext` in-thread; model weights stay loaded
- Returns the actual `n_ctx` reported by llama.cpp back to the caller
- `LlamaCppBackend.context_size` is an `AtomicU32` so `reconfigure_context(&self, ...)` can update it without `&mut self`

### `cmd_bench`
- Loads the model once, then sweeps all `(context_size × kv_bits)` combinations
- Uses `tokio::task::block_in_place` for blocking `reconfigure_context` calls
- Prints an aligned table; optionally writes JSON via `--output`

## Assumptions and Risks
- `apply_kv_cache_config` is currently unused at the call site (InferenceEngine uses `load_full`). Retained in the trait for symmetry with Phase 4 runtime reconfiguration.
- Dead-code warnings for `gpu_stats`, `KvCacheBackend` methods, and `collect_full` are expected — these APIs land in Phase 4 (HTTP server stats endpoint).
- `TurboQuantCache` falls back to `LlamaNativeCache` with a warning until `tq-kv` adds Qwen3.5 MoE support.
