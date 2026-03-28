# Phase 02 — Inference Engine

**Status:** Complete
**Date Started:** 2026-03-28
**Date Completed:** 2026-03-28

## Goal
Model loads on both GPUs, tokens stream to terminal, `run` command works interactively.
No HTTP server yet; no benchmarking. Just end-to-end inference from the REPL.

## Files Created
- `src/inference/sampler.rs` — `build_sampler()` (behind `llama-backend` feature)
- `src/inference/stream.rs` — `TokenStream` with `next_event()` / `collect_full()`
- `src/inference/engine.rs` — `InferenceEngine`, `ChatMessage`, `ChatRequest`, `NoopKvCache` stub
- `src/model/llama_cpp.rs` — `LlamaCppBackend` (behind `llama-backend` feature)
- `docs/testing.md` — smoke test checklist
- `aiChangeLog/phase-02.md` — this file

## Files Modified
- `src/inference/mod.rs` — declared `engine`, `sampler`, `stream` submodules
- `src/model/mod.rs` — added `llama_cpp` module (cfg-gated)
- `src/main.rs` — `#[tokio::main]` async entry point; `cmd_run` wired to async REPL

## Architecture Decisions

### Dedicated inference thread
`LlamaContext<'_>` is `!Send` (raw pointer + model lifetime). Rather than fighting
the type system, we spawn a named `std::thread` ("llama-inference") that owns both
the `LlamaBackend`, `LlamaModel`, and `LlamaContext` for the process lifetime.

Callers send `BackendCommand` via `std::sync::mpsc::SyncSender` (backpressure = 4).
Each generation streams tokens back through a `tokio::sync::mpsc::Sender<GenerateEvent>`.

### Startup handshake
A one-shot `mpsc::sync_channel` carries `Result<(model_name, context_size)>` from
the inference thread back to `LlamaCppBackend::load()`. Errors during model load
propagate cleanly to the caller.

### Context reuse
One `LlamaContext` is created at thread startup and reused for all requests.
`ctx.clear_kv_cache()` is called between requests to reset state.

### KV cache type
Hardcoded to Q4_0 for Phase 2. Phase 3 will thread the `KvCacheConfig` through
to `ctx_params.with_type_k / with_type_v`.

### Multi-GPU (tensor split)
`LlamaModelParams::with_devices(&[0, 1])` selects both GPUs.
The Rust API does not expose tensor split weights — VRAM-proportional balancing
is handled automatically by ggml's backend scheduler.

### Chat template
Manual ChatML formatting in `inference::engine::format_messages`.
A default system prompt is injected if the conversation has no system message.
Stop string: `"<|im_end|>"`.

## llama-cpp-2 v0.1.140 API notes
- `LlamaSampler::chain_simple([...])` — takes `IntoIterator<Item = LlamaSampler>`
- `model.str_to_token(text, AddBos::Always)` → `Vec<LlamaToken>`
- `model.token_to_str(token, Special::Tokenize)` — deprecated; using with `#[allow(deprecated)]`
- `LlamaContextParams` derives `Clone`; safe to construct once and reuse pattern
- `LlamaModelParams::with_n_gpu_layers(u32::MAX)` is the "all layers" signal

## Build Verification
- `cargo check --no-default-features` ✅ — CPU-only, no LLVM/CMake needed
- `cargo check --features cuda` ✅ — CUDA path, all llama.cpp FFI bindings resolve

## Assumptions & Risks
- `token_to_str` is deprecated but functional; switch to `token_to_piece` + `encoding_rs::Decoder` in Phase 3
- Approximate tokenize/detokenize in `LlamaCppBackend` is fine for Phase 2 (not used in generation path)
- No conversation context pruning — very long conversations will overflow the KV cache
- Qwen3.5's hybrid attention (10 full + 30 linear layers) is transparent to the llama.cpp API
- MoE expert routing is handled internally by llama.cpp; no special Phase 2 handling needed
