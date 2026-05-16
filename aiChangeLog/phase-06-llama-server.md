# Phase 06 — llama-server Subprocess Backend

**Status:** Complete

## Motivation

`llama-cpp-2` (Rust bindings) lags behind llama.cpp feature releases — no MTP
speculative decoding, no `--tensor-split`, no Vulkan tensor split. Intel Arc Pro B70
(32 GB VRAM) requires Vulkan and is invisible to CUDA builds. Switching to a managed
`llama-server` subprocess gives all llama.cpp features immediately, including:
- `--spec-type draft-mtp --spec-draft-n-max 2` (MTP speculative decoding)
- `--flash-attn`
- `--main-gpu 1 --tensor-split 16.0,32.0` (Arc B70 as primary, RTX 4070 Ti overflow)

## Architecture Change

TurboQuantLoader is now a **process manager + API translator + metrics layer**:
```
Client → TurboQuantLoader (7432) → llama-server (7433)
```
- OpenAI route: tool injection → proxy to llama-server (transparent for non-streaming)
- Anthropic route: translate Anthropic↔OpenAI + StreamParser for `<tool_call>` XML

## Files Created

| File | Description |
|------|-------------|
| `src/config/backend.rs` | `BackendVariant` enum, `BackendConfig` struct |
| `src/server/llama_process.rs` | `LlamaProcess::start()`, health polling, Windows-safe Drop |
| `src/server/proxy.rs` | `proxy_request()`, `spawn_event_reader()`, `build_chat_body()` |
| `aiChangeLog/phase-06-llama-server.md` | This file |

## Files Modified

| File | Change |
|------|--------|
| `src/config/mod.rs` | Added `backend: BackendConfig` to `AppConfig` |
| `src/config/model.rs` | Added `main_gpu: i32`, changed `context_size` default to 262144, updated `tensor_split` default to `[16.0, 32.0]` |
| `src/server/mod.rs` | `AppState`: replaced `engine: Arc<InferenceEngine>` with `process: Arc<LlamaProcess>` + `config: Arc<AppConfig>`; `serve()` now spawns subprocess |
| `src/server/routes/models.rs` | Model name from config path stem instead of engine |
| `src/server/routes/openai.rs` | Uses `spawn_event_reader` + `proxy_request`; removes `InferenceEngine` dependency |
| `src/server/routes/anthropic.rs` | Uses `spawn_event_reader`; removes `InferenceEngine` dependency; Anthropic↔OpenAI translation preserved |
| `config.toml` | Added `[backend]` section; updated model path to Qwen3.6-27B Q6_K; `context_size = 262144`; `main_gpu = 1`; `tensor_split = [16.0, 32.0]` |
| `run.ps1` | Removed `--features cuda`; updated comments for Vulkan/proxy mode |
| `Cargo.toml` | Added `reqwest = { version = "0.12", features = ["stream", "json"] }` |

## New Model

`J:/llama/Models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q6_K.gguf`
- 27B Q6_K (~22 GB), fits entirely on Arc Pro B70 (32 GB)
- MTP speculative decoding: `--spec-type draft-mtp --spec-draft-n-max 2`
- Max context: 262,144 tokens

## Additional files (completed in same session)

| File | Description |
|------|-------------|
| `src/metrics.rs` | `MetricsCollector` — atomic counters, rolling VecDeque (100 requests), background GPU poller (2 s), p50/p95/p99 for TPS and TTFT |
| `src/server/routes/metrics.rs` | `GET /health` (rich JSON), `GET /metrics` (Prometheus text), `GET /v1/admin/stats` (histogram JSON) |
| `src/server/routes/admin.rs` | `GET /v1/admin/status`, `POST /v1/admin/restart`, `POST /v1/admin/load` (hot-swap model) |
| `scripts/build-backend.ps1` | Builds `llama-server-vulkan.exe` (ggml-org/llama.cpp, `-DGGML_VULKAN=ON`) and `llama-server-turboquant.exe` (TheTom fork, `-DGGML_CUDA=ON`) |

## Assumptions and Risks

- llama-server health endpoint returns `{"status":"ok"}` on port 7433 — verified via manual test on build b9189
- `--spec-type draft-mtp` accepted by b9189 binary (confirmed via `--help` output)
- Vulkan device ordering: Arc Pro B70 = device [1], RTX 4070 Ti Super = device [0] — to be verified on first run
- `stream_options: {include_usage: true}` may not be supported by all llama-server versions; gracefully falls back to token counting
