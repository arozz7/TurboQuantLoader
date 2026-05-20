# TurboQuantLoader

A local LLM inference server in Rust. Manages a `llama-server` subprocess, applies
TurboQuant KV cache compression for long-context efficiency, and exposes an
OpenAI-compatible HTTP API so any tool that speaks OpenAI (Claude Code, Cursor,
Continue.dev, coding agents, etc.) can use your local models.

```
Client (7432) → TurboQuantLoader → llama-server (7433)
```

TurboQuantLoader is a **process manager + API translator + metrics layer**:
it spawns `llama-server`, health-checks it, proxies requests, injects tool-call
scaffolding for Qwen3, and exposes Prometheus metrics + a hot-swap admin API.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation & Config | ✓ Complete |
| 2 | Inference Engine | ✓ Complete |
| 6 | llama-server Subprocess Backend | ✓ Complete |
| 7 | Client-Driven Model Selection | ✓ Complete |
| 8 | File Logging & API Instrumentation | ✓ Complete |
| 9 | Conversation Logging (JSONL) | ✓ Complete |

See [`docs/plan.md`](docs/plan.md) for the full phased implementation plan.

## Features

| Feature | Phase | Status |
|---------|-------|--------|
| llama-server subprocess manager | 6 | Done |
| OpenAI-compatible HTTP API (`/v1/chat/completions`) | 6 | Done |
| Streaming SSE responses (proxy) | 6 | Done |
| Anthropic API translation layer | 6 | Done |
| Qwen3 tool-call XML injection + parsing | 6 | Done |
| MTP speculative decoding (`--spec-type draft-mtp`) | 6 | Done |
| Prometheus metrics (`/metrics`) | 6 | Done |
| Admin API — hot-swap model, restart | 6 | Done |
| Multi-GPU tensor split (Vulkan) | 6 | Done |
| KV cache compression (4-bit, llama-native) | 6 | Done |
| Dynamic model selection via `model` field | 7 | Done |
| Named model registry (`[[models]]` in config) | 7 | Done |
| Idle guard — prevents mid-session model switches | 7 | Done |
| Daily rolling log files with retention cleanup | 8 | Done |
| Structured per-request log events (tokens, TPS, TTFT) | 8 | Done |
| Conversation logging — full prompt + response JSONL | 9 | Done |
| Interactive terminal chat (`run` command) | 2 | Done |
| GGUF model loading via llama-cpp-2 | 2 | Done (behind `llama-backend` feature) |
| KV cache benchmarks | 3 | Planned |
| ratatui TUI with live stats | 5B | Planned |
| Android / iOS | 7 | Planned |

## Platform Support

| Platform | GPU Backend | Notes |
|----------|------------|-------|
| Windows | Vulkan (Intel Arc, NVIDIA) | Primary dev platform — `run.ps1` |
| Windows | CUDA (NVIDIA) | `--features cuda` for embedded llama-cpp-2 `run`/`bench` |
| Linux | Vulkan / CUDA | Same feature flags |
| macOS (Apple Silicon) | Metal | `--features metal` |
| Any | CPU-only | Default build — no GPU features |

**The default Rust build is CPU-only** and has no GPU dependencies. GPU acceleration
is handled entirely inside the `llama-server` binary; TurboQuantLoader itself does
not need GPU feature flags to run in proxy mode.

## Hardware (Primary Dev Machine)

- Intel Arc Pro B70 — 32 GB VRAM (Vulkan device 1, primary inference GPU)
- RTX 4070 Ti Super — 16 GB VRAM (Vulkan device 0, KV cache overflow)
- Windows 11

## Build Prerequisites

**All platforms — required:**
```sh
rustup update stable
```
No other dependencies for a CPU-only build (`cargo build`).

---

**Windows — CUDA for embedded `run`/`bench` (`--features cuda`):**

| Component | Version | Notes |
|-----------|---------|-------|
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 12.x (12.6 recommended) | 13.x untested with llama.cpp |
| [LLVM](https://github.com/llvm/llvm-project/releases) | Any recent (22.x tested) | Required by bindgen |
| Visual Studio Build Tools | 2022 (17.x) | MSVC linker + Windows SDK |

After installing LLVM, `.cargo/config.toml` already points `LIBCLANG_PATH` to
`C:/Program Files/LLVM/bin`. If you installed LLVM elsewhere, update that path.

---

**macOS — Metal (`--features metal`):**
```sh
xcode-select --install   # Metal SDK ships with Xcode CLT
```

---

**Linux — CUDA (`--features cuda`):**
```sh
sudo apt install build-essential libclang-dev
# + CUDA Toolkit 12.x from NVIDIA
```

## Quick Start

### 1. Get a Vulkan-capable llama-server binary

Download a pre-built Vulkan binary from the llama.cpp releases page, or build
one yourself:

```sh
git clone https://github.com/ggml-org/llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j
# Binary: build/bin/Release/llama-server.exe (Windows)
```

Or use `scripts/build-backend.ps1` to build both the Vulkan and TurboQuant variants:

```powershell
.\scripts\build-backend.ps1 -Target vulkan
```

### 2. Configure

Edit `config.toml`:

```toml
[backend]
binary_path = "J:/llama/llama-server.exe"   # path to your llama-server binary
internal_port = 7433

[model]
model_path = "/path/to/your-model.gguf"
main_gpu = 1          # primary GPU index (Vulkan device enumeration)
tensor_split = [16.0, 32.0]   # proportional VRAM split across GPUs
context_size = 32768
```

For local overrides without affecting git, copy to `config.local.toml` (gitignored).

### 3. Build and run

```powershell
# Windows — recommended (sets env vars, starts server)
.\run.ps1

# Or directly:
cargo build --release
cargo run --release -- serve
```

```sh
# macOS / Linux
cargo build --release
./target/release/turbo-quant-loader serve
```

### Other commands

```sh
# List available models (scans models_dir from config.toml)
cargo run --release -- list

# Interactive chat via embedded llama-cpp-2 (requires --features cuda/metal)
cargo run --release --features cuda -- run
```

## Configuration Reference

Full `config.toml` settings:

**`[server]`**

| Setting | Default | Description |
|---------|---------|-------------|
| `port` | `7432` | External API port |
| `host` | `127.0.0.1` | Bind address |
| `max_concurrent_requests` | `4` | Request concurrency limit |
| `request_timeout_secs` | `300` | Per-request timeout |

**`[backend]`**

| Setting | Default | Description |
|---------|---------|-------------|
| `binary_path` | — | Path to `llama-server` executable |
| `internal_port` | `7433` | Port llama-server listens on |
| `variant` | `llama_server` | `llama_server` or `turbo_quant` |
| `extra_flags` | `[]` | Extra CLI flags passed verbatim to llama-server |
| `restart_on_crash` | `true` | Auto-restart subprocess on unexpected exit |
| `startup_timeout_secs` | `180` | Max seconds to wait for llama-server `/health` |

**`[model]`**

| Setting | Default | Description |
|---------|---------|-------------|
| `model_path` | — | Path to the default `.gguf` model file |
| `models_dir` | `models` | Root directory scanned by the `list` command and model auto-discovery |
| `main_gpu` | `-1` | Primary GPU device index (`-1` = auto) |
| `tensor_split` | `[]` | Per-device VRAM weights for multi-GPU splitting |
| `context_size` | `262144` | Max context window in tokens |
| `batch_size` | `512` | Prompt evaluation batch size |
| `threads` | half CPUs | CPU threads for non-GPU ops |
| `n_gpu_layers` | `-1` | Layers offloaded to GPU (`-1` = all) |
| `model_idle_timeout_secs` | `1800` | Seconds of inactivity before a client-requested model switch is allowed. Prevents mid-session reloads. Set `0` to disable. |

**`[kv_cache]`**

| Setting | Default | Description |
|---------|---------|-------------|
| `bits` | `8` | KV cache quantization bits: `2`, `3`, `4`, or `8` |
| `strategy` | `llama_native` | `llama_native` or `turbo_quant` |

**`[logging]`**

| Setting | Default | Description |
|---------|---------|-------------|
| `log_dir` | `logs` | Directory for rolling log files (created if absent) |
| `log_retention_days` | `7` | Delete log files older than N days at startup. `0` = keep forever. |
| `file_log_level` | `info` | Log level for the file appender. Accepts `RUST_LOG` syntax. |
| `stdout_log_level` | `info` | Log level for stdout. Overridden by `RUST_LOG` env var. |
| `log_conversations` | `false` | When `true`, write full prompt + response to `conversations.<date>.jsonl`. |

**`[[models]]`** (array — one entry per named model)

| Setting | Required | Description |
|---------|----------|-------------|
| `name` | Yes | Short identifier used as the OpenAI `model` field |
| `path` | Yes | Absolute path to the `.gguf` file |
| `context_size` | No | Overrides `[model] context_size` for this model |
| `n_gpu_layers` | No | Overrides `[model] n_gpu_layers` |
| `main_gpu` | No | Overrides `[model] main_gpu` |
| `batch_size` | No | Overrides `[model] batch_size` |
| `tensor_split` | No | Overrides `[model] tensor_split` |

## API Endpoints

### Inference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| `POST` | `/v1/messages` | Anthropic-compatible chat (streaming + non-streaming) |
| `GET` | `/v1/models` | List available models |

### Observability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Rich JSON health — llama-server status, uptime |
| `GET` | `/metrics` | Prometheus text metrics — TPS, TTFT, GPU stats |
| `GET` | `/v1/admin/stats` | Histogram JSON — p50/p95/p99 TPS and TTFT |

### Admin

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/admin/status` | Subprocess PID, uptime, config snapshot |
| `POST` | `/v1/admin/restart` | Gracefully restart the llama-server subprocess |
| `POST` | `/v1/admin/load` | Hot-swap model: `{"model_path": "/path/to/new.gguf"}` |

## Dynamic Model Selection

Agents select a model by passing its name as the OpenAI `model` field in any request:

```json
{ "model": "Qwen3.6-27B-Q6_K", "messages": [...] }
```

### Name Resolution Order

1. **Exact match** — `name` in `[[models]]` config array
2. **Substring match** — any `[[models]]` entry whose `name` contains the requested string
3. **File-stem scan** — walks `models_dir` looking for a `.gguf` whose filename contains the requested string
4. **Fallback** — unknown name → serve with the currently loaded model (no switch)

### Auto-Switch Flow

When a recognized model name differs from the currently loaded one:

1. Server returns **HTTP 503** with `Retry-After: 10` header immediately
2. Background task kills the old llama-server, starts a new one with the resolved model's settings
3. Client retries after 10 seconds — by then the new model is loaded and health-checked
4. While switching, all requests return 503 until the new model passes `/health`

### Idle Guard

To prevent mid-session reloads when an agent sends a different `model` string across turns, client-requested switches are blocked if the model was used within the last `model_idle_timeout_secs` seconds (default: 30 minutes).

```toml
[model]
model_idle_timeout_secs = 1800   # 30 minutes; set 0 to disable
```

The admin `POST /v1/admin/load` endpoint always bypasses the idle guard.

---

## Logging

TurboQuantLoader writes two independent log streams, both configured under `[logging]` in `config.toml`.

### Operational Log

Daily-rolling plain-text log files in `log_dir` (default: `logs/`):

```
logs/turboquant.2025-05-17.log
logs/turboquant.2025-05-18.log
...
```

Each completed request emits a structured event with:

| Field | Description |
|-------|-------------|
| `id` | Request UUID (correlates across log lines) |
| `model` | Active model name |
| `prompt_tokens` | Tokens in the prompt |
| `completion_tokens` | Tokens generated |
| `ttft_ms` | Time-to-first-token (ms) |
| `generation_ms` | Total generation time |
| `tps` | Tokens per second |
| `finish_reason` | `stop`, `length`, or `tool_calls` |

Log files older than `log_retention_days` are deleted at startup.

### Conversation Log (JSONL)

When `log_conversations = true`, each completed streaming request is appended as a JSON line to a separate daily file:

```
logs/conversations.2025-05-17.jsonl
```

**Record format:**
```json
{
  "ts": "2025-05-17T10:23:45Z",
  "id": "chatcmpl-abc123",
  "model": "Qwen3.6-27B-Q4_K_S",
  "protocol": "openai",
  "stream": true,
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Explain transformer attention..."}
  ],
  "response": "The transformer attention mechanism works by...",
  "prompt_tokens": 2048,
  "completion_tokens": 312,
  "tps": 17.1,
  "finish_reason": "stop"
}
```

`messages` contains the **prepared** messages (after tool injection into the system prompt), so the record reflects exactly what the model received.

Enable in `config.toml`:
```toml
[logging]
log_conversations = true
```

Errors in the logger (disk full, permission denied) are emitted as `WARN` events and never propagate to the inference path. Conversation files share the same `log_retention_days` cleanup sweep as the operational log.

---

## Claude Code Integration

Once the server is running, configure Claude Code:

```json
{
  "openai": {
    "apiKey": "local",
    "baseUrl": "http://127.0.0.1:7432/v1",
    "model": "Qwen3.6-27B-Q6_K"
  }
}
```

See [`docs/claude-code-setup.md`](docs/claude-code-setup.md) for full instructions.

## coding-agent Integration

TurboQuantLoader exposes a fully OpenAI-compatible `/v1/chat/completions` endpoint,
so the [coding-agent](../coding-agent) can use it as a local model provider.

Add to `coding-agent/config/models.yaml`:

```yaml
- name: Qwen3.6-27B-Q6_K
  type: local
  provider: turboquant      # non-lmstudio: skips LM Studio load/unload API; OllamaClient still hits /v1/chat/completions
  endpoint: ${TURBOQUANT_URL:-http://127.0.0.1:7432}
  context_window: 262144
  is_coding_optimized: true
  recommended_for: [coding, code_review, test_generation, planning, research]
  rate_limit_rpm: 60
```

Add to `coding-agent/.env`:
```dotenv
TURBOQUANT_URL=http://127.0.0.1:7432
```

**Important:** `single_model_only: false` must be set in `local_runtime` (already done
in the example above). TurboQuantLoader manages its own model lifecycle — it does not
implement LM Studio's load/unload API. Using `provider: turboquant` (any non-`lmstudio`
value) ensures the router skips those LM Studio-specific endpoints automatically.

To make TurboQuantLoader the default coding model:
```yaml
defaults:
  coding_model: Qwen3.6-27B-Q6_K
  planning_model: Qwen3.6-27B-Q6_K
```

## Model

Primary model: **Qwen3.6-27B** (Unsloth Q6_K quantization)
- 27B parameters, ~22 GB on disk (Q6_K)
- Fits entirely on Intel Arc Pro B70 (32 GB VRAM)
- 262k token context window
- MTP speculative decoding: `--spec-type draft-mtp --spec-draft-n-max 2`
- Path: `J:/llama/Models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q6_K.gguf`

## MTP Speculative Decoding

Qwen3.6 ships with Multi-Token Prediction (MTP) heads baked into the `.gguf`.
llama-server uses these as a draft model — no separate draft model download needed.

Expected acceptance rate: ~83% (per Unsloth benchmarks). At 262k context this
can yield 1.5–2× throughput improvement over vanilla sampling.

Enabled by default in `config.toml`:
```toml
[backend]
extra_flags = ["--flash-attn", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
```

## License

MIT
