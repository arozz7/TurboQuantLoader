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

> **Phase 1 — Foundation & Config** (complete) — skeleton compiles, `list` command works
> **Phase 2 — Inference Engine** (complete) — model loads on both GPUs, tokens stream, `run` command works
> **Phase 6 — llama-server Subprocess Backend** (complete) — proxy mode, MTP speculative decoding, metrics API, hot-swap

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

Key `config.toml` settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `server.port` | `7432` | External API port |
| `backend.binary_path` | — | Path to `llama-server` executable |
| `backend.internal_port` | `7433` | Port llama-server listens on |
| `backend.variant` | `llama_server` | `llama_server` or `turbo_quant` |
| `backend.extra_flags` | `[]` | Extra CLI flags passed to llama-server |
| `backend.restart_on_crash` | `true` | Auto-restart on subprocess crash |
| `model.model_path` | — | Path to `.gguf` model file |
| `model.main_gpu` | `1` | Primary GPU device index |
| `model.tensor_split` | `[16.0, 32.0]` | VRAM split weights |
| `model.context_size` | `262144` | Max context tokens |
| `kv_cache.bits` | `4` | KV cache quantization bits (2/3/4/8) |

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
  provider: lmstudio        # reuses the existing OllamaClient (/v1/chat/completions)
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

**Important:** Set `single_model_only: false` in the `local_runtime` section (or
move this model above other local models in the yaml so it becomes the default).
TurboQuantLoader manages its own model lifecycle — it does not implement the LM Studio
load/unload API. The `single_model_only` flag triggers LM Studio-specific endpoints
that TurboQuantLoader does not serve.

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
