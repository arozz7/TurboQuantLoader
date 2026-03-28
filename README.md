# TurboQuantLoader

A local LLM inference server in Rust. Loads GGUF models with multi-GPU support,
applies TurboQuant KV cache compression for long-context efficiency, and exposes
an OpenAI-compatible HTTP API so any tool that speaks OpenAI (Claude Code, Cursor,
Continue.dev, etc.) can use your local models.

## Status

> **Phase 1 — Foundation & Config** (complete) — skeleton compiles, `list` command works
> **Phase 2 — Inference Engine** (complete) — model loads on both GPUs, tokens stream, `run` command works
> **Phase 3 — KV Cache & Benchmarks** (next)

See [`docs/plan.md`](docs/plan.md) for the full phased implementation plan.

## Features (planned)

| Feature | Phase | Status |
|---------|-------|--------|
| GGUF model loading (IQ3_XXS, MoE, hybrid attention) | 2 | Planned |
| Multi-GPU tensor split (4070 Ti Super + RTX 2060) | 2 | Planned |
| Interactive terminal chat (`run` command) | 2 | Planned |
| KV cache compression (4-bit default, switchable) | 3 | Planned |
| Benchmark command with GPU telemetry | 3 | Planned |
| OpenAI-compatible HTTP API (`/v1/chat/completions`) | 4 | Planned |
| Streaming SSE responses | 4 | Planned |
| Vision / multimodal support (mmproj) | 5A | Planned |
| ratatui TUI with live stats | 5B | Planned |

## Platform Support

| Platform | GPU Backend | Feature Flag | Phase |
|----------|------------|--------------|-------|
| Windows | CUDA (NVIDIA) | `--features cuda` | 2 |
| Linux | CUDA (NVIDIA) | `--features cuda` | 2 |
| macOS (Apple Silicon / Intel) | Metal | `--features metal` | 2 |
| Any platform | CPU-only | *(no features)* | 2 |
| Android | Vulkan | `--features vulkan` | 6 |
| iOS | Metal | `--features metal` | 6 |

**Default build is CPU-only** — it compiles and runs on any OS without GPU drivers.
Pick your platform's feature flag to enable GPU acceleration.

## Hardware (Primary Dev Machine)

- RTX 4070 Ti Super — 16 GB VRAM (CUDA device 0, tensor split primary)
- RTX 2060 — 6 GB VRAM (CUDA device 1, tensor split secondary)
- Windows 11

## Build Prerequisites

**All platforms — required:**
```sh
rustup update stable
```
No other dependencies for a CPU-only build (`cargo build --no-default-features`).

---

**Windows — CUDA (`--features cuda`):**

| Component | Version | Notes |
|-----------|---------|-------|
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 12.x (12.6 recommended) | 13.x untested with llama.cpp |
| [LLVM](https://github.com/llvm/llvm-project/releases) | Any recent (22.x tested) | Required by bindgen to generate llama.cpp bindings |
| Visual Studio Build Tools | 2022 (17.x) | MSVC linker + Windows SDK |

After installing LLVM, `.cargo/config.toml` already points `LIBCLANG_PATH` to
`C:/Program Files/LLVM/bin`. If you installed LLVM elsewhere, update that path.

> **Why LLVM?** `llama-cpp-2` uses `bindgen` to auto-generate Rust FFI bindings
> from the llama.cpp C++ headers. `bindgen` calls `libclang.dll` to parse those
> headers. Without it, the build script panics before any Rust code is compiled.
> This only applies to GPU builds — `--no-default-features` skips `llama-cpp-2`
> entirely and requires no native tooling beyond Rust itself.

---

**macOS — Metal (`--features metal`):**
```sh
xcode-select --install   # Metal SDK ships with Xcode CLT, nothing else needed
```

---

**Linux — CUDA (`--features cuda`):**
```sh
sudo apt install build-essential libclang-dev
# + CUDA Toolkit 12.x from NVIDIA
```

## Quick Start

```sh
# CPU-only — works on any platform, no GPU required
cargo build --release

# Windows / Linux with NVIDIA GPU
cargo build --release --features cuda

# macOS with Apple GPU
cargo build --release --features metal

# List available models
cargo run --release -- list

# Interactive chat (add GPU feature as above)
cargo run --release -- run

# Start API server
cargo run --release -- serve
```

## Configuration

Edit `config.toml` to set model path, GPU layers, context size, and KV cache settings.
For local overrides without affecting git, copy to `config.local.toml` (gitignored).

Key defaults:
- API server: `http://127.0.0.1:7432`
- KV cache: 4-bit compression
- GPU layers: all offloaded to GPU
- Context size: 8,192 tokens (increase once stable)

## Claude Code Integration

Once the server is running (`cargo run --release -- serve`), configure Claude Code:

```json
{
  "openai": {
    "apiKey": "local",
    "baseUrl": "http://127.0.0.1:7432/v1",
    "model": "Qwen3.5-35B-A3B-UD-IQ3_XXS"
  }
}
```

See [`docs/claude-code-setup.md`](docs/claude-code-setup.md) for full instructions.

## Model

Primary test model: **Qwen3.5-35B-A3B** (Unsloth IQ3_XXS quantization)
- 35B parameters, ~3.5B active (MoE: 8 of 256 experts per token)
- 262k token context window
- Hybrid attention (10 full-attention + 30 linear-attention layers)
- File size: 14.7 GB

## License

MIT
