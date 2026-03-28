# TurboQuantLoader

A local LLM inference server in Rust. Loads GGUF models with multi-GPU support,
applies TurboQuant KV cache compression for long-context efficiency, and exposes
an OpenAI-compatible HTTP API so any tool that speaks OpenAI (Claude Code, Cursor,
Continue.dev, etc.) can use your local models.

## Status

> **Phase 1 — Foundation & Config** (in progress)

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

## Hardware

Designed and tested on:
- RTX 4070 Ti Super — 16 GB VRAM (primary, tensor split index 0)
- RTX 2060 — 6 GB VRAM (secondary, tensor split index 1)
- Windows 11

## Build Prerequisites

1. **Rust stable** — `rustup update stable`
2. **CUDA Toolkit 12.x** — required for `--features cuda` (default)
   - Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
3. **Visual Studio Build Tools** (MSVC linker) — or Visual Studio 2022
4. **LLVM/Clang** (optional) — only if `llama-cpp-2` build requires `LIBCLANG_PATH`
   - See `.cargo/config.toml` for path hints

## Quick Start

```powershell
# Build (CUDA enabled by default)
cargo build --release

# List available models
cargo run --release -- list

# Interactive chat
cargo run --release -- run

# Start API server
cargo run --release -- serve

# Build without CUDA (CPU only)
cargo build --release --no-default-features
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
