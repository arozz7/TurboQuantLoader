# TurboQuantLoader

A local LLM inference server in Rust. Loads GGUF models, applies TurboQuant KV cache
compression, and serves an OpenAI-compatible HTTP API on `http://127.0.0.1:7432`.

## Core Philosophy
> Readability and Order > Speed and Complex Interconnections

Prioritize maintainability and clarity over clever, hyper-optimized code.
Follow SOLID, DRY, and KISS principles.

## The Golden Rule
**DO NOT WRITE CODE** until an Implementation Plan is explicitly approved (for
multi-file or architectural changes). **NO DELETIONS** without explicit confirmation.

## Stack
- **Language:** Rust (edition 2021), `stable` toolchain
- **Inference backend:** `llama-cpp-2` (GGUF loading, multi-GPU, KV cache quantization)
- **GPU backends:** CUDA (`--features cuda`), Metal (`--features metal`), Vulkan (`--features vulkan`), CPU-only (default/no features)
- **KV cache compression:** `tq-kv` / `turboquant` (behind `turbo-kv` feature flag until Qwen3.5 MoE is supported — currently uses llama.cpp native KV quantization)
- **HTTP server:** `axum` 0.7
- **CLI:** `clap` 4 (derive)
- **TUI:** `ratatui` + `crossterm` (Phase 5, behind `tui` feature flag, desktop only)
- **Config:** TOML (`config.toml`)

## Platform Targets
| Platform | GPU | Feature Flag |
|----------|-----|--------------|
| Windows | CUDA (NVIDIA) | `--features cuda` |
| Linux | CUDA (NVIDIA) | `--features cuda` |
| macOS | Metal (Apple) | `--features metal` |
| iOS | Metal | `--features metal` (Phase 6) |
| Android | Vulkan | `--features vulkan` (Phase 6) |
| Any | CPU-only | *(no features — the default)* |

**The default feature set is empty** — never add a GPU feature to `default = [...]`.
The crate must compile and run CPU-only on every platform without extra tooling.

## Implementation Plan
Full phased plan: `docs/plan.md`

## Project Structure
```
src/
  config/       — AppConfig, ServerConfig, ModelConfig, KvCacheConfig
  model/        — ModelBackend trait, LlamaCppBackend, ModelRegistry
  inference/    — InferenceEngine, Sampler, TokenStream
  kv_cache/     — KvCacheBackend trait, LlamaNativeCache, TurboQuantCache (stub)
  server/       — Axum router, OpenAI-compatible routes, SSE streaming
  tui/          — ratatui TUI (Phase 5)
  gpu_stats.rs  — nvml-wrapper GPU telemetry
  main.rs       — CLI entry point
docs/           — Documentation, setup guides
scripts/
  release/      — Versioning, tagging
  deploy/       — Smoke tests
aiChangeLog/    — Phase-based change logs
```

## GPU Configuration
- Primary:   RTX 4070 Ti Super — 16 GB VRAM (tensor split weight: 16.0)
- Secondary: RTX 2060 — 6 GB VRAM (tensor split weight: 6.0)
- Intel GPU: ignored for inference

## Default Port
`7432` — chosen to avoid collision with common dev ports (3000, 5432, 8080, 11434)

## Git Workflow
- Branch naming: `feature/phase-XX-short-desc` or `fix/issue-desc`
- Protected main branch — never force push
- Conventional commits: `feat(scope):`, `fix(scope):`, `docs(scope):`, `refactor(scope):`
- Stage specific files — avoid `git add .`
- Write change logs to `aiChangeLog/phase-XX.md` after each phase

## Rust Standards
- Strict typing — no `unwrap()` in library code, use `?` operator and `anyhow`
- `thiserror` for domain error types
- `tracing` for structured logging (not `println!` in production paths)
- `clippy` clean — `#![deny(clippy::unwrap_used)]` in lib code
- All public items documented with `///`
- No `unsafe` without explicit justification comment

## Shell & Terminal Standards (Windows)
- Primary Shell: PowerShell (pwsh) for scripts
- Scripts: `.ps1` extension
- Claude Code environment uses bash — use Unix syntax in tool calls

## Change Logging
After each phase, update `aiChangeLog/phase-XX.md` with:
- Files created / modified (include refactor mappings)
- Behavior changes
- Assumptions and risks

## Build Prerequisites by Platform

**All platforms (required):**
- Rust stable toolchain: `rustup update stable`

**Windows (`--features cuda`):**
- CUDA Toolkit 12.x
- Visual Studio Build Tools (MSVC linker)
- `LIBCLANG_PATH` if llama-cpp-2 build requires it (see `.cargo/config.toml`)

**macOS (`--features metal`):**
- Xcode Command Line Tools: `xcode-select --install`
- Metal ships with macOS SDK — no extra install

**Linux (`--features cuda`):**
- CUDA Toolkit 12.x + NVIDIA drivers
- `build-essential`, `libclang-dev`

**Android (`--features vulkan`, Phase 6):**
- Android NDK r26+
- `cargo-ndk`: `cargo install cargo-ndk`

**iOS (`--features metal`, Phase 6):**
- macOS build machine required (Xcode)
- `rustup target add aarch64-apple-ios aarch64-apple-ios-sim`
