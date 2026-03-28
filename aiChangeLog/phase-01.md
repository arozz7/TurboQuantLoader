# Phase 01 — Foundation & Config

**Status:** Complete
**Date Started:** 2026-03-27
**Date Completed:** 2026-03-28

## Goal
Compilable project skeleton with all interfaces defined. No inference runs in this phase.

## Files Created
- `Cargo.toml` — full dependency manifest with all planned deps and feature flags
- `src/main.rs` — stub entry point
- `config.toml` — default configuration (all settings documented)
- `CLAUDE.md` — project-specific Claude Code instructions
- `README.md` — project overview and build prerequisites
- `rust-toolchain.toml` — pins stable Rust channel
- `.cargo/config.toml` — Windows/CUDA build hints, CUDA_DEVICE_ORDER env
- `.gitignore` — Rust, CUDA, model files
- `.github/workflows/ci.yml` — check + fmt + clippy on push (no-CUDA for CI)
- `docs/plan.md` — full phased implementation plan
- `aiChangeLog/phase-01.md` — this file

## Directories Created
- `src/config/`, `src/model/`, `src/inference/`, `src/kv_cache/`, `src/server/`, `src/tui/`
- `docs/`, `aiChangeLog/`, `scripts/release/`, `scripts/deploy/`
- `.cargo/`, `.github/workflows/`

## Phase 1 Tasks — Completed
- [x] `config/server.rs` — `ServerConfig` with host, port, concurrency, timeout
- [x] `config/model.rs` — `ModelConfig` with model_path, mmproj_path, models_dir, n_gpu_layers, tensor_split, context_size, batch_size, threads
- [x] `config/kv_cache.rs` — `KvCacheConfig`, `KvBits` (2/3/4/8 int serde), `KvStrategy` (snake_case string serde)
- [x] `config/mod.rs` — `AppConfig`, `CliOverrides`, `load_from_file()`, `apply_cli_overrides()`
- [x] `model/backend.rs` — `ModelBackend` trait, `GenerateRequest`, `GenerateEvent`, `GenerateSummary`, `GenerateStream`, `SamplerParams`
- [x] `kv_cache/mod.rs` — `KvCacheBackend` trait, `CacheStats`
- [x] `model/registry.rs` — `ModelEntry`, `ModelRegistry::scan()`, `ModelRegistry::find_by_name()`
- [x] `model/mod.rs` — module re-exports
- [x] `inference/mod.rs`, `server/mod.rs`, `tui/mod.rs` — phase stubs
- [x] `main.rs` — full clap CLI (serve, run, bench, list); `list` fully functional, others stub with `bail!`
- [x] `Cargo.toml` — made `llama-cpp-2` optional behind new `llama-backend` feature
- [x] `cargo check --no-default-features` passes clean

## Cargo.toml Changes
- `llama-cpp-2` changed from unconditional dep to `optional = true`
- New `llama-backend` feature added; enabled by `cuda`, `metal`, `vulkan`
- This allows `cargo check --no-default-features` to run without LLVM/libclang installed
- Phase 2 `LlamaCppBackend` impl must be gated with `#[cfg(feature = "llama-backend")]`

## Assumptions & Risks
- `llama-cpp-2` crate may require specific CUDA Toolkit version — document in README
- `tq-kv` and `turboquant` crate versions are `*` — pin once exact versions confirmed
- CI runs without CUDA (GitHub Actions has no GPU); local builds use `--features cuda`
- Potential plan update: `llama-cpp-turboquant` fork (TheTom) integrates TurboQuant at C++ level
  — may be a better backend than mainline `llama-cpp-2` + separate `tq-kv`
  — evaluation deferred to Phase 3 start
