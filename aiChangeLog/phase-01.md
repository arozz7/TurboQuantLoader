# Phase 01 — Foundation & Config

**Status:** In Progress
**Date Started:** 2026-03-27

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

## Remaining Phase 1 Tasks
- [ ] `config/mod.rs` — AppConfig, load_from_file, apply_cli_overrides
- [ ] `config/server.rs` — ServerConfig
- [ ] `config/model.rs` — ModelConfig
- [ ] `config/kv_cache.rs` — KvCacheConfig, KvBits, KvStrategy enums
- [ ] `model/backend.rs` — ModelBackend trait, GenerateRequest, GenerateEvent, GenerateSummary
- [ ] `kv_cache/mod.rs` — KvCacheBackend trait, CacheStats
- [ ] `model/registry.rs` — ModelEntry, ModelRegistry scan
- [ ] `main.rs` — full CLI with clap subcommands (serve, run, bench, list)
- [ ] Verify `cargo check --no-default-features` passes

## Assumptions & Risks
- `llama-cpp-2` crate may require specific CUDA Toolkit version — document in README
- `tq-kv` and `turboquant` crate versions are `*` — pin once exact versions confirmed
- CI runs without CUDA (GitHub Actions has no GPU); local builds use `--features cuda`
- Potential plan update: `llama-cpp-turboquant` fork (TheTom) integrates TurboQuant at C++ level
  — may be a better backend than mainline `llama-cpp-2` + separate `tq-kv`
  — evaluation deferred to Phase 3 start
