# Phase 17 — Fix CI: `cargo fmt`, GPU-feature compile error, clippy

## Goal
Get the `feature/phase-01-impl` PR's CI green — all 4 jobs (`Check Metal feature —
macOS`, `Check & Lint` × macOS/Ubuntu/Windows) were failing.

## Why
User reported the PR's CI checks failing; `gh run view` on the latest run showed two
distinct, unrelated causes, neither introduced by this session's earlier commits
(both are pre-existing — the CI history shows failures going back to the PR's original
open date):

1. **`cargo fmt --all -- --check` failing on all 3 `Check & Lint` platforms** — 15
   formatting diffs across `main.rs`, `model/registry.rs`, `server/llama_process.rs`,
   `server/routes/anthropic.rs`, `server/routes/metrics.rs`, `server/stream_parser.rs`.
   None of these were introduced by this session; they'd simply never been run through
   `cargo fmt` before committing, on this machine or otherwise. This also explains why
   `Cargo clippy (CPU-only)` never ran on any platform — it's gated behind `cargo fmt`
   succeeding first in the workflow.
2. **`cargo check --no-default-features --features metal` failing to compile** —
   `error[E0063]: missing field `cached_tokens` in initializer of `GenerateSummary``
   at `src/model/llama_cpp.rs:475`. This is the in-process `llama-cpp-2` backend
   (compiled only under `metal`/`cuda`/`vulkan` features — never exercised by this
   session's `cargo check`/`cargo test` runs, which only cover the default CPU-only
   feature set on this Windows dev machine, since neither a macOS toolchain nor a
   working local `cmake`/CUDA/Vulkan SDK setup was available to build it). The
   `cached_tokens` field was added to `GenerateSummary` in an earlier commit
   ("prompt cache hit-rate metrics") that updated `server/proxy.rs`'s construction
   site but missed this one.

## Files Modified
| File | Change |
|------|--------|
| `src/model/llama_cpp.rs` | Added `cached_tokens: 0` to the `GenerateSummary` literal — the in-process llama-cpp-2 backend has no llama-server-style prompt cache to report a hit rate for. |
| `src/main.rs`, `src/model/registry.rs`, `src/server/llama_process.rs`, `src/server/routes/anthropic.rs`, `src/server/routes/metrics.rs`, `src/server/stream_parser.rs` | `cargo fmt --all` — whitespace/formatting only, no logic changes. |
| `src/server/stream_parser.rs` | 3 `clippy::manual_contains` lints in test code (`evts.iter().any(\|e\| *e == X)` → `evts.contains(&X)`) — not part of the CI-blocking clippy invocation (`cargo clippy --no-default-features`, which doesn't lint `#[cfg(test)]` without `--tests`/`--all-targets`), but caught while verifying with `cargo clippy --all-targets` and fixed since they were free. |

## Behavior Changes
None — formatting only, one missing-field fix with an unambiguous correct value
(`0`, matching the "no cache to report" semantics already used for this backend
elsewhere), and test-only lint fixes.

## Assumptions & Risks
- The `metal` feature fix could not be compiled end-to-end in this session — no
  macOS machine available. Verified by reading the error location and the
  `GenerateSummary` struct definition directly; the fix is a single struct-literal
  field addition with an unambiguous value, low risk, but not empirically confirmed
  green. Watch the next CI run on this PR to confirm.
- `cuda`/`vulkan` feature builds were not re-verified either (this machine lacks a
  working `cmake`/Vulkan SDK setup, per earlier sessions) — if either of those
  features has its own separate `GenerateSummary` construction site or other drift
  from `cargo fmt`/clippy, it wouldn't have been caught here. `cuda`/`vulkan` aren't
  in this repo's CI matrix currently (only `metal` is checked as a GPU feature,
  alongside CPU-only `Check & Lint`), so this is a latent gap in CI coverage, not
  something this phase introduced.
