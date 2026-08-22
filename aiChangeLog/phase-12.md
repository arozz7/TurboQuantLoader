# Phase 12 — Independent K/V Cache Quantization Types

## Goal
Replace the single shared `[kv_cache] bits` setting (2/3/4/8, applied
identically to both K and V tensors) with independent `type_k` / `type_v`
fields that accept llama.cpp's actual `--cache-type-k` / `--cache-type-v`
type strings, defaulting both to `f16` (no quantization — llama-server's own
default).

## Why
Investigating a token-generation slowdown (~27-30 tok/s dropping to ~12-13
tok/s over the course of long agentic completions) led to comparing
TurboQuantLoader's KV cache config against LM Studio's, which exposes K and V
quantization as two independent dropdowns. Our config forced both to the same
quantization level via one `bits` field, and that field's `Two`/`Three`
variants mapped to `q2_K`/`q3_K` — types `llama-server --cache-type-k/-v`
doesn't actually accept (confirmed against the installed binary's `--help`
output: only `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` are
allowed), so those two variants would have failed at the subprocess boundary
if ever selected. K (used in attention score dot-products) is also more
sensitive to quantization error than V, so a common tuning pattern —
quantize V more aggressively than K — was structurally impossible with one
shared field.

## Files Modified
| File | Change |
|------|--------|
| `src/config/kv_cache.rs` | Replaced `KvBits` (numeric enum, `TryFrom<u8>`/`From<u8>`) with `KvType` (string enum: `F32, F16, Bf16, Q8_0, Q4_0, Q4_1, Iq4Nl, Q5_0, Q5_1`, `#[default]` `F16`). Added `as_cli_str()`, `bytes_per_element()`, and `FromStr`. `KvCacheConfig.bits` → `type_k` + `type_v`. |
| `src/config/mod.rs` | `CliOverrides.kv_bits: Option<KvBits>` → `kv_type_k` / `kv_type_v: Option<KvType>`; `apply_cli_overrides` updated. |
| `src/server/llama_process.rs` | `build_args` emits `--cache-type-k`/`--cache-type-v` from `type_k`/`type_v` independently (previously computed one `bit_type` string reused for both flags). |
| `src/model/llama_cpp.rs` | `kv_cache_type()` maps `KvType` → `llama_cpp_2::context::params::KvCacheType` (verified against the installed crate source, `llama-cpp-2 0.1.146` — all 9 variants, including `IQ4_NL`/`BF16`, exist 1:1). `rebuild_context` now calls `.with_type_k(...)`/`.with_type_v(...)` with independently-mapped types instead of one shared `kv_type`. |
| `src/kv_cache/llama_native.rs` | `LlamaNativeCache` holds `type_k`/`type_v` instead of one `bits` field; `stats().compression_ratio` computed from the average of both types' `bytes_per_element()` against an F16 baseline. |
| `src/main.rs` | `serve --kv-bits <u8>` → `serve --kv-type-k <str> --kv-type-v <str>`; `bench --bits "4,8"` → `bench --kv-types "f16,q8_0,q4_0"` (sweep applies the same type to both K and V per combination); `BenchRow.kv_bits: u8` → `kv_type: String`. |
| `config.toml` | `[kv_cache] bits = 4` → `type_k = "f16"` / `type_v = "f16"`, with a comment noting the K-more-sensitive-than-V tuning pattern. |

## Behavior Changes
- Default KV cache is now unquantized (`f16`), matching llama-server's own
  default. Previously the default was `bits = 4` (`q4_0` on both K and V).
  This trades ~2x KV cache VRAM footprint for removing 4-bit dequant overhead
  from the attention path — the working hypothesis for part of the observed
  token-generation slowdown on long-context generations, pending a live
  before/after comparison.
- K and V can now be quantized independently via `config.toml` or
  `--kv-type-k` / `--kv-type-v`.
- `bits = 2` / `bits = 3` (previously silently mapped to `q2_K`/`q3_K`, which
  llama-server's CLI never actually accepted) are no longer expressible —
  the new `KvType` enum only contains llama.cpp's real accepted values.

## Assumptions & Risks
- The `model/llama_cpp.rs` (`llama-backend`/`cuda`/`vulkan` feature) mapping
  could not be compiled end-to-end in this session — the dev machine's shell
  didn't have `cmake` on `PATH`, which `llama-cpp-sys-2`'s build script
  requires. Verified instead by reading the exact `KvCacheType` enum
  definition in the installed `llama-cpp-2 0.1.146` crate source
  (`~/.cargo/registry/src/.../llama-cpp-2-0.1.146/src/context/params.rs`) and
  matching variant names 1:1. `cargo check`/`cargo test` with default
  (CPU-only) features pass (24/24 tests), which covers `config/`,
  `server/llama_process.rs`, and `kv_cache/llama_native.rs`.
- `bytes_per_element()` estimates are approximate (block-quant scale/min
  overhead ignored), consistent with the precision the old `bits`-based
  `compression_ratio` calculation already used — not meant to be exact.
