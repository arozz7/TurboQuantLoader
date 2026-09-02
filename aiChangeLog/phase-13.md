# Phase 13 — Arc B70 Performance Tuning (config-only)

## Goal
Apply a published, hardware-matched tuning recipe (sergiiob.dev's "Qwen3.6 27B on
Intel Arc Pro B70: The Full Recipe") to `config.toml` — same GPU (Arc Pro B70), same
model family (Qwen3.6/3.8-27B) — to test whether its reported gains (asymmetric K/V
KV cache quantization, paired speculative-decoding tuning, larger batch/ubatch)
translate from their SYCL/Linux setup to our Vulkan/Windows one.

## Why
Investigating a token-generation slowdown (documented in earlier sessions — see
conversation history, no dedicated phase file since it was pure investigation, not
code) ruled out KV cache bit-width as the cause (f16 decayed identically to q4_0 at
long context) and identified the decay as inherent to growing context length. Two
external articles on this exact GPU/model combination surfaced concrete, tested
config values worth trying: a Medium post reporting Vulkan-vs-SYCL backend gains
(+52%, out of scope here — needs a new binary build), and the sergiiob.dev recipe
below, which is entirely config-level and testable with the existing binary.

## Files Modified
| File | Change |
|------|--------|
| `config.toml` | `[kv_cache] type_k`/`type_v`: `f16`/`f16` → `q8_0`/`q4_1` (recipe: "never use q4_0 for K" — KL-divergence ~5.5 vs ~0.003 for q8_0; V "tolerates aggressive quantization" fine). `[backend] spec_draft_n_max`: `1` → `4`, paired with a new `--spec-draft-p-min 0.75` in `extra_flags` (recipe reports 93-94% draft acceptance with this pairing, vs. the ~50-60% we measured with `n_max` alone and no probability floor). `extra_flags`: `--ubatch-size` `2048` → `4096`; added `--no-mmap`. `[model] batch_size`: `4096` → `8192`. |

## Behavior Changes
- KV cache VRAM footprint changes (asymmetric q8_0/q4_1 vs. the prior f16/f16 —
  smaller than f16/f16, larger than the original q4_0/q4_0 default from before this
  investigation started).
- Speculative decoding now gates draft proposals on a minimum probability
  (`--spec-draft-p-min 0.75`) in addition to the existing `n_max` cap — previously
  no probability floor was set (default `0.00`, i.e. no gating).
- `--no-mmap` disables llama-server's default mmap-based model loading. Untested here
  whether this matters on Windows/Vulkan the way the recipe's Linux/SYCL numbers
  suggest — flagged for observation, not a confirmed win on this platform.

## Assumptions & Risks
- All values are carried over from a recipe tuned for a **different backend** (SYCL)
  and **different OS** (Linux) on the same GPU model. Nothing here has been verified
  against actual before/after throughput numbers on this Vulkan/Windows deployment —
  this phase applies the config, a follow-up session needs to capture comparable
  long-completion decay curves (same methodology as the earlier KV-bits investigation)
  to confirm or revert each change independently.
- `--spec-draft-p-min` is only in `extra_flags` (global, verbatim), not a typed
  `[backend]` field — unlike `spec_draft_n_max`, it has no per-model override via
  `[models.load]`. Every model launched shares the same 0.75 floor regardless of its
  own draft-head calibration.
- The SYCL backend swap this recipe also validates was explicitly left out of scope —
  it requires building a separate SYCL-enabled llama-server binary (Intel oneAPI
  toolkit), which the current binary (Vulkan-only, no SYCL symbols) does not have.
