# Phase 20 — KV cache: switch q8_0/q4_1 → f16/f16 after matched-depth A/B/C test

## Goal
Investigate whether Vulkan-specific techniques could flatten the decode-speed
decay observed once a session's context exceeds ~50% of the 131072-token
window, following a user report and a research pass into Vulkan-backend
long-context performance options.

## Why
A research pass (see conversation history 2026-08-28) into Vulkan-specific
decode speedups flagged a hardware-matched benchmark (Intel Arc Pro B70)
claiming `q8_0` KV cache is *slower* than `f16` past ~16K context on this
exact GPU — directly contradicting the sergiiob.dev recipe our current
`type_k=q8_0`/`type_v=q4_1` config was based on (phase 13). Worth a real,
controlled test rather than trusting either blog post.

Ran a matched-depth comparison: same 320KB synthetic prompt (concatenated
`src/**/*.rs`, tokenizing to 83,098 prompt tokens — 63% of the 131072
context), `max_tokens=300`, across three KV configs, full TurboQuantLoader
restart between each (required — `/v1/admin/restart` and `/v1/admin/load`
only reuse the in-memory config, they don't re-read `config.toml`, so a
config.toml edit needs the whole process restarted to take effect).

| Config | Decode | Prefill | Draft accept | VRAM |
|---|---|---|---|---|
| `q8_0`/`q4_1` (old default) | 4.31 tok/s | 72.8 tok/s | 91.1% | 21.5 GB idle |
| `f16`/`f16` | 3.87 tok/s (-10%) | **132.0 tok/s (+81%)** | **97.0%** | ~26.2 GB |
| `f16`/`q8_0` | 3.47 tok/s (worst) | 95.0 tok/s (+30%) | 92.7% | ~24.3 GB |

The blog's specific decode-speed claim didn't reproduce (quantized KV was
actually ~10% *faster* at raw decode here, not slower) — but prefill told a
different, much larger story: quantized KV cost 44% of prefill throughput
compared to `f16`/`f16`, and prefill dominates wall-clock on large prompts
(the full curl round-trip nearly halved, 20m48s → 11m47s, on this one test).
Draft (MTP) acceptance also improved meaningfully with unquantized KV
(91.1% → 97.0%), plausibly because the draft head predicts better against
un-quantized K/V state. `f16`/`q8_0` (tested as a VRAM middle ground) turned
out to be dominated — worse decode than either symmetric option, less
prefill gain than full `f16`/`f16`, and only modest VRAM savings — not worth
keeping.

VRAM was initially assumed to be a growing risk as a session's actual token
count approaches the 131072 ceiling — re-checked directly and that's wrong:
`llama-server` pre-allocates the KV buffer for the full configured
`--ctx-size` at model-load time, confirmed by measuring VRAM immediately
after startup (before any request) at 21.5 GB for `q8_0`/`q4_1` — already
close to its post-request value. VRAM headroom is therefore a one-time
budget check at whatever `context_size`/KV-type combination is configured,
not something that shrinks further as a conversation gets deeper. `f16`/`f16`
leaves ~20% headroom (vs. ~34% for `q8_0`/`q4_1`) — tighter, but stable, not
degrading with use.

## Files Modified
| File | Change |
|------|--------|
| `config.toml` | `[kv_cache] type_k`/`type_v`: `q8_0`/`q4_1` → `f16`/`f16`. Comment rewritten with the measured A/B/C numbers and the corrected VRAM-allocation understanding. |

## Behavior Changes
- Large-prompt requests (prefill-heavy, e.g. big agentic sessions with
  60K+-token context) should see substantially faster time-to-first-token.
- Steady-state decode tok/s at deep context drops slightly (~10%) — this
  matters less than prefill for typical agentic usage, where most wall-clock
  at depth is spent waiting on the prompt, not generation.
- Draft acceptance (speculative decoding hit rate) should improve, which
  partially offsets the raw decode-speed loss.
- VRAM headroom drops from ~34% to ~20% free on the 32 GB Arc B70. Confirmed
  stable (doesn't grow with session depth), but leaves less margin for any
  other concurrent GPU consumer (e.g. LM Studio running at the same time).

## Assumptions & Risks
- Single-run measurements per config, not averaged over multiple trials —
  real production variance (concurrent load, thermal state) wasn't
  controlled for. The prefill gap (72.8 → 132.0 tok/s) is large enough to be
  confident it's real; the decode gap (4.31 → 3.87 tok/s, ~10%) is smaller
  and closer to normal run-to-run noise.
- Only tested one context depth (83,098 tokens, 63% of max). Behavior at
  the true ceiling (approaching 131072) wasn't directly measured — VRAM
  pre-allocation theory suggests it should hold, but not empirically
  re-confirmed at 100%+ depth.
- If VRAM pressure becomes a problem in practice (e.g. `bad allocation` or
  `device lost` recurs), the first revert candidate is back to
  `q8_0`/`q4_1` — the config comment documents the full comparison so this
  isn't a guess next time.
