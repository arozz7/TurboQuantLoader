# Phase 18 — Revert `--no-mmap` / batch-ubatch bump after production crashes

## Goal
Stop repeated `bad allocation` crashes observed during live pi/coding-agent sessions
by reverting the two phase-13 changes most likely responsible for the increased
memory footprint, while keeping the changes that showed real measured gains.

## Why
Within ~3.5 minutes of a fresh restart, decode hit `got exception: bad allocation`
mid-generation. Immediately after, llama-server's prompt-cache subsystem also failed
to allocate and **self-reduced its budget from a normal size down to 180 MiB**
(`cache size limit reduced to 180.337 MiB`) — a reactive shrink in response to a real
allocation failure, not a config value we set. Every subsequent request then hit the
same wall with progressively less headroom (3283 → 1990 → 599 tokens before failing)
until a fatal `GGML_ASSERT(batch.slot_batched || batch.size() == 0)` crash killed the
process; the watchdog restarted it, and the same pattern repeated. This happened
twice in one session.

Considered and ruled out: a concurrently-running LM Studio contending for VRAM.
Checked LM Studio's own server log (`~/.lmstudio/server-logs/`) — its logged activity
started *after* the first crash and its one model-load attempt (cancelled before
completing) happened *after* TurboQuantLoader's watchdog had already restarted, so it
couldn't have caused the initial crash. Confirmed directly: the crash reproduced again
even with LM Studio killed.

That leaves the phase-13 config changes. Of those, `type_k`/`type_v` (KV quantization)
and the speculative-decoding tuning (`spec_draft_n_max=4` + `--spec-draft-p-min 0.75`)
showed real, measured improvements (draft acceptance 77-92% vs. the prior 52-60%,
higher decode floor at depth — see the performance check documented in conversation
history around 2026-08-23). The `batch_size`/`ubatch-size` bump (4096→8192,
2048→4096) and `--no-mmap` had **no measured benefit confirmed** — phase-13's own
changelog flagged both as "untested here whether this matters on Windows/Vulkan".
Larger batch/ubatch sizes mean larger compute buffers per decode step; `--no-mmap`
forces the full model into committed RAM/VRAM instead of OS-reclaimable mmap'd pages,
removing a pressure release valve. Reverting these two is the safe first step; if
crashes recur, `type_k` (currently `q8_0`, up from the original `q4_0`) is the next
lever to free VRAM, since it's a plausible secondary contributor (larger retained
context-checkpoint states — logs showed individual checkpoints of 450-555 MiB) even
though it wasn't the primary suspect.

## Files Modified
| File | Change |
|------|--------|
| `config.toml` | `[backend] extra_flags`: removed `--no-mmap`; `--ubatch-size` `4096` → `2048`. `[model] batch_size`: `8192` → `4096`. `spec_draft_n_max=4` + `--spec-draft-p-min 0.75` and `[kv_cache] type_k`/`type_v` = `q8_0`/`q4_1` left unchanged — both showed measured gains and neither is the primary suspect for the memory growth. |

## Behavior Changes
- Prefill throughput will likely regress somewhat from what the batch/ubatch bump
  measured (phase-16's "request complete" logging should make this visible going
  forward) — an intentional trade against stability. Steady-state decode tok/s and
  draft acceptance should be unaffected, since those came from the KV type and
  speculative-decoding changes, which are untouched.
- Model loading returns to using mmap (llama-server's own default `load_mode=auto`).

## Assumptions & Risks
- Not yet re-verified under a real long agent session post-revert — restart and
  monitor for recurrence. If `bad allocation` still occurs, the next candidates to
  revert in order: `type_k` (q8_0 → q4_0, largest remaining VRAM lever from phase 13),
  then `spec_draft_n_max`/`--spec-draft-p-min` (the MTP draft context itself holds
  additional KV buffers, scaling somewhat with n_max).
- The exact mechanism (VRAM vs. host RAM exhaustion, and why it manifests progressively
  worse across consecutive requests rather than failing immediately) wasn't root-caused
  beyond "one or both of these two changes increased peak memory footprint past a
  tipping point" — this phase treats the symptom via a targeted revert of the
  unverified changes, not a full memory-accounting investigation.
