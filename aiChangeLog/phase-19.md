# Phase 19 — Watchdog: detect fatal GPU errors that don't kill the process; wire up `--cache-ram`

## Goal
Fix a real production outage: a Vulkan `device lost` error on 2026-08-27 left
`llama-server` running-but-broken for ~14 minutes with zero automatic recovery,
because the crash watchdog only detects process **exit**, not "alive but every
request fails." Separately, wire up the `[kv_cache] memory_budget_mb` config
field, which was defined but never actually passed to llama-server.

## Why
Investigating `logs/turboquant.2026-08-27.log`, a `device lost on Vulkan1` /
`ErrorDeviceLost` at 00:19:43 was followed by ~14 minutes of every subsequent
request failing with the identical error, with **no** `watchdog: llama-server
crashed — restarting` log line anywhere — the recovery at 00:33:37 was a
`shutdown signal received` (a manual restart), not the watchdog.

Root cause: `ProcessState::Crashed` (`src/server/llama_process.rs`) is only
set in one place — when the subprocess's stderr stream closes, meaning the
process exited (`spawn_child`'s stderr-forwarding task). A Vulkan device-lost
error doesn't kill the `llama-server` process; it keeps its HTTP server up
and keeps answering requests, just failing every decode. Stderr never closes,
`ProcessState` stays `Ready`, and `crash_watchdog` (`server/mod.rs`) — which
only polls for `ProcessState::Crashed` — never has anything to act on.

Registry-side, this was confirmed **not** a recurrence of the earlier TDR
issue (phase not yet numbered — `TdrDelay`/`TdrDdiDelay` fix): `TdrDelay` is
still 60 in the registry, and Windows' System event log has zero events
(no driver-reset/TDR event) in the crash window, meaning the OS itself never
detected a GPU dispatch timeout this time. This looks like a different,
memory-pressure-flavored failure — `n_tokens = 108998` (~83% of the
131072 context) and a `prompt state size 8259.523 MiB exceeds cache size
limit 8192.000 MiB` warning right at the same depth — but the watchdog gap is
real and worth fixing regardless of the exact GPU-side root cause.

Separately, while reading `build_args`, `[kv_cache].memory_budget_mb` was
defined in config but never emitted as a CLI flag — llama-server was silently
using its own `--cache-ram` default (8192 MiB, which matches the exact number
in the crash log) with no way to change it from `config.toml`.

## Files Modified
| File | Change |
|------|--------|
| `src/server/llama_process.rs` | Added `is_fatal_backend_error()` — checks a forwarded stderr line for `"device lost"` / `"ErrorDeviceLost"`. The stderr-forwarding loop in `spawn_child` now calls it per line and flips `ProcessState::Crashed` immediately (same transition as a real exit), so the existing watchdog restart logic (`server/mod.rs::crash_watchdog`) fires without any changes there. Also wired `config.kv_cache.memory_budget_mb` to llama-server's `--cache-ram <MB>` flag when set (omitted when `None`, preserving llama-server's own default). 4 new unit tests. |

## Behavior Changes
- A Vulkan (or any) `device lost` error now triggers an automatic watchdog
  restart within one 5s watchdog tick, instead of leaving the server silently
  broken until someone notices and restarts it manually.
- `memory_budget_mb` in `[kv_cache]` is no longer a dead config field — set it
  to explicitly control llama-server's prompt-cache RAM ceiling. Left unset,
  behavior is unchanged (llama-server's own 8192 MiB default applies).

## Assumptions & Risks
- `is_fatal_backend_error` matches on two known substrings seen in this
  session's actual crash logs. Other fatal-but-process-survives GPU errors
  (a different Vulkan error string, a CUDA equivalent) won't be caught unless
  added later — this is a targeted fix for the observed failure mode, not a
  general "any GPU error is fatal" classifier.
- `--cache-ram` (prompt-cache **host RAM**, used for fast prefix-reuse
  checkpoints) is a different memory pool than the **GPU VRAM** the
  `device lost` crash actually came from. Wiring it up gives explicit control
  over the host-RAM side and stops the log warning about cache size in a
  targeted way, but it does not by itself address VRAM pressure at very deep
  context (~100K+ tokens) — that would need `context_size`/`batch_size`
  tuning, which wasn't changed here since it directly trades off against the
  large-context use case this server is configured for.
- Not yet re-verified against a live recurrence (no way to safely reproduce a
  real Vulkan device-lost on demand) — watch the logs for the next occurrence
  to confirm the watchdog now restarts automatically.
