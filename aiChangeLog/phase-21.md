# Phase 21 — Fail fast if the internal port is already held (orphan-process guard)

## Goal
Prevent the exact failure mode hit during today's KV-cache A/B testing: a
restart left an orphaned `llama-server.exe` bound to the internal port while
a new TurboQuantLoader instance spawned a second, unreachable copy —
silently, with health checks passing against the wrong process — and Pi's
requests stalled/retried for 37+ minutes with no visible error anywhere.

## Why
Root cause of the incident: `spawn_child()` never checked whether
`backend.internal_port` (7433) was actually free before spawning a new
`llama-server` child. When it wasn't (an earlier instance's child was
orphaned rather than cleanly killed — in this case because a `Bash`
background job's owning shell session was reaped between tool calls,
which is specific to how this session was launched, not a normal restart
path), the new child's own bind attempt failed, but `wait_until_ready()`'s
health poll happily got a `200` back from the *old*, still-running process
on that port and reported success. The new `LlamaProcess` believed it owned
a working backend it never actually controlled, while the stray process
kept serving real traffic underneath it — two full model copies loaded,
routing unpredictable, and no error logged anywhere to point at the actual
problem.

## Files Modified
| File | Change |
|------|--------|
| `src/server/llama_process.rs` | Added `check_port_available(port)` — binds `127.0.0.1:{port}` via `tokio::net::TcpListener`, immediately releases it if free, or returns a clear error if occupied. Called at the top of `spawn_child()`, so it runs on every path that starts or restarts the backend (`LlamaProcess::start`, `restart`, and therefore the crash watchdog and both admin endpoints). 2 new async unit tests. |

## Behavior Changes
- Starting TurboQuantLoader (or any restart path) while something else
  already holds the internal port now fails immediately with a clear error
  naming the port and suggesting how to find the stray process, instead of
  silently spawning a second, unreachable `llama-server` instance.
- No change to the normal restart path — `kill()` already releases the port
  before `spawn_child()` runs, so a legitimate restart is unaffected.

## Assumptions & Risks
- This closes the specific failure mode observed today (silent double-bind
  masked by a health check hitting the wrong process). It does not add
  automatic cleanup of a detected stray process — deliberately: killing an
  unidentified process bound to that port without knowing whether it's
  actually an orphaned `llama-server` or something unrelated the operator
  is running would be a worse failure mode than a loud, immediate error.
- Small bind/drop-then-spawn race window exists (another process could grab
  the port between the check and the actual spawn), but this only needs to
  catch the common case — a stray process already sitting on the port from
  a prior run — not defend against adversarial contention.
- Not yet deployed to the live binary — the running instance holds the
  current `target/release/turboquant-loader.exe` file locked, so applying
  this requires a restart (verified the release build compiles cleanly to
  an isolated target dir first).
