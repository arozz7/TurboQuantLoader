# Phase 16 — "request complete" logging for non-streaming responses

## Goal
Log a per-request completion summary for non-streaming `/v1/chat/completions`
responses, matching the one the streaming path already had.

## Why
Reviewing today's logs to evaluate the phase-13 tuning changes showed only 1 of 22
completed OpenAI requests produced a `"request complete"` summary line
(`prompt_tokens`/`completion_tokens`/`ttft_ms`/`generation_ms`/`tps`/`finish_reason`).
That line is emitted by `spawn_tracked_reader` (`src/server/proxy.rs`), which only the
**streaming** code path uses. `non_streaming_response`
(`src/server/routes/openai.rs`) already computed the same numbers — for
`conv_logger.log()` and `metrics.record()` — but never logged them as a readable
summary line, so non-streaming traffic (apparently the majority of both pi's and our
own coding-agent's calls in this deployment) was invisible to a `grep "request
complete"` pass over the logs.

## Files Modified
| File | Change |
|------|--------|
| `src/server/routes/openai.rs` | `non_streaming_response` now emits a `tracing::info!(..., "request complete")` line with the same field set as the streaming path's, right before `conv_logger.log()`/`metrics.record()` (which already computed the values — no new work, just surfaced). `ttft_ms` is set equal to `generation_ms`, since a non-streaming response has no earlier partial signal — the whole body arrives at once. |

## Behavior Changes
- Purely additive logging — no change to request handling, response bodies, or
  metrics recording. `grep "request complete"` over the logs now shows both
  streaming and non-streaming completions with a consistent field set.

## Assumptions & Risks
- None — this is a log-line addition using data that was already being computed.
