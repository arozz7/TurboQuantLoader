# Phase 4 — HTTP Server (OpenAI + Anthropic API)

## Files Created
| File | Purpose |
|------|---------|
| `src/server/error.rs` | `ApiError` — wraps `anyhow::Error` into a JSON HTTP 500 response |
| `src/server/sse.rs` | `sse_response()`, `data_event()`, `named_event()` — shared SSE helpers |
| `src/server/types/mod.rs` | Module declarations |
| `src/server/types/openai.rs` | OpenAI request/response/chunk types |
| `src/server/types/anthropic.rs` | Anthropic request/response/event types |
| `src/server/routes/mod.rs` | Route module declarations |
| `src/server/routes/models.rs` | `GET /v1/models` |
| `src/server/routes/openai.rs` | `POST /v1/chat/completions` (streaming + non-streaming) |
| `src/server/routes/anthropic.rs` | `POST /v1/messages` (streaming + non-streaming) |

## Files Modified
| File | Changes |
|------|---------|
| `src/server/mod.rs` | Rewrote: `AppState`, `build_router()`, `serve()` |
| `src/inference/stream.rs` | Added `into_inner()` to expose `GenerateStream` for SSE adapters |
| `src/main.rs` | `cmd_serve` now calls `server::serve()`; removed `bail` import |

## Endpoints

| Method | Path | Protocol | Streaming |
|--------|------|----------|-----------|
| `GET` | `/v1/models` | both | no |
| `POST` | `/v1/chat/completions` | OpenAI | yes (`"stream": true`) |
| `POST` | `/v1/messages` | Anthropic | yes (`"stream": true`) |

## Anthropic SSE Event Sequence
```
message_start → content_block_start → ping
content_block_delta × N
content_block_stop → message_delta → message_stop
```
All three terminal events are emitted atomically from the `Done` handler via
`flat_map`, ensuring correct ordering.

## Design Decisions
- **No auth** — `x-api-key` / `Authorization` headers accepted but not validated
- **Single router** — both endpoint families always active on the same port; no switch needed
- **`block_in_place`** — used around `engine.chat()` since `SyncSender::send` can briefly block
- **`tokio_stream::StreamExt`** for `filter_map` (sync); **`futures::StreamExt`** for `flat_map` (returns `Stream`)
- **`system` field** — top-level Anthropic `system` is prepended as a system `ChatMessage`
- **Tool use** — silently ignored (unknown fields dropped by serde); model may still emit tool-call text

## Assumptions and Risks
- `input_tokens` in `message_start` is always 0 — llama.cpp doesn't expose prompt token count synchronously before generation
- Non-streaming responses buffer the entire response in memory; fine for typical assistant outputs
- The inference thread processes one request at a time — concurrent HTTP requests queue at the `SyncSender` (capacity 4)
