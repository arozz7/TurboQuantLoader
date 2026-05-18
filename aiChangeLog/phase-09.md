# Phase 09 — Conversation Logging

## Goal
Capture full prompt and response content per inference request into a separate
daily-rotating JSONL file, kept independent of the operational tracing log.
Disabled by default; opt-in via `log_conversations = true` in `config.toml`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/conversation_log.rs` | New — `ConversationLogger`, `ConversationEntry`, `LogMessage`; daily-rotating JSONL writer with date-change detection and no-op mode |
| `src/config/logging.rs` | Added `log_conversations: bool` (default `false`) |
| `src/main.rs` | Added `mod conversation_log` |
| `src/server/mod.rs` | Added `conv_logger: Arc<ConversationLogger>` to `AppState`; created in `serve()` |
| `src/server/routes/openai.rs` | `streaming_response()` accepts `request_id`, `log_messages`, `conv_logger`; collects tokens into `response_buf`; writes `ConversationEntry` on `Done` |
| `src/server/routes/anthropic.rs` | Same changes to `streaming_response()`; uses `request_id` from handler instead of internal `new_id` |
| `config.toml` | Added `log_conversations = false` to `[logging]` |

---

## Log File Format

**Location:** `<log_dir>/conversations.<YYYY-MM-DD>.jsonl`

**One JSON line per completed streaming request:**
```json
{
  "ts": "2025-05-17T10:23:45Z",
  "id": "chatcmpl-abc123",
  "model": "Qwen3.6-27B-Q4_K_S",
  "protocol": "openai",
  "stream": true,
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Explain transformer attention..."}
  ],
  "response": "The transformer attention mechanism works by...",
  "prompt_tokens": 2048,
  "completion_tokens": 312,
  "tps": 17.1,
  "finish_reason": "stop"
}
```

Messages logged are the **prepared messages** (after tool injection into the system prompt), so the log reflects exactly what the model received.

## Enabling

```toml
# config.toml
[logging]
log_conversations = true
```

Restart the server. The file is created on the first completed request.

## Notes
- Only streaming requests are captured (non-streaming bypasses the token channel).
- The `ConversationLogger` is a no-op (no file opened, no allocation) when disabled.
- Errors in the logger (disk full, permission denied) are emitted as `WARN` tracing events and never propagate to the inference path.
- The daily rotation uses the same `log_retention_days` cleanup sweep as the main log (both files share the `log_dir`).
- `tps` and token counts come from `llama-server`'s usage stats; TTFT and generation_ms are in the main tracing log correlated by `id`.
