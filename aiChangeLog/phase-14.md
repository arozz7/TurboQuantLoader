# Phase 14 — Separate `reasoning_content` from `content` (OpenAI route)

## Goal
Stop merging chain-of-thought (`<think>...</think>`) text into the same `content`
field as the model's actual answer on the OpenAI-compatible route, for both streaming
and non-streaming responses. Additionally forward real token-usage numbers on the
streaming path so clients that track context-window usage (e.g. an agent's
"X/128K tokens used" indicator) have real data to read.

## Why
The pi coding agent (external tool, `~/.pi/agent/models.json`, `"reasoning": true` on
its TurboQuantLoader model entry) showed noticeably worse "move along after thinking"
behavior against this server than against LM Studio, at similar raw tokens/sec.
Root cause: `src/server/routes/openai.rs`'s streaming handler matched
`ParsedEvent::TextToken(text) | ParsedEvent::ThinkingToken(text)` as a single arm,
sending both through the same `delta.content` field with no boundary marker — the
industry convention (llama.cpp, vLLM, LM Studio, OpenRouter, DeepSeek) is to stream
reasoning via a separate `delta.reasoning_content` field so clients can distinguish
"still thinking" from the real answer. `DeltaMessage` had no such field at all.

A second, related gap: TurboQuantLoader already requests
`stream_options: {"include_usage": true}` from llama-server and already captures the
real `prompt_tokens`/`completion_tokens` internally (`GenerateEvent::Done`), but never
forwarded that back out to OpenAI clients on the streaming path — `ChatCompletionChunk`
had no `usage` field. This is very likely why context-size tracking looked wrong to
pi even after the reasoning_content fix.

Our own `coding-agent` project (`J:\Projects\coding-agent`, `provider: turboquant` →
`llm/cloud_api_client.py`'s OpenAI-compatible path) only ever reads `delta.content` /
`message.content`, so it never noticed the merge — this was in effect "optimized for"
that client's tolerance of merged output at pi's expense.

## Files Modified
| File | Change |
|------|--------|
| `src/server/types/openai.rs` | Added `reasoning_content: Option<String>` to `DeltaMessage`. Added `stream_options: Option<StreamOptions>` (`include_usage: bool`) to `ChatCompletionRequest`. Added `usage: Option<Usage>` to `ChatCompletionChunk`. `Usage` now derives `Clone` (previously only used by the unused `ChatCompletionResponse`). |
| `src/server/routes/openai.rs` | Streaming: `TextToken`/`ThinkingToken` now emit separate delta chunks (`content` vs. `reasoning_content`). When the request sets `stream_options.include_usage`, an extra final chunk with empty `choices` and populated `usage` is sent before `[DONE]` (opt-in — clients that don't ask for it, including our own coding-agent, see zero behavior change). Non-streaming: added `split_reasoning()`, a complete-string equivalent of the streaming `StreamParser`'s `<think>` handling — rewrites the proxied JSON so `message.content` is thinking-stripped and `message.reasoning_content` is populated, falling back to the raw upstream bytes untouched on any parse failure. Added 4 unit tests for `split_reasoning`. |

## Behavior Changes
- OpenAI-route clients that don't inspect `reasoning_content` (our own coding-agent)
  now receive only the final answer in `content` — previously they received thinking
  text interleaved into it. This is a visible change for any consumer that was
  relying on seeing the raw `<think>` trace in `content` (none identified in this
  codebase's own `coding-agent` client, which only logs/returns whatever `content`
  contained).
- Clients that set `stream_options.include_usage: true` now receive a real usage
  chunk; clients that don't, see nothing new.
- `<tool_call>` XML handling in the non-streaming path is deliberately unchanged —
  `split_reasoning` only strips `<think>` blocks, leaving any `<tool_call>` text as-is
  in `content` (see `split_reasoning_preserves_tool_call_tags_in_visible_content` test).

## Companion change (separate repo)
`J:\Projects\coding-agent\llm\cloud_api_client.py`'s `_openai_generate` (non-streaming,
used for `provider: turboquant`) now mirrors `llm/ollama_client.py`'s existing guard:
if `content` comes back empty but `reasoning_content` has data (model exhausted its
`max_tokens` budget mid-`<think>`), raises `RuntimeError` with a clear message instead
of silently returning an empty string. This closes a regression the reasoning_content
split would otherwise have introduced for that caller. Tracked in that repo, not this
one's `aiChangeLog`.

## Assumptions & Risks
- Real usage forwarding only covers TurboQuantLoader's own streaming and
  non-streaming responses; whether pi (or any other client) actually sets
  `stream_options.include_usage` was not verified end-to-end in this session — the
  fix makes the data available on our side but relies on the client requesting it.
- `split_reasoning` (non-streaming) only extracts a single `<think>...</think>` block
  — matches the streaming `StreamParser`'s behavior, which also only recognizes one
  thinking block per response, consistent with Qwen3.x's chat template.
