# Phase 15 — Fix silent tool-call drop on malformed JSON

## Goal
Stop silently discarding a model's attempted tool call when its JSON is malformed,
and recover more malformed variants than before.

## Why
Observed in production: the pi coding agent stopped mid-task with no visible error.
Server logs showed the model emitted `{"name="bash", "arguments": {"command":"true"}}`
(a missing colon — `"name="bash"` instead of `"name": "bash"`). The strict JSON parse
in `try_parse_tool_call` failed as expected, falling through to the fallback
name-extraction loop — but that loop searched for the literal substring `"name"`
(quote-n-a-m-e-**quote**), which never matched `"name=` (quote-n-a-m-e-**equals**), so
`ext_name` stayed `None` and the whole function returned `None`. The caller
(`stream_parser.rs`, both the streaming `</tool_call>` handler and `flush()`) only
logged a `tracing::warn!` on `None` and pushed no event at all — the entire buffered
tool-call text vanished. The stream then ended with llama-server's own
`finish_reason: stop` and empty `content`, which looks to a client exactly like "the
assistant chose to say nothing and finished normally," not an error — hence pi had
nothing to act on or retry.

## Files Modified
| File | Change |
|------|--------|
| `src/server/stream_parser.rs` | Added `extract_value_after_key()` — replaces the old literal `"name"`/`"function"` substring search with one that tolerates the key/value separator being `:`, `=`, whitespace, or the key's closing quote being dropped entirely (only the JSON *value* itself must still be well-formed). Both call sites that previously discarded content on total parse failure (`process()`'s `</tool_call>` handler and `flush()`) now emit the raw buffered text as a `ParsedEvent::TextToken` instead of dropping it, so a response can no longer silently collapse to empty content. Added 2 regression tests: `tool_call_with_equals_separator_still_parses_name`, `unparseable_tool_call_emits_raw_text_instead_of_dropping`. |

## Behavior Changes
- A tool-call JSON with a mistyped/missing separator (`=` instead of `:`, or similar)
  now recovers the tool `name`. Its `arguments`, however, are not reliably recovered
  in this pass — see Assumptions & Risks below; a bare `{}` is emitted rather than the
  correct argument object in that case.
- Any tool-call JSON that fails to parse *entirely* (no recognizable name/function key
  at all) now surfaces as visible text instead of vanishing. This changes what the
  client sees in that edge case from "empty response, finish_reason=stop" to "raw
  attempted-tool-call text, finish_reason=stop" — strictly more information, never
  less, but it is a visible content change for that failure mode specifically.

## Assumptions & Risks
- The pre-existing fallback *arguments* extraction (separate code path from the name
  extraction fixed here) has its own bug: it always slices from the first `{` to the
  *last* `}` in the buffer regardless of nesting depth, so for a buffer like
  `{"name="bash", "arguments": {"command":"true"}}` it fails to produce
  `{"command":"true"}` and falls back to an empty `{}`. This bug predates this phase
  and was previously unreachable in this exact failure mode (the function returned
  `None` before ever reaching the arguments-extraction code) — fixing the name
  extraction makes this pre-existing arguments limitation reachable more often than
  before. Not fixed in this phase; flagged as a known follow-up.
- This fallback path is inherently best-effort for malformed model output; broadening
  it further (e.g. a real tolerant mini-parser instead of string-search widening)
  would be a larger, separate change if more malformed variants show up in practice.
