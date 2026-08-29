# Phase 23 — Recover more malformed tool calls: merged key/value, mismatched brace matching

## Goal
Improve on the phase-15 "never silently drop a malformed tool call" fix —
a warning was observed in production that still fell all the way through
to the raw-text fallback when it should have been fully recoverable:

```
WARN failed to parse tool_call JSON — emitting raw text instead of dropping it
json={"function=read", "arguments": {"limit": 116, "offset": 20, "path": "J:/Projects/quake-remake/PLAN-phase2.md"}}
```

## Why
Two separate bugs in `src/server/stream_parser.rs`'s fallback tool-call
recovery, both exposed by this one input:

1. **`extract_value_after_key` couldn't handle a merged key/value.**
   The phase-15 fix already tolerated `"name="bash"` (key and value in
   separate quote pairs, missing colon). This case is more mangled —
   `"function=read"` is a *single* JSON string literal, key and value
   sharing one quote pair with no separate opening quote for the value at
   all. The old code required `rest.strip_prefix('"')?` to succeed (a
   fresh opening quote for the value) and returned `None` otherwise,
   so this shape fell straight through to "no name found."
2. **The nested-arguments search used `buf.rfind('}')` (last brace in the
   whole buffer) instead of finding the brace that actually matches each
   candidate `{`.** With a well-formed `arguments: {...}` object nested
   inside a malformed outer wrapper, this pairs the *inner* `{` with the
   *outer* closing `}`, producing an unbalanced slice (`{...}}`, one open
   vs. two closes) that fails to parse — so even if bug 1 were the only
   issue, `arguments` would have silently come back as `{}`, losing the
   real `limit`/`offset`/`path` values.

## Files Modified
| File | Change |
|------|--------|
| `src/server/stream_parser.rs` | `extract_value_after_key`: after skipping the separator, also try stripping a leading quote from the *value* if present, but fall back to reading up to the next quote either way — recovers the merged-quote-pair case without changing behavior for the already-working `"name": "bash"` and `"name="bash"` cases. Added `find_matching_brace(buf, open_idx)` — a depth-counting, string-aware scanner — and switched the fallback `arguments` search to use it instead of `buf.rfind('}')`. 1 new unit test reproducing the exact production input (asserts both the recovered name *and* all three recovered argument values). |

## Behavior Changes
- The reported input now parses fully: `name: "read"`, `arguments: {"limit": 116, "offset": 20, "path": "J:/Projects/quake-remake/PLAN-phase2.md"}` — a real, executable tool call instead of raw text the client can't act on.
- Any other malformed tool call with a well-formed nested object (not just
  `arguments`) benefits from the brace-matching fix too, since it's a
  general-purpose scanner, not specific to this one key.
- The raw-text fallback (phase-15) is unchanged and still the last resort
  for inputs that aren't recoverable at all — this phase only shrinks the
  set of cases that fall through to it.

## Assumptions & Risks
- `find_matching_brace` treats `\` only as an escape character while inside
  a string (matching JSON's own escaping rules) — correct for well-formed
  string contents even with embedded `\"` or `\\`, but a string value with
  truly pathological unescaped content could still confuse it, same
  residual risk as any lightweight recovery scanner working on non-JSON
  input.
- Not yet deployed — same file-lock constraint as phases 21/22.
