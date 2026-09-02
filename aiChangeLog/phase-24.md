# Phase 24 — Resolve model identifiers that are full filesystem paths

## Goal
Fix a recurring warning observed in production:

```
WARN model not found in registry or models_dir — ignoring switch request
model=J:/llama/Models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_S.gguf
```

## Why
`ModelRegistry::resolve` and `ModelRegistry::matches_current`
(`src/model/registry.rs`) only ever compared a requested model name against
*short* names — `config.models[].name` or a scanned file's stem (e.g.
`Qwen3.8-27B-Q4_K_S`). A client that sends the full GGUF path as `model`
can never match either: the path is far longer than any short name, so
neither the exact-match nor substring checks succeed.

Traced the actual client: a Pi Coding Agent server entry's cached model
`id` was `J:/llama/Models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_S.gguf`
— the full-path format llama-server's own (non-TQL) `/v1/models` reports,
picked up from when this client was briefly talking to llama-server's
internal port directly instead of through TQL. TQL's own `/v1/models`
(`src/server/routes/models.rs`) has always reported the short name, so any
client using *that* listing was unaffected.

Harmless in the single-model case observed (the switch is "ignored" and the
request proceeds against the already-loaded model, which happened to be the
correct one), but the resolver would silently fail to switch models at all
for any client sending a full path once more than one model is configured.

## Files Modified
| File | Change |
|------|--------|
| `src/model/registry.rs` | Added `normalize_model_query(name) -> String`: if the input contains a path separator, reduce it to its file stem (matching the short-name convention used everywhere else); otherwise return it unchanged. Applied at the top of `ModelRegistry::resolve` and `ModelRegistry::matches_current`, before any comparison. 3 new unit tests: path reduces to stem (both `/` and `\` separators), short names pass through untouched, `matches_current` now accepts a full path for the currently-loaded model and still rejects a full path for a different one. |

## Behavior Changes
- A `model` field containing a full GGUF path now resolves and matches
  exactly as if the short file-stem name had been sent instead.
- No change for clients that already send short names (the normalization
  only activates when the string contains `/` or `\`).

## Assumptions & Risks
- Only triggers on path-looking input (contains a separator) — a model
  short name that happens to contain a slash would be mis-handled, but no
  such name exists in this project's convention (GGUF file stems don't
  contain path separators).
- Not yet deployed — same file-lock-on-restart constraint as prior phases.
