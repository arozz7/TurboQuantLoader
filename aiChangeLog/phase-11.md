# Phase 11 — Per-Model Load Control

## Goal
Let each `[[models]]` entry override load-time flags (speculative decoding,
chat-template kwargs) and sampling defaults, falling back to `[backend]`
globals — so models with different requirements (e.g. Qwen's built-in
`draft-mtp` head vs. a future DeepSeek-V4-Flash entry needing an external
`draft-dspark` drafter file) can coexist without a global `extra_flags` string
that can't express both at once.

## Why
`extra_flags` in `[backend]` was global and verbatim — `config.toml` had
`--spec-type draft-mtp --spec-draft-n-max 2` hard-baked into it, which is only
correct for Qwen's built-in MTP draft head. Speculative decoding is
fundamentally per-model. Flag names (`--spec-type`, `--spec-draft-n-max`,
`--spec-draft-model`, `--chat-template-kwargs`) were verified against the
installed `llama-server.exe --help` output before implementation, not just
the vendor docs.

## Files Modified
| File | Change |
|------|--------|
| `src/config/model.rs` | Added `LoadConfig` struct (`spec_type`, `spec_draft_n_max`, `draft_model`, `chat_template_kwargs`, `temperature`, `top_p`, `min_p`, `extra_flags` — all optional) and `load: Option<LoadConfig>` on `ModelDefinition` |
| `src/config/backend.rs` | Promoted `spec_type` / `spec_draft_n_max` / `draft_model` out of `extra_flags` into typed `Option` fields on `BackendConfig`; added `chat_template_kwargs` / `temperature` / `top_p` / `min_p` as global fallbacks |
| `src/server/llama_process.rs` (`build_args`) | Emits `--spec-type`, `--spec-draft-n-max`, `--spec-draft-model`, `--chat-template-kwargs` from the merged `BackendConfig`; added unit tests for each |
| `src/server/mod.rs` | Added `apply_load_overrides` — merges a resolved `ModelDefinition.load` onto a cloned config's `[backend]` section (`Some` wins, `None` falls back to global); called from `trigger_model_switch` |
| `src/server/routes/openai.rs`, `anthropic.rs` | Sampling fallback chain changed to `request value → model's configured value (cfg.backend.*) → hardcoded default`, so per-model `temperature`/`top_p`/`min_p` actually take effect (previously the route handlers always sent an explicit value, so a load-time default was unreachable) |
| `src/server/proxy.rs` (`build_chat_body`) | Added optional `min_p` parameter, included in the JSON body only when set |
| `src/server/types/openai.rs`, `types/anthropic.rs` | Added `min_p: Option<f32>` to both request types |
| `src/model/registry.rs` | Added `is_drafter` (same pattern as `is_mmproj`) — excludes `dspark-*.gguf` drafter files from `ModelRegistry::scan` results so they don't appear as selectable models; added unit tests |
| `config.toml` | Moved `--spec-type draft-mtp --spec-draft-n-max 2` out of `extra_flags` into the new typed `[backend]` fields; added a commented example `[[models]]` entry showing `[models.load]` for a hypothetical DeepSeek-V4-Flash-0731 entry |

## Behavior Changes
- `[backend] extra_flags` no longer carries spec-decoding flags in the default
  config — same effective CLI args, different config surface.
- Chat completion requests that omit `temperature`/`top_p`/`min_p` now consult
  the active model's configured value before falling back to the hardcoded
  default (0.6 / 0.95 / none). No change in behavior for existing configs
  since no `[models.load]` sections are populated yet.
- `ModelRegistry::scan` (used by `list` and the `models_dir` fallback in
  `resolve`) now excludes `dspark-*.gguf` files.

## Assumptions & Risks
- `--spec-draft-model` and `--spec-type draft-dspark` are confirmed to exist
  in the currently-installed `llama-server.exe` (b10488), but end-to-end
  behavior with an actual DSpark drafter file was not tested — no
  DeepSeek-V4-Flash model is downloaded yet. The commented example in
  `config.toml` is illustrative, not verified against real weights.
- The `admin/load` HTTP endpoint (`POST /v1/admin/load`) takes a raw model
  path, not a registry name, so it does not resolve a `ModelDefinition` and
  therefore does not go through `apply_load_overrides`. Only
  `trigger_model_switch` (client `model:` field-triggered switch) does.
- No `tempfile` dependency was added; `is_drafter`/`is_mmproj` are tested as
  pure functions rather than via a filesystem-backed `scan` test.
