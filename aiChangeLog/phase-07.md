# Phase 07 — Client-Driven Model Selection

## Goal
Allow connecting agents to specify which model to load by passing its name in the
standard OpenAI `model` field. The server resolves the name against a named registry
(config) or a filesystem scan, then hot-swaps `llama-server` in the background.
Requests that arrive during the swap receive HTTP 503 with `Retry-After: 10`.

---

## Files Modified

| File | Change |
|------|--------|
| `src/config/model.rs` | Added `ModelDefinition` struct; added `default_model: Option<String>` to `ModelConfig` |
| `src/config/mod.rs` | Added `models: Vec<ModelDefinition>` to `AppConfig`; exposed `ModelDefinition` |
| `src/model/registry.rs` | Added `ModelRegistry::resolve()` and `ModelRegistry::matches_current()` |
| `src/server/mod.rs` | Added `switching: Arc<AtomicBool>` to `AppState`; added `current_model_name()` and `trigger_model_switch()` methods |
| `src/server/routes/openai.rs` | Added switching guard and auto-switch logic to `chat_completions`; added `switching_503()` helper |
| `src/server/routes/anthropic.rs` | Same guard and auto-switch logic in `create_message` |
| `src/server/routes/models.rs` | Rewritten to return all models from named registry + `models_dir` scan |
| `src/server/types/openai.rs` | Removed `#[allow(dead_code)]` from `ChatCompletionRequest.model`; updated doc comment |
| `config.toml` | Added `[[models]]` named registry section with `qwen3-27b` entry and commented examples |

---

## New Behavior

### Named model registry (`[[models]]` in `config.toml`)
Each entry has:
- `name` — short identifier (the value clients send as `model`)
- `path` — absolute path to the GGUF file
- Optional overrides: `context_size`, `n_gpu_layers`, `main_gpu`, `batch_size`, `tensor_split`

### Resolution order when a client sends `model: "some-name"`
1. Exact case-insensitive match on `name` in `[[models]]`
2. Substring match on `name` in `[[models]]`
3. File-stem substring scan under `models_dir`
4. Name unknown → serve with currently-loaded model (no switch, no error)

### Auto-switch flow
1. Chat request arrives with `model: "llama3-8b"`
2. Handler compares to current loaded model (stem of `model_path`)
3. If different and resolvable: `trigger_model_switch()` fires a background Tokio task that kills the old `llama-server`, updates `AppConfig`, starts a new process, then clears `switching`
4. Handler returns **HTTP 503** with `Retry-After: 10` so the client retries
5. Any concurrent request during the switch also gets 503

### `GET /v1/models`
Now returns the full available set:
- All named `[[models]]` registry entries (with `"active": true` on the current model)
- Additional GGUF files discovered in `models_dir` not already in the registry

---

## Assumptions & Risks
- Model switching is synchronous to the server restart (30–180 s). Clients must tolerate 503 and retry.
- Only one switch runs at a time — if another switch is already in flight, new switch requests return 503 immediately (no queuing).
- If the requested name matches no known model, the server silently serves the currently-loaded model rather than erroring.
- `default_model` field is parsed from config but not yet wired to startup selection (reserved for phase 08).
