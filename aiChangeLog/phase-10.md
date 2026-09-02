# Phase 10 — Qwen3.8-27B Upgrade + Prompt Cache Metrics

## Goal
Upgrade the local model and llama-server binary to support Qwen3.8-27B (released
Aug 13-14 2026), and add KV prefix cache hit-rate tracking to the metrics layer.

---

## Part 1: Qwen3.8-27B Model & Binary Upgrade

### Why
The previous `llama-server.exe` (build b9189, May 16 2026) predates Qwen3.8's
release and its two load-bearing features:
- **Gated DeltaNet hybrid attention** (48/64 layers linear-attn, 16/64 full-attn) —
  early llama.cpp builds had a silent CUDA-path output-corruption bug on these
  layers, fixed around commit `ece963f41`. Vulkan was reportedly unaffected, but
  b9189 predates the fix regardless.
- **MTP (multi-token prediction) draft heads** — support landed in PR #22673
  (merged July 2026), after b9189.

### Changes
| File | Change |
|------|--------|
| `J:/llama/llama-server.exe` (+ DLLs) | Upgraded b9189 → **b10488** (Aug 18 2026). Old binaries preserved at `J:/llama/_backup_b9189/` for rollback. |
| `config.toml` | `model.model_path` → `Qwen3.8-27B-Q4_K_S.gguf`; `model.mmproj_path` → Qwen3.8's vision mmproj; `backend.extra_flags` now enables MTP (`--spec-type draft-mtp --spec-draft-n-max 2 --parallel 1`) instead of `--spec-type none`; updated stale comments referencing Qwen3.6 sizing |

### Verification
- Loaded the GGUF directly against the new `llama-server.exe`: model loads clean,
  MTP draft context initializes, no corruption in generated output.
- Full stack smoke test via `turboquant-loader.exe serve`: TQL correctly spawns
  `llama-server` with the new binary/model/flags, `/health` reports
  `model: Qwen3.8-27B-Q4_K_S`, `state: Ready`; a chat completion through the
  `:7432 → :7433` proxy returned coherent output with MTP draft acceptance
  (48/62 and 6/6 tokens accepted across two test requests).

### Rollback
Point `config.toml: model.model_path` back to
`J:/llama/Models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q4_K_S.gguf` and restore
the binaries from `J:/llama/_backup_b9189/`.

---

## Part 2: Prompt Cache Hit-Rate Metrics

### Goal
Surface how much of each request's prompt is served from llama-server's KV
prefix cache, to help judge whether context reuse (e.g. repeated system
prompts / long conversation histories) is actually landing.

### Files Modified
| File | Change |
|------|--------|
| `src/metrics.rs` | Added `cached_tokens` to `RequestMetrics`; added `total_cached_tokens: AtomicU64` to `MetricsCollector`; added `cache_hit_rate()` |
| `src/model/backend.rs` | Added `cached_tokens: u32` to `GenerateSummary` |
| `src/server/proxy.rs` | `spawn_event_reader` and `spawn_tracked_reader` extract `cached_tokens` from llama-server's `usage.prompt_tokens_details.cached_tokens` (falling back to legacy `tokens_cached`) |
| `src/server/routes/openai.rs`, `anthropic.rs` | Non-streaming paths extract and record `cached_tokens` the same way |
| `src/server/routes/metrics.rs` | `/health`, `/metrics` (Prometheus), and `/v1/admin/stats` all expose a `prompt_cache` block (`total_cached_tokens`, `total_prompt_tokens`, `hit_rate_pct`); Prometheus adds `tql_cached_tokens_total` and `tql_cache_hit_rate` |

### Notes
- Reads `usage.prompt_tokens_details.cached_tokens` (llama.cpp ≥ b3900); falls
  back to the older top-level `tokens_cached` field for compatibility.
- Streaming and non-streaming paths, OpenAI and Anthropic routes, are all covered.

---

## Assumptions & Risks
- Qwen3.8-27B's vision path (`mmproj-F16.gguf`) was loaded but not functionally
  tested with an actual image request — text-only smoke tests only.
- b10488 is a point-in-time "latest"; llama.cpp ships frequently, so re-check
  before assuming this stays current.
- `turbo-kv` (TurboQuant native KV compression) remains gated off — it targets a
  different KV cache mechanism than Qwen3.8's Gated DeltaNet, so this upgrade
  doesn't change that feature's readiness.
