# Phase 08 — Logging Infrastructure

## Goal
Add persistent file-based logging with daily rotation, configurable retention,
and structured per-request log events so model loading, switching, and usage
patterns can be traced across sessions.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `Cargo.toml` | Added `tracing-appender = "0.2"` |
| `src/config/logging.rs` | New — `LoggingConfig` struct (`log_dir`, `log_retention_days`, `file_log_level`, `stdout_log_level`) |
| `src/config/mod.rs` | Added `logging: LoggingConfig` to `AppConfig`; exposed `LoggingConfig` |
| `src/main.rs` | Config loaded before tracing init; `init_tracing()` sets up dual subscriber (stdout + rolling file); `cleanup_old_logs()` deletes files past retention window at startup |
| `src/server/proxy.rs` | Added `model: String` param to `spawn_tracked_reader`; emits `request complete` log with tokens, TPS, TTFT, finish reason |
| `src/server/routes/openai.rs` | Emits `OpenAI chat request` log (model, stream, message count, tool count, max_tokens); passes model name to `spawn_tracked_reader` |
| `src/server/routes/anthropic.rs` | Emits `Anthropic messages request` log (same fields); passes model name to `spawn_tracked_reader` |
| `config.toml` | Added `[logging]` section |

---

## New Behaviour

### Log files
Daily rolling files written to `logs/` (configurable via `log_dir`):
```
logs/turboquant.2025-05-17.log
logs/turboquant.2025-05-18.log
...
```

### Retention
Files older than `log_retention_days` (default 7) are deleted at startup.
Set to `0` to keep all files indefinitely.

### Log levels
- `stdout_log_level` — controls terminal output; overridden by `RUST_LOG` env var.
- `file_log_level` — controls file output; independent of stdout/RUST_LOG.

Both accept RUST_LOG-style directives: `"info"`, `"debug"`,
`"turboquant_loader=debug,reqwest=warn"`, etc.

### Per-request structured events

**Request received** (both OpenAI and Anthropic routes):
```
INFO turboquant_loader::server::routes::openai: OpenAI chat request model="Qwen3.6-27B-Q4_K_S" stream=true messages=12 tools=3 max_tokens=32768
INFO turboquant_loader::server::routes::anthropic: Anthropic messages request model="Qwen3.6-27B-Q4_K_S" stream=true messages=12 tools=true max_tokens=4096
```

**Request complete** (emitted from proxy layer after streaming finishes):
```
INFO turboquant_loader::server::proxy: request complete model="Qwen3.6-27B-Q4_K_S" prompt_tokens=2048 completion_tokens=312 ttft_ms=423 generation_ms=18234 tps="17.1" finish_reason="stop"
```

**Model switch events** (existing, now visible in file logs):
```
INFO turboquant_loader::server::mod: model switch started model="J:/llama/Models/.../Qwen3.6-27B-Q6_K.gguf"
INFO turboquant_loader::server::mod: model switch complete model="J:/llama/Models/.../Qwen3.6-27B-Q6_K.gguf"
WARN turboquant_loader::server::mod: model switch blocked — model active within idle timeout model="other-model" idle_secs=42 required_idle_secs=1800
```

---

## Assumptions & Risks
- `tracing-appender`'s `WorkerGuard` is held in `main()` — logs flush cleanly on normal shutdown (Ctrl-C). An OS kill (`SIGKILL` / task manager) may lose the last few buffered lines.
- Log directory path is resolved relative to the **working directory at startup**, not the binary location. When run as a service, ensure the working directory is set appropriately.
- `cleanup_old_logs` uses file `mtime` (last-modified time) for age calculation. On Windows, NTFS tracks mtime accurately.
