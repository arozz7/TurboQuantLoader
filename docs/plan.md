# TurboQuantLoader — Full Implementation Plan

## Meta

| | |
|---|---|
| Language | Rust (edition 2021) |
| Default port | **7432** |
| Config format | TOML (`config.toml`) |
| Primary GPU | RTX 4070 Ti Super — 16 GB (tensor split primary) |
| Secondary GPU | RTX 2060 — 6 GB (tensor split secondary) |
| KV cache default | 4-bit (switchable to 3/2/8 via config) |
| API compatibility | OpenAI `/v1/chat/completions` |

---

## Model Under Test

**File:** `J:/llama/Models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf` (14.7 GB)
**Vision projector:** `mmproj-F32.gguf` (1.7 GB) — Phase 5

Architecture highlights:
- Mixture of Experts: 256 experts, 8 active per token (~3.5B active params at inference)
- Hybrid attention: 3× linear-attention + 1× full-attention repeated across 40 layers
  - Only **10 of 40 layers** use traditional KV cache — already much smaller than a dense transformer
- Max context: **262,144 tokens**
- GQA: 2 KV heads, head_dim = 256

KV cache size estimates (10 full-attention layers only):

| Context | FP16 | 4-bit compressed |
|---------|------|-----------------|
| 8k      | 160 MB | 40 MB |
| 32k     | 640 MB | 160 MB |
| 128k    | 2.5 GB | 625 MB |
| 262k    | 5.2 GB | 1.3 GB |

---

## TurboQuant Integration Reality

| Option | Status |
|--------|--------|
| `llama-cpp-2` — GGUF loading, IQ3_XXS, MoE, hybrid attention | Works today |
| `tq-kv` — TurboQuant KV compression (Candle-based) | Qwen3.5 MoE **not yet supported** |
| `turboquant` — `RealModelRunner` | Qwen3.5 MoE **not yet supported** |
| llama.cpp native KV quantization (`Q4_0`/`Q8_0`) | Works today — used as Track A |

**Strategy:**
- **Track A (immediate):** llama.cpp native KV cache quantization
- **Track B (upgrade):** `tq-kv` drop-in behind `turbo-kv` Cargo feature flag — activates when Qwen3.5 MoE support lands
- Interface is identical; swapping Track A → B requires zero caller changes

---

## Dependency Manifest

```toml
[package]
name    = "turboquant-loader"
version = "0.1.0"
edition = "2021"

[features]
default  = ["cuda"]
cuda     = ["llama-cpp-2/cuda"]
turbo-kv = ["tq-kv", "turboquant"]   # off until Qwen3.5 MoE supported
tui      = ["dep:ratatui", "dep:crossterm"]

[dependencies]
# Inference
llama-cpp-2          = { version = "0.1", features = [] }

# TurboQuant (optional)
tq-kv                = { version = "*", optional = true }
turboquant           = { version = "*", optional = true }

# HTTP server
axum                 = { version = "0.7", features = ["macros"] }
tokio                = { version = "1", features = ["full"] }
tokio-stream         = "0.1"

# Serialization
serde                = { version = "1", features = ["derive"] }
serde_json           = "1"
toml                 = "0.8"

# CLI
clap                 = { version = "4", features = ["derive", "env"] }

# Logging
tracing              = "0.1"
tracing-subscriber   = { version = "0.3", features = ["env-filter", "fmt"] }

# Errors
anyhow               = "1"
thiserror            = "1"

# GPU stats
nvml-wrapper         = "0.10"

# Async
futures              = "0.3"
uuid                 = { version = "1", features = ["v4"] }

# TUI (Phase 5, optional)
ratatui              = { version = "0.26", optional = true }
crossterm            = { version = "0.27", optional = true }
```

---

## Phase 1 — Foundation & Config

**Goal:** Compilable skeleton. All interfaces defined, nothing wired that isn't stubbed.
No inference runs in this phase.

### Tasks

#### 1.1 — Workspace & Project Scaffold
- `Cargo.toml` with all dependencies pinned
- Full directory tree created
- `CLAUDE.md` updated for this project
- `rust-toolchain.toml` pinning stable channel
- `.cargo/config.toml` with Windows CUDA build hints
- `.github/workflows/ci.yml` — `cargo check` + `cargo clippy` on push

#### 1.2 — `config/mod.rs` — Top-level `AppConfig`
```rust
pub struct AppConfig {
    pub server:   ServerConfig,
    pub model:    ModelConfig,
    pub kv_cache: KvCacheConfig,
}
// load_from_file(path: &Path) -> Result<AppConfig>
// apply_cli_overrides(&mut self, args: &CliArgs)
```

#### 1.3 — `config/server.rs`
```rust
pub struct ServerConfig {
    pub host:                    String,   // "127.0.0.1"
    pub port:                    u16,      // 7432
    pub max_concurrent_requests: usize,    // 4
    pub request_timeout_secs:    u64,      // 300
}
```

#### 1.4 — `config/model.rs`
```rust
pub struct ModelConfig {
    pub model_path:    PathBuf,         // required
    pub mmproj_path:   Option<PathBuf>, // Phase 5 vision
    pub models_dir:    PathBuf,         // scan root
    pub n_gpu_layers:  i32,             // -1 = all to GPU
    pub tensor_split:  Vec<f32>,        // [16.0, 6.0]
    pub context_size:  u32,             // 8192
    pub batch_size:    u32,             // 512
    pub threads:       u32,             // num_cpus / 2
}
```

#### 1.5 — `config/kv_cache.rs`
```rust
pub enum KvBits     { Two, Three, Four, Eight }
pub enum KvStrategy { LlamaNative, TurboQuant }

pub struct KvCacheConfig {
    pub bits:             KvBits,          // Four
    pub strategy:         KvStrategy,      // LlamaNative
    pub memory_budget_mb: Option<u32>,     // None = unlimited
}
```

#### 1.6 — `model/backend.rs` — `ModelBackend` trait
```rust
pub trait ModelBackend: Send + Sync {
    fn load(config: &ModelConfig) -> Result<Self> where Self: Sized;
    fn tokenize(&self, text: &str)        -> Result<Vec<i32>>;
    fn detokenize(&self, tokens: &[i32])  -> Result<String>;
    fn context_size(&self)                -> u32;
    fn model_name(&self)                  -> &str;
    fn generate(&self, req: GenerateRequest) -> Result<GenerateStream>;
    fn apply_kv_cache_config(&mut self, cfg: &KvCacheConfig) -> Result<()>;
}

pub struct GenerateRequest {
    pub prompt:       String,
    pub max_tokens:   u32,
    pub sampler:      SamplerParams,
    pub stop_strings: Vec<String>,
}

// GenerateStream = tokio::sync::mpsc::Receiver<GenerateEvent>
pub enum GenerateEvent {
    Token(String),
    Done(GenerateSummary),
    Error(String),
}

pub struct GenerateSummary {
    pub tokens_generated:  u32,
    pub tokens_per_second: f32,
    pub context_tokens:    u32,
}
```

#### 1.7 — `kv_cache/mod.rs` — `KvCacheBackend` trait
```rust
pub trait KvCacheBackend: Send + Sync {
    fn configure(&mut self, cfg: &KvCacheConfig) -> Result<()>;
    fn stats(&self) -> CacheStats;
    fn reset(&mut self);
}

pub struct CacheStats {
    pub used_mb:               f32,
    pub compression_ratio:     f32,   // 1.0 = no compression
    pub context_tokens:        u32,
    pub full_attention_layers: u32,   // 10 for Qwen3.5-35B-A3B
}
```

#### 1.8 — `model/registry.rs` — Model Discovery
```rust
pub struct ModelEntry {
    pub name:        String,
    pub path:        PathBuf,
    pub size_bytes:  u64,
    pub arch:        Option<String>,    // from GGUF header
    pub quant_type:  Option<String>,    // e.g. "IQ3_XXS"
    pub has_mmproj:  bool,
}

// ModelRegistry::scan(dir: &Path) -> Result<Vec<ModelEntry>>
//   - recurse for *.gguf
//   - exclude mmproj-*.gguf from main list
//   - mark entries where sibling mmproj-*.gguf exists
// ModelRegistry::find_by_name(name: &str) -> Option<&ModelEntry>
```

#### 1.9 — `main.rs` — CLI Entry Point
```rust
enum Command {
    Serve(ServeArgs),  // start HTTP API
    Run(RunArgs),      // interactive terminal chat
    Bench(BenchArgs),  // benchmark context sizes vs KV cache
    List,              // list discovered models
}
// Each command: load AppConfig → apply CLI overrides → dispatch
// Phase 1: all commands print "not yet implemented" except List
```

#### 1.10 — `config.toml` defaults
```toml
[server]
host                    = "127.0.0.1"
port                    = 7432
max_concurrent_requests = 4
request_timeout_secs    = 300

[model]
model_path   = "J:/llama/Models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf"
models_dir   = "J:/llama/Models"
n_gpu_layers = -1
tensor_split = [16.0, 6.0]
context_size = 8192
batch_size   = 512
threads      = 8

[kv_cache]
bits     = 4
strategy = "llama_native"
```

#### 1.11 — `aiChangeLog/phase-01.md`

---

## Phase 2 — Inference Engine

**Goal:** Model loads on both GPUs, tokens stream to terminal, `run` command works interactively.

### Tasks

#### 2.1 — `model/llama_cpp.rs` — `LlamaCppBackend`
Implements `ModelBackend` via `llama-cpp-2`.

Key setup:
- `LlamaParams`: `n_gpu_layers`, `tensor_split: [16.0, 6.0]`, CUDA devices 0+1
- Context params: `n_ctx = context_size`, `n_batch = batch_size`
- Chat template: read from GGUF metadata automatically (Qwen3.5 = ChatML format)
  ```
  <|im_start|>system\n{system}<|im_end|>\n
  <|im_start|>user\n{content}<|im_end|>\n
  <|im_start|>assistant\n
  ```
- `apply_kv_cache_config`: sets `cache_type_k` and `cache_type_v`

#### 2.2 — `inference/sampler.rs`
```rust
pub struct SamplerParams {
    pub temperature:   f32,   // 0.7
    pub top_p:         f32,   // 0.9
    pub top_k:         u32,   // 40
    pub min_p:         f32,   // 0.05
    pub repeat_penalty: f32,  // 1.1
    pub repeat_last_n: u32,   // 64
    pub seed:          Option<u64>,
}
impl Default for SamplerParams { ... }
```
Maps directly to llama-cpp-2 sampler chain.

#### 2.3 — `inference/stream.rs`
```rust
pub struct TokenStream {
    rx: mpsc::Receiver<GenerateEvent>,
}
impl TokenStream {
    pub async fn next_token(&mut self)              -> Option<GenerateEvent>
    pub async fn collect_full(&mut self) -> Result<(String, GenerateSummary)>
}
```
Generation runs in `tokio::task::spawn_blocking` (llama.cpp is synchronous C),
tokens forwarded through `mpsc` channel.

#### 2.4 — `inference/engine.rs` — `InferenceEngine`
```rust
pub struct InferenceEngine {
    backend:  Box<dyn ModelBackend>,
    kv_cache: Box<dyn KvCacheBackend>,
    config:   Arc<AppConfig>,
}
impl InferenceEngine {
    pub fn new(config: Arc<AppConfig>)                     -> Result<Self>
    pub fn format_messages(&self, msgs: &[ChatMessage])    -> String
    pub async fn chat(&self, req: ChatRequest)             -> Result<TokenStream>
}

pub struct ChatRequest {
    pub messages:     Vec<ChatMessage>,
    pub params:       SamplerParams,
    pub max_tokens:   u32,
    pub stop_strings: Vec<String>,
}
```

#### 2.5 — CLI `run` subcommand
- REPL loop: print `> `, read line, maintain history, call `engine.chat()`
- Stream tokens to stdout as they arrive
- On `Done`: print dim stats line `[42.3 tok/s · 312 tokens · 8.1 MB KV]`
- Special commands: `/quit`, `/clear` (reset history), `/stats`, `/model <name>`
- Clean Ctrl+C exit

#### 2.6 — Smoke test checklist (`docs/testing.md`)
- [ ] Model loads without OOM
- [ ] Both GPUs show utilization (Task Manager / nvidia-smi)
- [ ] Tokens stream to terminal
- [ ] `/stats` returns non-zero KV cache usage
- [ ] `/clear` resets context without crash
- [ ] Ctrl+C exits cleanly

#### 2.7 — `aiChangeLog/phase-02.md`

---

## Phase 3 — KV Cache Compression

**Goal:** KV compression active at runtime. Bit-width switchable. Bench command produces comparison table.

### Tasks

#### 3.1 — `kv_cache/llama_native.rs` — Track A

Bit-width mapping:
```
bits = 4  →  cache_type_k = Q4_0,  cache_type_v = Q4_0  (default)
bits = 8  →  cache_type_k = Q8_0,  cache_type_v = Q8_0
bits = 3  →  cache_type_k = Q4_0   (closest available, warn in logs)
bits = 2  →  cache_type_k = Q4_0   (same, warn in logs)
```

`CacheStats` population:
- `used_mb`: from `llama_get_state_size()`
- `compression_ratio`: `fp16_theoretical / actual`
- `context_tokens`: from `n_past`
- `full_attention_layers`: 10 (hardcoded for Qwen3.5-35B-A3B)

#### 3.2 — `kv_cache/turbo_quant.rs` — Track B (stub, `#[cfg(feature = "turbo-kv")]`)
```rust
#[cfg(feature = "turbo-kv")]
pub struct TurboQuantCache { inner: LlamaNativeCache }

#[cfg(feature = "turbo-kv")]
impl KvCacheBackend for TurboQuantCache {
    fn configure(&mut self, cfg: &KvCacheConfig) -> Result<()> {
        // TODO: wire tq-kv MultiHeadKVCache when Qwen3.5 MoE is added
        tracing::warn!("TurboQuant KV not yet supported for Qwen3.5 MoE — using LlamaNative");
        self.inner.configure(cfg)
    }
    // rest delegates to self.inner
}
```
Interface-identical to `LlamaNativeCache` — zero caller changes needed when support lands.

#### 3.3 — `kv_cache/mod.rs` factory
```rust
pub fn create_kv_cache(cfg: &KvCacheConfig) -> Box<dyn KvCacheBackend> {
    match cfg.strategy {
        KvStrategy::LlamaNative => Box::new(LlamaNativeCache::new()),
        #[cfg(feature = "turbo-kv")]
        KvStrategy::TurboQuant  => Box::new(TurboQuantCache::new()),
        #[cfg(not(feature = "turbo-kv"))]
        KvStrategy::TurboQuant  => {
            tracing::warn!("turbo-kv feature not compiled — using LlamaNative");
            Box::new(LlamaNativeCache::new())
        }
    }
}
```

#### 3.4 — `gpu_stats.rs`
```rust
pub struct GpuStats {
    pub device_index:    u32,
    pub name:            String,
    pub vram_used_mb:    u32,
    pub vram_total_mb:   u32,
    pub utilization_pct: u32,
}
pub fn query_all_gpus() -> Result<Vec<GpuStats>>
```
Used by bench command and `/stats` in `run`.

#### 3.5 — CLI `bench` subcommand
```
turboquant-loader bench
  [--context-sizes 1024,8192,32768]
  [--bits 4,8]
  [--output bench_results.json]
```
For each (context_size × bits) combination:
1. Load model with that KV config
2. Generate fixed prompt from `docs/bench_prompt.txt` to target context length
3. Record: `tok/s`, `vram_used_mb`, `kv_cache_mb`, `compression_ratio`
4. Print table:
```
Context   Bits   tok/s   VRAM (MB)   KV Cache   Ratio
──────────────────────────────────────────────────────
1024      fp16   47.3    15,200      82 MB      1.0x
1024      4-bit  48.1    15,200      21 MB      3.9x
8192      fp16   44.2    15,400      640 MB     1.0x
8192      4-bit  45.7    15,200      163 MB     3.9x
32768     fp16   38.1    16,800      2,560 MB   1.0x
32768     4-bit  40.3    15,900      655 MB     3.9x
```
5. Write JSON if `--output` specified.

#### 3.6 — `aiChangeLog/phase-03.md`

---

## Phase 4 — OpenAI API Server

**Goal:** Full OpenAI-compatible HTTP API at `http://127.0.0.1:7432`. Claude Code points here directly.

### Tasks

#### 4.1 — `server/types.rs` — OpenAI Schema Types

```rust
// Request
pub struct ChatCompletionRequest {
    pub model:       String,
    pub messages:    Vec<ApiMessage>,
    pub temperature: Option<f32>,
    pub top_p:       Option<f32>,
    pub max_tokens:  Option<u32>,
    pub stream:      Option<bool>,
    pub stop:        Option<StopCondition>,  // String or Vec<String>
    pub seed:        Option<u64>,
}

pub struct ApiMessage {
    pub role:    Role,           // "system" | "user" | "assistant"
    pub content: MessageContent, // Phase 5: extends to ContentPart array
}

pub enum Role            { System, User, Assistant }
pub enum MessageContent  { Text(String) }   // Phase 5: Parts(Vec<ContentPart>)

// Non-streaming response
pub struct ChatCompletionResponse {
    pub id:      String,   // "chatcmpl-{uuid}"
    pub object:  String,   // "chat.completion"
    pub created: i64,
    pub model:   String,
    pub choices: Vec<Choice>,
    pub usage:   UsageStats,
}
pub struct Choice {
    pub index:         u32,
    pub message:       ApiMessage,
    pub finish_reason: FinishReason,  // "stop" | "length"
}

// Streaming chunk
pub struct ChatCompletionChunk {
    pub id:      String,
    pub object:  String,   // "chat.completion.chunk"
    pub created: i64,
    pub model:   String,
    pub choices: Vec<ChunkChoice>,
}
pub struct ChunkChoice {
    pub index:         u32,
    pub delta:         Delta,
    pub finish_reason: Option<FinishReason>,
}
pub struct Delta {
    pub role:    Option<Role>,
    pub content: Option<String>,
}

pub struct UsageStats {
    pub prompt_tokens:     u32,
    pub completion_tokens: u32,
    pub total_tokens:      u32,
}

// Models endpoint
pub struct ModelListResponse {
    pub object: String,        // "list"
    pub data:   Vec<ModelObject>,
}
pub struct ModelObject {
    pub id:       String,
    pub object:   String,      // "model"
    pub created:  i64,
    pub owned_by: String,      // "local"
}

// Error
pub struct ApiError       { pub error: ApiErrorDetail }
pub struct ApiErrorDetail {
    pub message: String,
    pub r#type:  String,
    pub code:    Option<String>,
}
```

#### 4.2 — `server/chat.rs` — Chat Completions Handler

`POST /v1/chat/completions`

**Non-streaming:**
1. Validate (messages non-empty, model loaded)
2. Map `ApiMessage[]` → `ChatRequest`
3. `engine.chat(req).await`
4. `stream.collect_full().await` → full text + summary
5. Return `ChatCompletionResponse` as JSON 200

**Streaming (SSE):**
1. Same validation + mapping
2. `engine.chat(req).await`
3. Return `axum::response::sse::Sse` stream
4. `GenerateEvent::Token(t)` → emit `ChatCompletionChunk`
5. `GenerateEvent::Done` → emit final chunk with `finish_reason: "stop"`, then `data: [DONE]`
6. `GenerateEvent::Error` → emit error chunk, close stream

Error responses:
- 400: bad request (missing messages, invalid params)
- 503: model busy — includes `Retry-After: 5` header

#### 4.3 — `server/models.rs`

`GET /v1/models` — returns `ModelListResponse` from `ModelRegistry::scan(models_dir)`

#### 4.4 — `server/routes.rs`
```rust
pub fn build_router(engine: Arc<InferenceEngine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_handler))
        .route("/v1/models",           get(models_handler))
        .route("/health",              get(health_handler))
        .layer(CorsLayer::permissive())     // localhost only — permissive is fine
        .layer(TraceLayer::new_for_http())
        .with_state(engine)
}
```

`GET /health` response:
```json
{
  "status": "ok",
  "model": "Qwen3.5-35B-A3B-UD-IQ3_XXS",
  "context_used": 312,
  "context_max": 8192,
  "vram": [
    { "device": "RTX 4070 Ti Super", "used_mb": 14200, "total_mb": 16376 },
    { "device": "RTX 2060",          "used_mb": 3800,  "total_mb": 6144  }
  ]
}
```

#### 4.5 — `server/mod.rs` — Server Startup
```rust
pub async fn run_server(config: Arc<AppConfig>, engine: Arc<InferenceEngine>) -> Result<()>
// - bind to config.server.host:port
// - log: "TurboQuantLoader listening on http://127.0.0.1:7432"
// - log: "Claude Code: set API base URL to http://127.0.0.1:7432/v1"
// - tokio::select! on server + ctrl_c for graceful shutdown
```

#### 4.6 — Concurrency guard

`InferenceEngine` protected by `tokio::sync::Semaphore(max_concurrent_requests)`.
llama.cpp is not thread-safe for concurrent generation on the same context.
Queued requests wait; response includes `Retry-After: 5` if semaphore is exhausted.

#### 4.7 — `docs/claude-code-setup.md`
Configuration for Claude Code and other OpenAI-compatible clients:
```json
{
  "openai": {
    "apiKey": "local",
    "baseUrl": "http://127.0.0.1:7432/v1",
    "model": "Qwen3.5-35B-A3B-UD-IQ3_XXS"
  }
}
```

#### 4.8 — `aiChangeLog/phase-04.md`

---

## Phase 5A — Vision Support

**Goal:** Load `mmproj-F32.gguf` and handle image content in chat messages.

### Tasks

#### 5A.1 — `ModelConfig` already has `mmproj_path: Option<PathBuf>` from Phase 1.
`ModelRegistry::scan()` already marks models with sibling mmproj files.

#### 5A.2 — Extend `LlamaCppBackend`
- Load mmproj via `llama-cpp-2` clip model API
- Vision encoder stays on VRAM if space allows (14.7 GB model + 1.7 GB mmproj ≤ 22 GB total)

#### 5A.3 — Extend `server/types.rs`
```rust
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}
pub enum ContentPart {
    Text     { text: String },
    ImageUrl { url: ImageUrl },   // "data:image/jpeg;base64,..."
}
```
Matches OpenAI vision API format exactly.

#### 5A.4 — Extend `inference/engine.rs`
- Detect `ContentPart::ImageUrl` in messages
- Decode base64 → bytes
- Pass to llama.cpp clip encoder → image tokens
- Inject into prompt at correct position

#### 5A.5 — `aiChangeLog/phase-05a.md`

---

## Phase 5B — TUI

**Goal:** Full `ratatui` terminal UI replacing the plain REPL.

### Tasks

#### 5B.1 — `tui/app.rs` — State Machine
```rust
pub enum AppMode { Chat, ModelSelect, Help }

pub struct App {
    pub mode:          AppMode,
    pub messages:      Vec<TuiMessage>,
    pub input_buffer:  String,
    pub scroll_offset: u16,
    pub stats:         LiveStats,
    pub is_generating: bool,
}
pub struct LiveStats {
    pub tokens_per_sec:    f32,
    pub context_tokens:    u32,
    pub context_max:       u32,
    pub kv_cache_mb:       f32,
    pub compression_ratio: f32,
    pub gpu_stats:         Vec<GpuStats>,
}
```

#### 5B.2 — `tui/chat_view.rs`
- Scrollable message list (ratatui `Paragraph` + `Scrollbar`)
- User messages: right-aligned, dim border
- Assistant messages: left-aligned, full width
- Code blocks: detected by ` ``` ` fences, block border
- Streaming: assistant message updates in-place as tokens arrive

#### 5B.3 — `tui/status_bar.rs` — one line always visible
```
Model: Qwen3.5-35B-A3B │ 42.3 tok/s │ 312/8192 ctx │ KV: 21MB (3.9x) │ 4070Ti: 14.2/16GB │ 2060: 3.8/6GB
```

#### 5B.4 — Key Bindings

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | Newline in input |
| `Ctrl+C` | Quit |
| `PgUp` / `PgDn` | Scroll history |
| `Ctrl+L` | Clear conversation |
| `Esc` | Cancel generation |
| `F1` | Help overlay |
| `F2` | Model selector |

#### 5B.5 — Event loop
```rust
// tokio::select! on:
// - crossterm key events
// - token stream from engine
// - stats refresh timer (250 ms)
```

#### 5B.6 — `run` subcommand upgrade
- `--tui` flag or auto-detect TTY
- Falls back to plain REPL when TTY not available

#### 5B.7 — `aiChangeLog/phase-05b.md`

---

## Phase Summary

| Phase | Deliverable | Key Files | New Deps |
|-------|-------------|-----------|----------|
| 1 | Compilable skeleton, all traits | `config/`, `model/backend.rs`, `kv_cache/mod.rs`, `model/registry.rs` | clap, serde, toml, anyhow |
| 2 | Model loads + terminal chat | `model/llama_cpp.rs`, `inference/` | llama-cpp-2 |
| 3 | KV cache + bench command | `kv_cache/llama_native.rs`, `kv_cache/turbo_quant.rs`, bench cmd | nvml-wrapper |
| 4 | OpenAI API server | `server/` (all) | axum, tokio-stream |
| 5A | Vision (images in chat) | extend model + server + inference | — |
| 5B | ratatui TUI | `tui/` (all) | ratatui, crossterm |
