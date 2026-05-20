# Phase 5 — Claude Code Agent Mode

**Status:** Implemented (Tracks A, B, C complete)
**Goal:** Make TurboQuantLoader a functional Claude Code backend — fast enough for interactive
use and capable of executing tools.

Three independent tracks that can land separately:

| Track | Blocker removed | Expected speedup / unlock |
|-------|-----------------|--------------------------|
| A — KV prefix cache | 3+ min prefill per request | Follow-up messages < 30 s |
| B — Tool use | Model ignores tools | Claude Code can read/write/run code |
| C — Thinking streaming | Thinking suppressed / causes timeout | Thinking tokens stream as Anthropic `thinking` blocks |

---

## Context: What We Learned in Phase 4 Dogfooding

Connecting Claude Code (`ANTHROPIC_BASE_URL=http://127.0.0.1:7432`) exposed three issues:

1. **`try_send` lost `Done` events** — fixed in Phase 4.1 hot-patch (`blocking_send`)
2. **Qwen3.5 thinking tokens** — `<think>...</think>` consumed entire token budget before
   any visible output; fixed with empty `<think></think>` prefix in `format_messages`
3. **7 800-token prefill on every request** — `ctx.clear_kv_cache()` is called at the start
   of every `Generate` command, erasing the KV state for Claude Code's system prompt + tools.
   Each request re-processes ~7 000 tokens of unchanged content from scratch.

Items 1 and 2 are already shipped (hot-patches). Item 2's fix — the empty `<think>\n\n</think>`
prefix — is a workaround that suppresses thinking entirely. Track C replaces it with proper
thinking support. Items 3, tool use, and thinking are the Phase 5 deliverables.

---

## Track A — KV Prefix Cache

### The Problem

`do_generate` calls `ctx.clear_kv_cache()` before every request. Claude Code sends the
same system prompt + full tool definitions on every message (~7 000 tokens). Prefilling
7 000 tokens takes ~3 minutes on Qwen3.5-35B-A3B. This happens even if the only change
between requests is one new user sentence.

### How llama.cpp Prefix Caching Works

llama.cpp does not re-decode tokens whose KV entries are already in the cache.
If we keep the cache between requests and only submit tokens that are *new*, the engine
starts generating from where it left off. The catch: we must track which tokens are in
the cache and verify the new prompt starts with exactly those tokens before skipping the
prefill.

### Implementation Plan

#### A.1 — `PrefixCache` tracker (`src/kv_cache/prefix_cache.rs`)

```rust
/// Tracks which prompt tokens are currently resident in the llama.cpp KV cache
/// so the generation loop can skip re-decoding the common prefix.
pub struct PrefixCache {
    /// Token IDs of the last successfully decoded prompt (positions 0..len).
    cached_tokens: Vec<i32>,
}

impl PrefixCache {
    pub fn new() -> Self { ... }

    /// Returns how many tokens from `new_tokens` are already in the cache.
    /// A return value of 0 means "start from scratch — clear before decoding".
    pub fn common_prefix_len(&self, new_tokens: &[i32]) -> usize {
        self.cached_tokens
            .iter()
            .zip(new_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Call after successful generation with the full prompt token sequence.
    pub fn update(&mut self, prompt_tokens: Vec<i32>) {
        self.cached_tokens = prompt_tokens;
    }

    /// Call when the context is cleared (e.g. context overflow, explicit reset).
    pub fn invalidate(&mut self) {
        self.cached_tokens.clear();
    }
}
```

#### A.2 — Modify `LlamaCppBackend`

Add `prefix_cache: PrefixCache` as a field (lives inside the inference thread — no sync needed).

In `do_generate`:
```rust
// Current (clears everything):
ctx.clear_kv_cache();
// ... decode all prompt tokens from position 0

// New:
let n_past = prefix_cache.common_prefix_len(&prompt_tokens);
if n_past == 0 {
    ctx.clear_kv_cache();
}
// ... decode only prompt_tokens[n_past..] starting at position n_past
prefix_cache.update(prompt_tokens);
```

When `n_past > 0`, the batch only contains the new tokens with positions `n_past..`.
llama.cpp reuses the KV entries for positions `0..n_past` automatically.

#### A.3 — Context overflow handling

If `prompt_tokens.len() > n_ctx`, clear the cache and truncate the oldest conversation
turns until the prompt fits. Log a warning. Invalidate `prefix_cache`.

#### A.4 — Conversation-turn eviction fallback

If common prefix is very short (< 512 tokens — i.e. the system prompt changed),
treat it as a full cache miss: clear and re-decode from scratch.

#### A.5 — Tests

- Unit test: `PrefixCache::common_prefix_len` with matching / mismatching / empty prefix
- Integration: send two requests with the same system prompt, assert the second request's
  log line shows `n_past > 0` (skipped prefill tokens)

### Expected Impact

| Request | Without prefix cache | With prefix cache |
|---------|---------------------|-------------------|
| First message | ~3.5 min (7 800 tok prefill) | ~3.5 min (cold, unavoidable) |
| Follow-up | ~3.5 min (full prefill again) | ~15–30 s (new turn only) |
| Tool result reply | ~3.5 min | ~10–20 s (tool result is short) |

---

## Track B — Tool Use

### The Problem

Claude Code sends a `tools` array with every request (Read, Write, Bash, Glob, Grep, etc.).
The server currently ignores this array. Qwen3.5 generates plain text. Claude Code receives
text content but expects `tool_use` content blocks, so tools never execute.

### Qwen3.5 Tool Call Format

Qwen3.5 (and Qwen3) supports function calling via a documented XML envelope:

**System prompt injection:**
```
# Tools

You may call one or more functions to assist with the user query.

<tools>
[{"name": "Read", "description": "...", "parameters": {...}}, ...]
</tools>

For each function call return a JSON object with function name and arguments
within <tool_call></tool_call> XML tags:

<tool_call>
{"name": "function_name", "arguments": {"arg1": "value1"}}
</tool_call>
```

**Model response containing a tool call:**
```
I need to read that file first.
<tool_call>
{"name": "Read", "arguments": {"file_path": "/path/to/file.rs"}}
</tool_call>
```

**Tool result injected as user turn:**
```
<tool_response>
{"output": "file contents here..."}
</tool_response>
```

### Implementation Plan

#### B.1 — Tool injection into system prompt (`src/server/routes/anthropic.rs`)

In `to_chat_request()`: if `req.tools` is non-empty, append the Qwen3.5 tool block
to the system message content:

```rust
fn build_tool_system_injection(tools: &[serde_json::Value]) -> String {
    let json = serde_json::to_string_pretty(tools).unwrap_or_default();
    format!(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         <tools>\n{json}\n</tools>\n\n\
         For each function call return a JSON object with function name and arguments \
         within <tool_call></tool_call> XML tags:\n\n\
         <tool_call>\n{{\"name\": \"function_name\", \"arguments\": {{\"arg1\": \"value1\"}}}}\n\
         </tool_call>"
    )
}
```

#### B.2 — Tool result messages (`src/server/types/anthropic.rs`)

Add `tool_result` variant to `ContentBlock`:

```rust
pub enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: Vec<ContentBlock> },
}
```

In `into_text()`: convert `ToolResult` blocks to `<tool_response>` XML so the model
sees them correctly in the conversation history:

```rust
ContentBlock::ToolResult { content, .. } => {
    let result_text = content.iter().map(|c| c.into_text()).collect::<String>();
    format!("<tool_response>\n{result_text}\n</tool_response>")
}
```

#### B.3 — Tool call parser (`src/server/tool_call_parser.rs`)

A streaming state machine that processes token text as it arrives from the model:

```rust
pub enum ParserState {
    /// Emitting regular text content.
    Text,
    /// Collecting tokens inside a <tool_call>...</tool_call> block.
    InToolCall { buffer: String },
}

pub enum ParsedOutput {
    /// Regular text to emit as content_block_delta.
    Text(String),
    /// Parsed tool call — emit as tool_use content block.
    ToolCall { name: String, arguments: serde_json::Value },
    /// Start of <tool_call> tag — switch state, suppress from text output.
    ToolCallStart,
    /// End of </tool_call> tag — ready to emit ToolCall.
    ToolCallEnd,
}
```

Key cases:
- Token contains `<tool_call>`: flush current text block, enter `InToolCall`
- Token contains `</tool_call>`: parse buffer as JSON, emit `ToolCall`, return to `Text`
- Token straddles the tag boundary: handle by keeping a small lookahead window

#### B.4 — Wire parser into streaming SSE response (`src/server/sse.rs`)

Modify `streaming_response()` to wrap the `ReceiverStream` with the parser:

```rust
// When tools are present, tokens go through the parser before being emitted as SSE events.
// The parser may hold back text while collecting a <tool_call> block and then emit
// a tool_use content block event instead.
```

New SSE event types needed for tool_use (already in Anthropic spec):
```
content_block_start  {"type": "tool_use", "id": "toolu_...", "name": "Read", "input": {}}
content_block_delta  {"type": "input_json_delta", "partial_json": "..."}   (optional streaming)
content_block_stop   {}
```

For Phase 5 initial implementation: emit tool_use blocks as complete (non-streaming input)
after the `</tool_call>` is fully parsed. Streaming input JSON deltas can come in Phase 6.

#### B.5 — `tool_use` ID generation

Each tool call needs a unique ID (`toolu_<uuid>`). Generate with `uuid::Uuid::new_v4()`.

#### B.6 — `stop_strings` addition

Add `"</tool_call>"` to stop strings so generation pauses at the end of each tool call,
allowing the server to emit the tool_use block before the model continues:

```rust
stop_strings: vec!["<|im_end|>".into(), "</tool_call>".into()],
```

After emitting the tool_use block, re-invoke generation to continue the response.
This requires a multi-turn generation loop inside the handler for a single HTTP request.

#### B.7 — Multi-turn generation loop

```
loop:
  1. Generate until stop_string or max_tokens
  2. If stopped on </tool_call>:
       a. Parse and emit tool_use content block
       b. Continue generation (don't close the SSE stream yet)
  3. If stopped on <|im_end|> or max_tokens:
       a. Emit content_block_stop, message_delta, message_stop
       b. Close stream → exit loop
```

This means the `BackendCommand::Generate` design needs to support resuming generation
(continuing from n_past without re-decoding the prompt). The prefix cache from Track A
makes this efficient.

---

## Track C — Thinking Streaming

### The Problem

Qwen3.5-35B-A3B is a thinking model. It generates `<think>...</think>` before every response.
The current workaround adds an empty `<think>\n\n</think>` prefix to the assistant turn, which
suppresses thinking entirely. This discards the model's strongest capability — and breaks for
users who explicitly want CoT reasoning.

The right fix: parse `<think>...</think>` blocks from the token stream and emit them as
Anthropic `thinking` content blocks so Claude Code (and any Anthropic SDK client) can render
them correctly.

### Anthropic Extended Thinking SSE Format

A response with thinking uses two content blocks:

```
content_block_start  {"type": "content_block_start", "index": 0,
                       "content_block": {"type": "thinking", "thinking": ""}}
content_block_delta  {"type": "content_block_delta", "index": 0,
                       "delta": {"type": "thinking_delta", "thinking": "<token>"}}
  … × N thinking tokens …
content_block_stop   {"type": "content_block_stop", "index": 0}

content_block_start  {"type": "content_block_start", "index": 1,
                       "content_block": {"type": "text", "text": ""}}
content_block_delta  {"type": "content_block_delta", "index": 1,
                       "delta": {"type": "text_delta", "text": "<token>"}}
  … × M response tokens …
content_block_stop   {"type": "content_block_stop", "index": 1}

message_delta        {"stop_reason": "end_turn", "usage": {"output_tokens": N+M}}
message_stop
```

The `signature` field is required on real Anthropic responses (used to prevent tampering
when thinking is relayed in subsequent turns). For a local server, use a deterministic
HMAC-SHA256 of the thinking content keyed with a server-side secret. When thinking is
replayed in a subsequent `messages` request, verify the signature before trusting the block.

### Implementation Plan

#### C.1 — Revert the thinking suppression in `engine.rs`

Remove the `<think>\n\n</think>` prefix from `format_messages`. The stream parser (C.3)
replaces the workaround permanently.

```rust
// Before (workaround):
out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n");

// After:
out.push_str("<|im_start|>assistant\n");
```

#### C.2 — New SSE types for thinking (`src/server/types/anthropic.rs`)

```rust
// content_block_start for thinking
pub struct ThinkingBlockStartData {
    pub r#type: &'static str,  // "thinking"
    pub thinking: &'static str, // "" (empty — content arrives via deltas)
}

// content_block_delta for thinking token
pub struct ThinkingDelta {
    pub r#type: &'static str,  // "thinking_delta"
    pub thinking: String,
}

pub struct ContentBlockDeltaThinkingEvent {
    pub r#type: &'static str,  // "content_block_delta"
    pub index: u32,
    pub delta: ThinkingDelta,
}

// message_start.message.usage needs cache_read_input_tokens / cache_creation_input_tokens
// for extended thinking clients — add as optional fields, default 0
```

#### C.3 — `StreamParser` — unified `<think>` + `<tool_call>` state machine (`src/server/stream_parser.rs`)

Consolidate the thinking and tool-call parsers (Tracks B and C) into one state machine:

```rust
pub enum ParseState {
    /// Normal text output → content_block_delta (text_delta)
    Text,
    /// Inside <think>...</think> → content_block_delta (thinking_delta)
    Thinking,
    /// Inside <tool_call>...</tool_call> → buffer until </tool_call>
    ToolCall { buffer: String },
}

pub enum ParsedEvent {
    TextToken(String),
    ThinkingToken(String),
    ThinkingEnd,          // </think> detected — close thinking block, open text block
    ToolCallReady { name: String, arguments: serde_json::Value },
}
```

Transitions:
- `Text` + sees `<think>` → `Thinking` (emit `ThinkingStart`)
- `Thinking` + sees `</think>` → `Text` (emit `ThinkingEnd`)
- `Text` + sees `<tool_call>` → `ToolCall { buffer: "" }` (emit `ToolCallStart`)
- `ToolCall` + sees `</tool_call>` → `Text` (emit `ToolCallReady`)

Tag-boundary handling: keep a 32-byte accumulator. When a token partially overlaps a tag
boundary, hold it in the accumulator until the next token resolves the ambiguity.

#### C.4 — Wire parser into streaming SSE (`src/server/routes/anthropic.rs`)

Replace the current direct `ReceiverStream` → SSE mapping with a parser-driven one:

```rust
// State carried across token events:
let mut parser = StreamParser::new();
let mut block_index: u32 = 0;
let mut thinking_open = false;
let mut text_open = false;

// On ParsedEvent::ThinkingToken(t):
//   If !thinking_open: emit content_block_start(thinking, index=block_index)
//                      thinking_open = true
//   Emit content_block_delta(thinking_delta, index=block_index, thinking=t)

// On ParsedEvent::ThinkingEnd:
//   Emit content_block_stop(index=block_index)
//   block_index += 1
//   thinking_open = false

// On ParsedEvent::TextToken(t):
//   If !text_open: emit content_block_start(text, index=block_index)
//                  text_open = true
//   Emit content_block_delta(text_delta, index=block_index, text=t)

// On ParsedEvent::ToolCallReady{name, arguments}:
//   Emit content_block_start(tool_use, id="toolu_<uuid>", name, input={})
//   Emit full arguments as single input_json_delta
//   Emit content_block_stop
```

#### C.5 — Thinking in `message_start` usage

The Anthropic SDK expects `message_start.message.usage` to include
`cache_read_input_tokens` and `cache_creation_input_tokens` when thinking is active.
Add these as optional `u32` fields (default 0) to `InputUsage`.

#### C.6 — Signature generation

```rust
use hmac::{Hmac, Mac};
use sha2::Sha256;

fn sign_thinking(thinking: &str, server_secret: &[u8]) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(server_secret).unwrap();
    mac.update(thinking.as_bytes());
    base64::encode(mac.finalize().into_bytes())
}
```

The server secret is generated once on startup (or read from config). Add `hmac`, `sha2`,
and `base64` to `Cargo.toml`. The signature goes on the `content_block_stop` event for the
thinking block (match the real Anthropic API shape).

#### C.7 — Thinking in subsequent request turns

When an incoming `messages` request contains a content block with `type: "thinking"`,
it must be re-injected into the prompt. Format it as:

```
<think>
{block.thinking}
</think>
```

This tells the model "here is your previous thinking" so it can continue coherently. The
signature should be verified before including (prevents prompt injection via forged thinking).

---

## Phase 5 File Plan

### New Files
| File | Purpose |
|------|---------|
| `src/kv_cache/prefix_cache.rs` | `PrefixCache` tracker (Track A) |
| `src/server/stream_parser.rs` | Unified `<think>` + `<tool_call>` state machine (Tracks B + C) |

### Modified Files
| File | Changes |
|------|---------|
| `src/model/llama_cpp.rs` | Add `PrefixCache` to inference thread; skip prefill for cached prefix (A) |
| `src/kv_cache/mod.rs` | Export `PrefixCache` (A) |
| `src/inference/engine.rs` | Remove `<think>\n\n</think>` suppression prefix (C.1) |
| `src/server/types/anthropic.rs` | Add `ToolUse`, `ToolResult`, `ThinkingBlock` variants; `thinking_delta` SSE event (B + C) |
| `src/server/routes/anthropic.rs` | Inject tool definitions; wire unified parser; multi-turn loop; thinking block state (B + C) |
| `src/server/sse.rs` | Add `thinking_block_start_event()`, `thinking_delta_event()`, `tool_use_event()` helpers (B + C) |
| `Cargo.toml` | Add `hmac`, `sha2`, `base64` for thinking signature (C.6) |

---

## Sequencing

Track A first — fast round-trips are required before iterating on parser logic.
Track C before Track B — `<think>` parsing is simpler than tool-call JSON parsing and
lets us validate the unified parser infrastructure before adding tool-call complexity.

Suggested order:
1. `PrefixCache` struct + unit tests (A)
2. Wire into `do_generate`, verify with Claude Code timing (A)
3. Remove thinking suppression prefix (C.1)
4. `StreamParser` with `<think>` only — no tool call yet (C.3 partial)
5. Wire thinking into SSE — verify Claude Code displays thinking blocks (C.4, C.5)
6. Add signature (C.6)
7. Add `<tool_call>` parsing to `StreamParser` (B.3, C.3 complete)
8. Tool injection into system prompt (B.1)
9. `ToolResult` + thinking relay handling (B.2, C.7)
10. Multi-turn generation loop (B.7)
11. End-to-end test: `claude` reads a file via TurboQuantLoader with visible thinking

---

## Implementation Notes (actual vs. planned)

### What was implemented
- **Track A (KV Prefix Cache):** Fully implemented as planned. `PrefixCache` in `src/kv_cache/prefix_cache.rs`; wired into `do_generate` with `LlamaToken → i32` conversion at call sites.
- **Track C (Thinking streaming):** Fully implemented. `StreamParser` state machine with 32-byte lookahead; thinking tokens stream as Anthropic `thinking` content blocks. Thinking suppression prefix removed from `format_messages`.
- **Track B (Tool use):** Core implemented:
  - Tool injection: `tools_to_qwen3_json()` converts Anthropic `input_schema` → Qwen3.5 `parameters`; injected as `<tools>…</tools>` block in system prompt.
  - `tool_result` handling: `into_qwen3_text()` converts `tool_use` → `<tool_call>` and `tool_result` → `<tool_response>` markup in message history.
  - `ToolCallReady` → `tool_use` SSE content block, emitted with unique `toolu_<uuid>` ID.
  - `stop_reason: "tool_use"` set in `message_delta` when a tool call was detected.

### What was deferred
- **Thinking signature (C.6):** Signature generation / verification for relayed thinking blocks. Not needed for Claude Code to function; can be added if cross-request thinking integrity is required.
- **Thinking relay (C.7):** Re-injecting `type: "thinking"` blocks from incoming messages as `<think>…</think>`. Currently skipped by `into_qwen3_text()`.
- **`</tool_call>` stop string (B.6):** Qwen3.5 naturally ends assistant turns with `<|im_end|>` after tool calls, so this is not strictly needed. Model stops correctly in practice.
- **Multi-turn generation loop (B.7):** Not needed — each HTTP request handles one generation turn; Claude Code handles multi-turn via new requests with `tool_result` content.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Qwen3.5 may not follow `<tool_call>` format reliably at IQ3_XXS quantization | Test with a simple tool call before full integration |
| Partial tag boundary in token stream causes parser to miss tags | Keep a 32-byte lookahead window; flush on `<|im_end|>` |
| Multi-turn loop makes max_tokens hard to enforce | Track total generated tokens across all turns in the loop |
| Context grows with each tool result + thinking, eventually hitting n_ctx | Implement conversation truncation before Phase 5 is complete |
| Prefix cache invalidated by any system prompt change | Log `n_past` on every request to surface cache hit rate |
| Thinking signature validation adds latency on each request | Sign/verify is O(thinking_length) — negligible vs. generation time |
