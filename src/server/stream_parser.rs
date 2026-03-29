//! Streaming token parser for Qwen3.5 special markup.
//!
//! Qwen3.5 embeds structured markers directly in the token stream:
//!
//! | Marker | Meaning |
//! |--------|---------|
//! | `<think>` … `</think>` | Chain-of-thought block — map to Anthropic `thinking` content |
//! | `<tool_call>` … `</tool_call>` | Function call JSON — map to Anthropic `tool_use` content |
//!
//! Tokens can straddle tag boundaries (e.g. a token that starts with `</` and
//! the next token completes `think>`).  The parser maintains a small lookahead
//! buffer to detect these cases reliably.

const LOOKAHEAD: usize = 32;

/// The parser state machine state.
#[derive(Debug, Clone, PartialEq)]
enum State {
    /// Emitting plain text — between `</think>` and the next `<think>` or end.
    Text,
    /// Inside `<think>…</think>` — emitting thinking tokens.
    Thinking,
    /// Inside `<tool_call>…</tool_call>` — buffering JSON.
    ToolCall { buffer: String },
}

/// Events emitted by the [`StreamParser`].
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedEvent {
    /// A plain text token to emit as a `text_delta` content block event.
    TextToken(String),
    /// A thinking token to emit as a `thinking_delta` content block event.
    ThinkingToken(String),
    /// `</think>` was detected — close the thinking block, open the text block.
    ThinkingEnd,
    /// A complete `<tool_call>` JSON object is ready.
    ToolCallReady { name: String, arguments: serde_json::Value },
}

/// Stateful parser that transforms a raw token stream into [`ParsedEvent`]s.
///
/// Feed each token string from the model via [`push`]; it returns zero or more
/// events.  Call [`flush`] after the model signals `Done` to drain any
/// buffered partial content.
pub struct StreamParser {
    state: State,
    /// Small lookahead buffer used to detect tags that span token boundaries.
    lookahead: String,
}

impl StreamParser {
    pub fn new() -> Self {
        Self { state: State::Text, lookahead: String::new() }
    }

    /// True if the parser is currently inside a thinking block.
    pub fn is_thinking(&self) -> bool {
        self.state == State::Thinking
    }

    /// Process one raw token from the model.
    ///
    /// Returns a (possibly empty) list of [`ParsedEvent`]s produced by this token.
    pub fn push(&mut self, token: &str) -> Vec<ParsedEvent> {
        self.lookahead.push_str(token);
        let mut events = Vec::new();
        self.process(&mut events);
        events
    }

    /// Drain any remaining buffered content as the appropriate event type.
    ///
    /// Must be called once after the model signals generation is complete.
    pub fn flush(&mut self) -> Vec<ParsedEvent> {
        let mut events = Vec::new();
        if self.lookahead.is_empty() {
            return events;
        }
        let text = std::mem::take(&mut self.lookahead);
        match &mut self.state {
            State::Thinking => events.push(ParsedEvent::ThinkingToken(text)),
            State::ToolCall { buffer } => {
                // Incomplete tool call — best-effort: try to parse what we have.
                buffer.push_str(&text);
                if let Some(evt) = try_parse_tool_call(buffer) {
                    events.push(evt);
                }
            }
            State::Text => {
                if !text.is_empty() {
                    events.push(ParsedEvent::TextToken(text));
                }
            }
        }
        events
    }

    // ── Internal processing ───────────────────────────────────────────────────

    fn process(&mut self, events: &mut Vec<ParsedEvent>) {
        loop {
            match &self.state {
                State::Text => {
                    if let Some(pos) = self.lookahead.find("<think>") {
                        // Emit text before the tag.
                        let before = self.lookahead[..pos].to_string();
                        if !before.is_empty() {
                            events.push(ParsedEvent::TextToken(before));
                        }
                        self.lookahead = self.lookahead[pos + "<think>".len()..].to_string();
                        self.state = State::Thinking;
                        // Continue processing the remainder in Thinking state.
                    } else if let Some(pos) = self.lookahead.find("<tool_call>") {
                        let before = self.lookahead[..pos].to_string();
                        if !before.is_empty() {
                            events.push(ParsedEvent::TextToken(before));
                        }
                        self.lookahead =
                            self.lookahead[pos + "<tool_call>".len()..].to_string();
                        self.state = State::ToolCall { buffer: String::new() };
                    } else {
                        // No tag detected yet.  Keep LOOKAHEAD bytes buffered
                        // to handle tags that span token boundaries; emit the rest.
                        let safe_len = self.lookahead.len().saturating_sub(LOOKAHEAD);
                        if safe_len > 0 {
                            // Ensure we split on a char boundary.
                            let split = floor_char_boundary(&self.lookahead, safe_len);
                            let emit = self.lookahead[..split].to_string();
                            self.lookahead = self.lookahead[split..].to_string();
                            if !emit.is_empty() {
                                events.push(ParsedEvent::TextToken(emit));
                            }
                        }
                        break;
                    }
                }

                State::Thinking => {
                    if let Some(pos) = self.lookahead.find("</think>") {
                        let thinking_text = self.lookahead[..pos].to_string();
                        if !thinking_text.is_empty() {
                            events.push(ParsedEvent::ThinkingToken(thinking_text));
                        }
                        self.lookahead =
                            self.lookahead[pos + "</think>".len()..].to_string();
                        self.state = State::Text;
                        events.push(ParsedEvent::ThinkingEnd);
                        // Continue processing remainder in Text state.
                    } else {
                        // No closing tag yet — keep LOOKAHEAD buffered, emit the rest.
                        let safe_len = self.lookahead.len().saturating_sub(LOOKAHEAD);
                        if safe_len > 0 {
                            let split = floor_char_boundary(&self.lookahead, safe_len);
                            let emit = self.lookahead[..split].to_string();
                            self.lookahead = self.lookahead[split..].to_string();
                            if !emit.is_empty() {
                                events.push(ParsedEvent::ThinkingToken(emit));
                            }
                        }
                        break;
                    }
                }

                State::ToolCall { .. } => {
                    if let Some(pos) = self.lookahead.find("</tool_call>") {
                        let json_fragment = self.lookahead[..pos].to_string();
                        let remainder =
                            self.lookahead[pos + "</tool_call>".len()..].to_string();
                        self.lookahead = remainder;

                        let buffer = if let State::ToolCall { buffer } = &mut self.state {
                            let mut b = std::mem::take(buffer);
                            b.push_str(&json_fragment);
                            b
                        } else {
                            unreachable!()
                        };

                        self.state = State::Text;

                        if let Some(evt) = try_parse_tool_call(&buffer) {
                            events.push(evt);
                        } else {
                            tracing::warn!(json = %buffer, "failed to parse tool_call JSON");
                        }
                        // Continue processing remainder in Text state.
                    } else {
                        // Buffer the tool call body; no safe-emit because we need the
                        // full JSON before we can do anything with it.
                        let body = std::mem::take(&mut self.lookahead);
                        if let State::ToolCall { buffer } = &mut self.state {
                            buffer.push_str(&body);
                        }
                        break;
                    }
                }
            }
        }
    }
}

impl Default for StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Try to parse a tool call JSON buffer into a [`ParsedEvent::ToolCallReady`].
fn try_parse_tool_call(buf: &str) -> Option<ParsedEvent> {
    let trimmed = buf.trim();
    let v: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    
    // Safely extract `arguments`
    let mut arguments = v.get("arguments").cloned().unwrap_or(serde_json::Value::Object(
        serde_json::Map::new(),
    ));

    // Handle stringified arguments (OpenAI format produced by some models)
    if let Some(s) = arguments.as_str() {
        if let Ok(parsed) = serde_json::from_str(s) {
            arguments = parsed;
        } else {
            tracing::warn!(raw_args = %s, "failed to un-stringify tool arguments; passing as raw string");
        }
    }

    tracing::info!(tool_call_name = %name, args = %arguments, "Successfully parsed model tool call output");

    Some(ParsedEvent::ToolCallReady { name, arguments })
}

/// Find the largest byte index ≤ `index` that falls on a UTF-8 char boundary.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(parser: &mut StreamParser, tokens: &[&str]) -> Vec<ParsedEvent> {
        let mut all = Vec::new();
        for t in tokens {
            all.extend(parser.push(t));
        }
        all.extend(parser.flush());
        all
    }

    #[test]
    fn plain_text_passes_through() {
        let mut p = StreamParser::new();
        let evts = feed(&mut p, &["Hello, ", "world!"]);
        assert!(evts.iter().all(|e| matches!(e, ParsedEvent::TextToken(_))));
        let combined: String = evts
            .iter()
            .filter_map(|e| if let ParsedEvent::TextToken(t) = e { Some(t.as_str()) } else { None })
            .collect();
        assert_eq!(combined, "Hello, world!");
    }

    #[test]
    fn thinking_block_produces_thinking_tokens_then_end() {
        let mut p = StreamParser::new();
        let evts = feed(&mut p, &["<think>", "reasoning here", "</think>", "answer"]);

        let thinking: Vec<_> = evts
            .iter()
            .filter(|e| matches!(e, ParsedEvent::ThinkingToken(_)))
            .collect();
        assert!(!thinking.is_empty());

        assert!(evts.iter().any(|e| *e == ParsedEvent::ThinkingEnd));

        let text: Vec<_> = evts
            .iter()
            .filter(|e| matches!(e, ParsedEvent::TextToken(_)))
            .collect();
        assert!(!text.is_empty());
    }

    #[test]
    fn tag_spanning_token_boundary() {
        let mut p = StreamParser::new();
        // Split <think> across two tokens
        let evts = feed(&mut p, &["prefix<th", "ink>thought</think>suffix"]);

        assert!(evts.iter().any(|e| matches!(e, ParsedEvent::ThinkingToken(_))));
        assert!(evts.iter().any(|e| *e == ParsedEvent::ThinkingEnd));
    }

    #[test]
    fn tool_call_parsed_correctly() {
        let mut p = StreamParser::new();
        let evts = feed(
            &mut p,
            &[
                "Let me read that.\n<tool_call>\n",
                r#"{"name": "Read", "arguments": {"file_path": "/foo.rs"}}"#,
                "\n</tool_call>",
            ],
        );

        let tool_evt = evts.iter().find(|e| matches!(e, ParsedEvent::ToolCallReady { .. }));
        assert!(tool_evt.is_some());
        if let Some(ParsedEvent::ToolCallReady { name, arguments }) = tool_evt {
            assert_eq!(name, "Read");
            assert_eq!(arguments["file_path"], "/foo.rs");
        }
    }

    #[test]
    fn no_thinking_block_emits_only_text() {
        let mut p = StreamParser::new();
        let evts = feed(&mut p, &["Just a plain response."]);
        assert!(evts.iter().all(|e| matches!(e, ParsedEvent::TextToken(_))));
        assert!(!evts.iter().any(|e| *e == ParsedEvent::ThinkingEnd));
    }
}
