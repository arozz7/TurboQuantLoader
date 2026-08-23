//! Streaming token parser for Qwen3.5 special markup.
//!
//! Qwen3.5 embeds structured markers directly in the token stream:
//!
//! | Marker | Meaning |
//! |--------|---------|
//! | `<think>` â€¦ `</think>` | Chain-of-thought block â€” map to Anthropic `thinking` content |
//! | `<tool_call>` â€¦ `</tool_call>` | Function call JSON â€” map to Anthropic `tool_use` content |
//!
//! Tokens can straddle tag boundaries (e.g. a token that starts with `</` and
//! the next token completes `think>`).  The parser maintains a small lookahead
//! buffer to detect these cases reliably.

const LOOKAHEAD: usize = 32;

/// The parser state machine state.
#[derive(Debug, Clone, PartialEq)]
enum State {
    /// Emitting plain text â€” between `</think>` and the next `<think>` or end.
    Text,
    /// Inside `<think>â€¦</think>` â€” emitting thinking tokens.
    Thinking,
    /// Inside `<tool_call>â€¦</tool_call>` â€” buffering JSON.
    ToolCall { buffer: String },
}

/// Events emitted by the [`StreamParser`].
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedEvent {
    /// A plain text token to emit as a `text_delta` content block event.
    TextToken(String),
    /// A thinking token to emit as a `thinking_delta` content block event.
    ThinkingToken(String),
    /// `</think>` was detected â€” close the thinking block, open the text block.
    ThinkingEnd,
    /// A complete `<tool_call>` JSON object is ready.
    ToolCallReady {
        name: String,
        arguments: serde_json::Value,
    },
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
        Self {
            state: State::Text,
            lookahead: String::new(),
        }
    }

    /// True if the parser is currently inside a thinking block.
    #[allow(dead_code)]
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
                } else if !buffer.is_empty() {
                    tracing::warn!(
                        json = %buffer,
                        "failed to parse tool_call JSON at flush — emitting raw text instead of dropping it"
                    );
                    events.push(ParsedEvent::TextToken(std::mem::take(buffer)));
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

    // â”€â”€ Internal processing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
                        self.lookahead = self.lookahead[pos + "<tool_call>".len()..].to_string();
                        self.state = State::ToolCall {
                            buffer: String::new(),
                        };
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
                        self.lookahead = self.lookahead[pos + "</think>".len()..].to_string();
                        self.state = State::Text;
                        events.push(ParsedEvent::ThinkingEnd);
                        // Continue processing remainder in Text state.
                    } else {
                        // No closing tag yet â€” keep LOOKAHEAD buffered, emit the rest.
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
                        let remainder = self.lookahead[pos + "</tool_call>".len()..].to_string();
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
                            // Never silently drop the model's attempted tool
                            // call — an unparseable one still represents real
                            // intent (e.g. a malformed-but-recognizable bash
                            // call), and dropping it produces a response that
                            // looks like a clean, empty "stop" to the client
                            // with no signal that anything went wrong.
                            tracing::warn!(
                                json = %buffer,
                                "failed to parse tool_call JSON — emitting raw text instead of dropping it"
                            );
                            if !buffer.is_empty() {
                                events.push(ParsedEvent::TextToken(buffer));
                            }
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

// â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Find `"key"` in `buf` and return the quoted value that follows it,
/// tolerating a malformed or missing separator between key and value (`:`,
/// `=`, whitespace, or the key's closing quote being dropped entirely) —
/// only the JSON *value* needs to be well-formed, not the syntax around it.
fn extract_value_after_key(buf: &str, key: &str) -> Option<String> {
    let key_pat = format!("\"{key}");
    let n_start = buf.find(&key_pat)?;
    let rest = &buf[n_start + key_pat.len()..];
    // Skip the key's closing quote, if the model included one.
    let rest = rest.strip_prefix('"').unwrap_or(rest);
    // Skip separator chars between key and value.
    let rest = rest.trim_start_matches(|c: char| c == ':' || c == '=' || c.is_whitespace());
    // The value itself must still be a well-formed quoted string.
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Try to parse a tool call JSON buffer into a [`ParsedEvent::ToolCallReady`].
fn try_parse_tool_call(buf: &str) -> Option<ParsedEvent> {
    let trimmed = buf.trim();

    // First try standard JSON
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(name) = v
            .get("name")
            .or_else(|| v.get("function"))
            .and_then(|n| n.as_str())
        {
            let mut arguments = v
                .get("arguments")
                .or_else(|| v.get("parameters"))
                .or_else(|| v.get("input"))
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            if let Some(s) = arguments.as_str() {
                if let Ok(p) = serde_json::from_str(s) {
                    arguments = p;
                }
            }
            tracing::info!(tool_call_name = %name, args = %arguments, "Successfully parsed model tool call output");
            return Some(ParsedEvent::ToolCallReady {
                name: name.to_string(),
                arguments,
            });
        }
    }

    // Fallback: Model hallucinated XML tags (e.g. <function_name>Write</function_name> or <Write>) OR produced invalid JSON structure
    let mut ext_name = None;

    // Try to extract name from `"name": "bash"` or `"function": "bash"` directly.
    // Tolerates the key/value separator being malformed (e.g. `"name="bash"`
    // with an `=` instead of `:` — an occasional model generation glitch)
    // since we only care about locating the quoted value, not validating the
    // JSON structure around it.
    for key in ["name", "function"] {
        if let Some(name) = extract_value_after_key(buf, key) {
            ext_name = Some(name);
            break;
        }
    }

    if ext_name.is_none() {
        if let Some(start) = buf.find("<function_name>") {
            if let Some(end) = buf[start..].find("</function_name>") {
                ext_name = Some(buf[start + 15..start + end].trim().to_string());
            }
        } else if let Some(start) = buf.find('<') {
            if let Some(end) = buf[start..].find('>') {
                let tag = buf[start + 1..start + end].trim();
                if !tag.contains('/') && !tag.contains(' ') {
                    ext_name = Some(tag.to_string());
                }
            }
        }
    }

    let name = ext_name?;

    // Fallback extract JSON arguments object embedded anywhere inside the invalid object
    let mut arguments = serde_json::json!({});
    let mut search_idx = 0;
    while let Some(start) = buf[search_idx..].find('{') {
        let abs_start = search_idx + start;
        if let Some(end) = buf.rfind('}') {
            if end >= abs_start {
                let json_slice = &buf[abs_start..=end];
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_slice) {
                    if let Some(obj) = parsed.as_object() {
                        if obj.contains_key("arguments") {
                            arguments = obj["arguments"].clone();
                            break;
                        } else if obj.contains_key("name")
                            && obj.len() <= 2
                            && !obj.contains_key("command")
                        {
                            // Probably parsed the outer shell successfully but it didn't contain anything useful
                            // Just continue searching inner brackets!
                        } else {
                            // Direct arguments embedded (e.g., {"command": "..."})
                            arguments = parsed;
                            break;
                        }
                    }
                }
            }
        }
        search_idx = abs_start + 1;
    }

    tracing::info!(tool_call_name = %name, args = %arguments, "Successfully extracted fallback model tool call output");
    Some(ParsedEvent::ToolCallReady { name, arguments })
}

/// Find the largest byte index â‰¤ `index` that falls on a UTF-8 char boundary.
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

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            .filter_map(|e| {
                if let ParsedEvent::TextToken(t) = e {
                    Some(t.as_str())
                } else {
                    None
                }
            })
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

        assert!(evts
            .iter()
            .any(|e| matches!(e, ParsedEvent::ThinkingToken(_))));
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

        let tool_evt = evts
            .iter()
            .find(|e| matches!(e, ParsedEvent::ToolCallReady { .. }));
        assert!(tool_evt.is_some());
        if let Some(ParsedEvent::ToolCallReady { name, arguments }) = tool_evt {
            assert_eq!(name, "Read");
            assert_eq!(arguments["file_path"], "/foo.rs");
        }
    }

    #[test]
    fn tool_call_with_function_key_parsed_correctly() {
        // Qwen3 sometimes emits `"function"` instead of the OpenAI-standard `"name"` key.
        let mut p = StreamParser::new();
        let evts = feed(
            &mut p,
            &[
                "<tool_call>\n",
                r#"{"function": "bash", "arguments": {"command": "ls"}}"#,
                "\n</tool_call>",
            ],
        );

        let tool_evt = evts
            .iter()
            .find(|e| matches!(e, ParsedEvent::ToolCallReady { .. }));
        assert!(tool_evt.is_some());
        if let Some(ParsedEvent::ToolCallReady { name, arguments }) = tool_evt {
            assert_eq!(name, "bash");
            assert_eq!(arguments["command"], "ls");
        }
    }

    #[test]
    fn no_thinking_block_emits_only_text() {
        let mut p = StreamParser::new();
        let evts = feed(&mut p, &["Just a plain response."]);
        assert!(evts.iter().all(|e| matches!(e, ParsedEvent::TextToken(_))));
        assert!(!evts.iter().any(|e| *e == ParsedEvent::ThinkingEnd));
    }

    #[test]
    fn tool_call_with_equals_separator_still_parses_name() {
        // Observed in production: model emitted `"name="bash"` (missing colon)
        // instead of `"name": "bash"`. The strict JSON parse fails, but the
        // fallback should still recover the tool name rather than dropping
        // the whole call.
        let mut p = StreamParser::new();
        let evts = feed(
            &mut p,
            &[
                "<tool_call>\n",
                r#"{"name="bash", "arguments": {"command":"true"}}"#,
                "\n</tool_call>",
            ],
        );

        let tool_evt = evts
            .iter()
            .find(|e| matches!(e, ParsedEvent::ToolCallReady { .. }));
        assert!(tool_evt.is_some(), "expected a recovered tool call, got: {evts:?}");
        if let Some(ParsedEvent::ToolCallReady { name, .. }) = tool_evt {
            assert_eq!(name, "bash");
        }
    }

    #[test]
    fn unparseable_tool_call_emits_raw_text_instead_of_dropping() {
        // No recognizable "name"/"function" key at all — try_parse_tool_call
        // returns None. The buffered text must still surface to the caller
        // instead of vanishing, so a response never silently collapses to
        // empty content with finish_reason=stop.
        let mut p = StreamParser::new();
        let evts = feed(
            &mut p,
            &["<tool_call>\n", "not json at all", "\n</tool_call>", "after"],
        );

        assert!(
            !evts.iter().any(|e| matches!(e, ParsedEvent::ToolCallReady { .. })),
            "garbage input should not fabricate a tool call"
        );
        let combined: String = evts
            .iter()
            .filter_map(|e| {
                if let ParsedEvent::TextToken(t) = e {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(combined.contains("not json at all"));
        assert!(combined.contains("after"));
    }
}
