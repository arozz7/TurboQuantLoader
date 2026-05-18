//! Conversation logging — appends one JSON line per completed exchange to a
//! daily-rotating `.jsonl` file in the configured log directory.
//!
//! Disabled by default (`log_conversations = false`). Enable in `config.toml`
//! to capture full prompt and response content for debugging agent behaviour.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Result;
use serde::Serialize;

// ── Public types ──────────────────────────────────────────────────────────────

/// A single message in a conversation (role + content).
#[derive(Debug, Clone, Serialize)]
pub struct LogMessage {
    pub role: String,
    pub content: String,
}

/// One complete request/response exchange written as a JSON line.
#[derive(Debug, Serialize)]
pub struct ConversationEntry {
    /// ISO-8601 timestamp of when the request completed.
    pub ts: String,
    /// Unique request ID (same as the `id` field in the OpenAI response).
    pub id: String,
    /// Short model name (file stem of the loaded GGUF).
    pub model: String,
    /// `"openai"` or `"anthropic"` — which protocol the client used.
    pub protocol: &'static str,
    /// Whether the client requested streaming.
    pub stream: bool,
    /// Messages sent to the model (after tool-injection pre-processing).
    pub messages: Vec<LogMessage>,
    /// Full assembled response text.
    pub response: String,
    /// Prompt token count (from llama-server usage stats).
    pub prompt_tokens: u32,
    /// Completion token count.
    pub completion_tokens: u32,
    /// Token generation throughput.
    pub tps: f32,
    /// Why generation stopped: `"stop"`, `"length"`, `"tool_calls"`.
    pub finish_reason: String,
}

// ── ConversationLogger ────────────────────────────────────────────────────────

struct LoggerInner {
    /// Date string the current file was opened for (e.g. `"2025-05-17"`).
    current_date: String,
    writer: BufWriter<File>,
}

/// Appends [`ConversationEntry`] records as JSON lines to a daily-rotating
/// file named `conversations.<YYYY-MM-DD>.jsonl` inside `log_dir`.
///
/// All public methods are no-ops when the logger is disabled.
pub struct ConversationLogger {
    log_dir: PathBuf,
    /// `None` when logging is disabled — all methods return immediately.
    inner: Option<Mutex<LoggerInner>>,
}

impl ConversationLogger {
    /// Create a logger. If `enabled` is `false` the struct is inert and no
    /// file is opened or created.
    pub fn new(log_dir: PathBuf, enabled: bool) -> Result<Self> {
        if !enabled {
            return Ok(Self {
                log_dir,
                inner: None,
            });
        }

        std::fs::create_dir_all(&log_dir)?;
        let today = today_str();
        let writer = open_writer(&log_dir, &today)?;

        Ok(Self {
            log_dir,
            inner: Some(Mutex::new(LoggerInner {
                current_date: today,
                writer,
            })),
        })
    }

    /// Returns `true` when conversation logging is active.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Append `entry` as a JSON line. Rotates to a new file if the calendar
    /// date has changed since the last write. Errors are logged as warnings
    /// rather than propagated — a logging failure must never crash inference.
    pub fn log(&self, entry: &ConversationEntry) {
        let Some(ref mu) = self.inner else { return };

        let mut inner = match mu.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!("conversation log mutex poisoned: {e}");
                return;
            }
        };

        // Rotate file on date change.
        let today = today_str();
        if today != inner.current_date {
            match open_writer(&self.log_dir, &today) {
                Ok(w) => {
                    inner.writer = w;
                    inner.current_date = today;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to rotate conversation log");
                    return;
                }
            }
        }

        match serde_json::to_string(entry) {
            Ok(line) => {
                if let Err(e) = writeln!(inner.writer, "{line}") {
                    tracing::warn!(error = %e, "failed to write conversation log entry");
                    return;
                }
                if let Err(e) = inner.writer.flush() {
                    tracing::warn!(error = %e, "failed to flush conversation log");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialise conversation entry");
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn today_str() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Seconds since epoch → YYYY-MM-DD (UTC, no external dep)
    let days = now / 86_400;
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn open_writer(log_dir: &PathBuf, date: &str) -> Result<BufWriter<File>> {
    let path = log_dir.join(format!("conversations.{date}.jsonl"));
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    Ok(BufWriter::new(file))
}

/// Convert days-since-Unix-epoch to (year, month, day) in the proleptic
/// Gregorian calendar (UTC). Used to avoid a chrono dependency.
fn days_to_ymd(mut days: u64) -> (u32, u32, u32) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u32, m as u32, d as u32)
}

/// Current UTC time as an ISO-8601 string (seconds precision).
pub fn now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86_400;
    let time = secs % 86_400;
    let (y, mo, d) = days_to_ymd(days);
    let h = time / 3600;
    let mi = (time % 3600) / 60;
    let s = time % 60;
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}
