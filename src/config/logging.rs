use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Logging configuration — file output, rotation, and retention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Directory where rolling log files are written.
    ///
    /// Relative paths are resolved from the working directory at startup.
    /// The directory is created automatically if it does not exist.
    #[serde(default = "default_log_dir")]
    pub log_dir: PathBuf,

    /// Number of days to retain rolling log files.
    ///
    /// Files older than this are deleted at startup. Set to `0` to keep
    /// all files indefinitely.
    #[serde(default = "default_log_retention_days")]
    pub log_retention_days: u32,

    /// Minimum log level written to the file (independent of stdout).
    ///
    /// Accepts the same syntax as `RUST_LOG`: `"trace"`, `"debug"`,
    /// `"info"`, `"warn"`, `"error"`, or a directive like
    /// `"turboquant_loader=debug,info"`.
    #[serde(default = "default_file_log_level")]
    pub file_log_level: String,

    /// Minimum log level written to stdout.
    ///
    /// Overridden by `RUST_LOG` if set. Defaults to `"info"`.
    #[serde(default = "default_stdout_log_level")]
    pub stdout_log_level: String,

    /// When `true`, each completed inference request is appended as a JSON line
    /// to `<log_dir>/conversations.<YYYY-MM-DD>.jsonl`, capturing the full
    /// prompt messages and assembled response text.
    ///
    /// Disabled by default — enable for debugging agent interaction patterns.
    #[serde(default)]
    pub log_conversations: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_dir: default_log_dir(),
            log_retention_days: default_log_retention_days(),
            file_log_level: default_file_log_level(),
            stdout_log_level: default_stdout_log_level(),
            log_conversations: false,
        }
    }
}

fn default_log_dir() -> PathBuf {
    PathBuf::from("logs")
}

fn default_log_retention_days() -> u32 {
    7
}

fn default_file_log_level() -> String {
    "info".to_string()
}

fn default_stdout_log_level() -> String {
    "info".to_string()
}
