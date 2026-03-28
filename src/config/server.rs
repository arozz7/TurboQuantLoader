use serde::{Deserialize, Serialize};

/// HTTP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind address (default: `"127.0.0.1"`).
    pub host: String,
    /// Listen port (default: `7432`).
    pub port: u16,
    /// Maximum number of requests processed concurrently (default: `4`).
    pub max_concurrent_requests: usize,
    /// Per-request timeout in seconds (default: `300`).
    pub request_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 7432,
            max_concurrent_requests: 4,
            request_timeout_secs: 300,
        }
    }
}
