use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Which llama-server binary variant to run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackendVariant {
    /// Standard llama.cpp llama-server (Vulkan build — primary, supports Arc B70).
    #[default]
    LlamaServer,
    /// TheTom/llama-cpp-turboquant fork — CUDA only, adds `--cache-type-k turbo3`.
    TurboQuant,
}

/// Configuration for the managed llama-server subprocess backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Which binary variant to launch.
    #[serde(default)]
    pub variant: BackendVariant,
    /// Absolute path to `llama-server.exe` (or equivalent on Linux/macOS).
    pub binary_path: PathBuf,
    /// Port the subprocess listens on internally (proxied from the external port).
    #[serde(default = "default_internal_port")]
    pub internal_port: u16,
    /// Seconds to wait for `/health` to return 200 before giving up.
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,
    /// Automatically respawn the subprocess if it exits unexpectedly.
    #[serde(default = "default_true")]
    pub restart_on_crash: bool,
    /// Additional CLI flags appended verbatim to the subprocess command line.
    ///
    /// Example: `["--flash-attn", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]`
    #[serde(default)]
    pub extra_flags: Vec<String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            variant: BackendVariant::LlamaServer,
            binary_path: PathBuf::from("llama-server"),
            internal_port: default_internal_port(),
            startup_timeout_secs: default_startup_timeout(),
            restart_on_crash: true,
            extra_flags: Vec::new(),
        }
    }
}

fn default_internal_port() -> u16 {
    7433
}

fn default_startup_timeout() -> u64 {
    180
}

fn default_true() -> bool {
    true
}
