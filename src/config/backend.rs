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
    /// Global default `--spec-type` value (e.g. `"draft-mtp"`, `"draft-dspark"`).
    ///
    /// Overridden per-model via `[models.load] spec_type`.
    #[serde(default)]
    pub spec_type: Option<String>,
    /// Global default `--spec-draft-n-max` value.
    ///
    /// Overridden per-model via `[models.load] spec_draft_n_max`.
    #[serde(default)]
    pub spec_draft_n_max: Option<usize>,
    /// Global default `--spec-draft-model` path, for architectures using an
    /// external drafter GGUF (e.g. DeepSeek's DSpark drafter).
    ///
    /// Overridden per-model via `[models.load] draft_model`.
    #[serde(default)]
    pub draft_model: Option<PathBuf>,
    /// Global default `--chat-template-kwargs` value, serialized to JSON.
    ///
    /// Overridden per-model via `[models.load] chat_template_kwargs`.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// Global default sampling temperature, used by request handlers as a
    /// fallback when neither the client request nor the model's `load`
    /// section specify one.
    ///
    /// Overridden per-model via `[models.load] temperature`.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Global default nucleus sampling (`top_p`).
    ///
    /// Overridden per-model via `[models.load] top_p`.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Global default `min_p`.
    ///
    /// Overridden per-model via `[models.load] min_p`.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Additional CLI flags appended verbatim to the subprocess command line.
    ///
    /// Example: `["--flash-attn", "on", "--parallel", "2"]`. Prefer the typed
    /// `spec_type` / `spec_draft_n_max` / `draft_model` fields above for
    /// speculative decoding — they support per-model overrides, this doesn't.
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
            spec_type: None,
            spec_draft_n_max: None,
            draft_model: None,
            chat_template_kwargs: None,
            temperature: None,
            top_p: None,
            min_p: None,
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
