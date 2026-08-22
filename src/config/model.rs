use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A named model entry in the `[[models]]` registry.
///
/// Fields other than `name` and `path` are optional per-model overrides; absent
/// values fall back to the corresponding `[model]` section defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    /// Short identifier used as the OpenAI `model` field (e.g. `"qwen3-27b"`).
    pub name: String,
    /// Absolute path to the GGUF model file.
    pub path: PathBuf,
    /// Override context window size for this model.
    pub context_size: Option<u32>,
    /// Override GPU layer count (`-1` = all layers).
    pub n_gpu_layers: Option<i32>,
    /// Override primary GPU device index.
    pub main_gpu: Option<i32>,
    /// Override prompt-evaluation batch size.
    pub batch_size: Option<u32>,
    /// Override tensor split weights (one float per device).
    pub tensor_split: Option<Vec<f32>>,
    /// Load-time and sampling overrides for this model (speculative decoding,
    /// chat-template kwargs, default sampling params). Absent fields fall back
    /// to the corresponding `[backend]` section defaults.
    #[serde(default)]
    pub load: Option<LoadConfig>,
}

/// Per-model load-time and sampling overrides.
///
/// Speculative decoding is fundamentally per-model — a model with a built-in
/// draft head (`draft-mtp`) needs no extra file, while a model needing an
/// external drafter (`draft-dspark`) needs `draft_model` set too. All fields
/// are optional; `None` falls back to the corresponding `[backend]` global.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadConfig {
    /// `--spec-type` value (e.g. `"draft-mtp"`, `"draft-dspark"`).
    pub spec_type: Option<String>,
    /// `--spec-draft-n-max` value.
    pub spec_draft_n_max: Option<usize>,
    /// `--spec-draft-model` path, for architectures using an external drafter.
    pub draft_model: Option<PathBuf>,
    /// `--chat-template-kwargs` value, serialized to JSON.
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// Default sampling temperature for this model.
    pub temperature: Option<f32>,
    /// Default nucleus sampling (`top_p`) for this model.
    pub top_p: Option<f32>,
    /// Default `min_p` for this model.
    pub min_p: Option<f32>,
    /// Escape hatch: verbatim CLI flags appended after the typed flags above,
    /// on top of the global `[backend] extra_flags`.
    #[serde(default)]
    pub extra_flags: Vec<String>,
}

/// Model loading and inference configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the primary GGUF model file.
    pub model_path: PathBuf,
    /// Optional vision projector GGUF (Phase 5).
    pub mmproj_path: Option<PathBuf>,
    /// Root directory scanned for available models (used by `list` and `bench`).
    pub models_dir: PathBuf,
    /// Number of transformer layers to offload to GPU. `-1` offloads all layers.
    pub n_gpu_layers: i32,
    /// Index of the primary GPU device for llama-server `--main-gpu`.
    /// `-1` lets llama-server pick automatically.
    #[serde(default = "default_main_gpu")]
    pub main_gpu: i32,
    /// Per-device VRAM weight used to split tensors across GPUs via `--tensor-split`.
    /// Example: `[16.0, 32.0]` for RTX 4070 Ti Super (16 GB) + Arc B70 (32 GB).
    #[serde(default)]
    pub tensor_split: Vec<f32>,
    /// KV cache context window size in tokens (default: `262144`).
    pub context_size: u32,
    /// Prompt evaluation batch size (default: `512`).
    pub batch_size: u32,
    /// CPU thread count for prompt evaluation (default: half of logical CPUs).
    pub threads: u32,
    /// Name of the model from the `[[models]]` registry to load at startup.
    ///
    /// If `None` the server uses `model_path` directly.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Seconds of inactivity required before a client-requested model switch is
    /// allowed. Prevents mid-session reloads when an agent sends a different
    /// `model` string across requests. Default: `1800` (30 minutes).
    ///
    /// Set to `0` to disable the guard (always switch immediately).
    /// Admin `POST /v1/admin/load` bypasses this timeout unconditionally.
    #[serde(default = "default_idle_timeout")]
    pub model_idle_timeout_secs: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            mmproj_path: None,
            models_dir: PathBuf::from("models"),
            n_gpu_layers: -1,
            main_gpu: default_main_gpu(),
            tensor_split: vec![],
            context_size: 262144,
            batch_size: 512,
            threads: default_thread_count(),
            default_model: None,
            model_idle_timeout_secs: default_idle_timeout(),
        }
    }
}

fn default_main_gpu() -> i32 {
    -1
}

fn default_idle_timeout() -> u64 {
    1800
}

/// Returns half the logical CPU count, falling back to `1`.
fn default_thread_count() -> u32 {
    std::thread::available_parallelism()
        .map(|n| ((n.get() as u32) / 2).max(1))
        .unwrap_or(1)
}
