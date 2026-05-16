use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
    pub tensor_split: Vec<f32>,
    /// KV cache context window size in tokens (default: `262144`).
    pub context_size: u32,
    /// Prompt evaluation batch size (default: `512`).
    pub batch_size: u32,
    /// CPU thread count for prompt evaluation (default: half of logical CPUs).
    pub threads: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            mmproj_path: None,
            models_dir: PathBuf::from("models"),
            n_gpu_layers: -1,
            main_gpu: default_main_gpu(),
            tensor_split: vec![16.0, 32.0],
            context_size: 262144,
            batch_size: 512,
            threads: default_thread_count(),
        }
    }
}

fn default_main_gpu() -> i32 {
    -1
}

/// Returns half the logical CPU count, falling back to `1`.
fn default_thread_count() -> u32 {
    std::thread::available_parallelism()
        .map(|n| ((n.get() as u32) / 2).max(1))
        .unwrap_or(1)
}
