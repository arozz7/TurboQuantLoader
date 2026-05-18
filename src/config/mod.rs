mod backend;
mod kv_cache;
mod logging;
mod model;
mod server;

pub use backend::{BackendConfig, BackendVariant};
pub use kv_cache::{KvBits, KvCacheConfig, KvStrategy};
pub use logging::LoggingConfig;
pub use model::{ModelConfig, ModelDefinition};
pub use server::ServerConfig;

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level application configuration.
///
/// Loaded from `config.toml` via [`load_from_file`], then refined by
/// [`AppConfig::apply_cli_overrides`] before use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// HTTP server settings.
    #[serde(default)]
    pub server: ServerConfig,
    /// Model loading and inference settings.
    #[serde(default)]
    pub model: ModelConfig,
    /// KV cache compression settings.
    #[serde(default)]
    pub kv_cache: KvCacheConfig,
    /// llama-server subprocess backend settings.
    #[serde(default)]
    pub backend: BackendConfig,
    /// Named model registry (`[[models]]` in TOML).
    ///
    /// Agents select a model by sending its `name` as the OpenAI `model` field.
    /// Absent → empty list (server uses `[model] model_path` directly).
    #[serde(default)]
    pub models: Vec<ModelDefinition>,
    /// Log file output and retention settings.
    #[serde(default)]
    pub logging: LoggingConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            model: ModelConfig::default(),
            kv_cache: KvCacheConfig::default(),
            backend: BackendConfig::default(),
            models: Vec::new(),
            logging: LoggingConfig::default(),
        }
    }
}

impl AppConfig {
    /// Override config values with any fields that were explicitly set on the CLI.
    pub fn apply_cli_overrides(&mut self, overrides: &CliOverrides) {
        if let Some(ref p) = overrides.model_path {
            self.model.model_path = p.clone();
        }
        if let Some(port) = overrides.port {
            self.server.port = port;
        }
        if let Some(ctx) = overrides.context_size {
            self.model.context_size = ctx;
        }
        if let Some(layers) = overrides.n_gpu_layers {
            self.model.n_gpu_layers = layers;
        }
        if let Some(bits) = overrides.kv_bits {
            self.kv_cache.bits = bits;
        }
    }
}

/// CLI-supplied values that override the corresponding `config.toml` settings.
///
/// All fields are `Option` — only `Some` fields overwrite the loaded config.
#[derive(Debug, Default)]
pub struct CliOverrides {
    /// Override `[model] model_path`.
    pub model_path: Option<std::path::PathBuf>,
    /// Override `[server] port`.
    pub port: Option<u16>,
    /// Override `[model] context_size`.
    pub context_size: Option<u32>,
    /// Override `[model] n_gpu_layers`.
    pub n_gpu_layers: Option<i32>,
    /// Override `[kv_cache] bits`.
    pub kv_bits: Option<KvBits>,
}

/// Load [`AppConfig`] from a TOML file at `path`.
///
/// Missing TOML sections fall back to their [`Default`] implementations.
/// Returns an error if the file cannot be read or contains invalid TOML.
pub fn load_from_file(path: &Path) -> Result<AppConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("invalid TOML in config file: {}", path.display()))
}
