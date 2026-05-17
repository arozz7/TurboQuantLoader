//! HTTP server — OpenAI and Anthropic compatible API on a single axum router.
//!
//! Both endpoint families are always active on the same port:
//!
//! | Method | Path                     | Protocol  |
//! |--------|--------------------------|-----------|
//! | GET    | `/v1/models`             | both      |
//! | POST   | `/v1/chat/completions`   | OpenAI    |
//! | POST   | `/v1/messages`           | Anthropic |
//! | GET    | `/health`                | both      |
//! | GET    | `/metrics`               | Prometheus|
//! | GET    | `/v1/admin/stats`        | both      |
//! | GET    | `/v1/admin/status`       | both      |
//! | POST   | `/v1/admin/restart`      | both      |
//! | POST   | `/v1/admin/load`         | both      |

mod error;
pub mod llama_process;
pub mod proxy;
mod routes;
mod sse;
pub mod stream_parser;
pub mod types;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::RwLock;
use tracing::info;

use crate::config::AppConfig;
use crate::metrics::MetricsCollector;
use crate::model::registry::ModelRegistry;
use llama_process::LlamaProcess;

/// Shared state injected into every request handler via axum `State`.
///
/// `process` and `config` are behind `RwLock` so the admin API can hot-swap
/// the backend and update the model path without restarting the server.
#[derive(Clone)]
pub struct AppState {
    /// The running llama-server subprocess. Swapped on `POST /v1/admin/load`.
    pub process: Arc<RwLock<Arc<LlamaProcess>>>,
    /// Active configuration. Written by admin/load to change model path.
    pub config: Arc<RwLock<Arc<AppConfig>>>,
    /// Rolling request metrics + GPU telemetry.
    pub metrics: Arc<MetricsCollector>,
    /// Set to `true` while a model switch is in progress.
    ///
    /// Handlers check this flag and return 503 when switching; clients should
    /// retry with a short back-off (the switch typically takes 30–180 s).
    pub switching: Arc<AtomicBool>,
    /// Timestamp of the most recent completed inference request.
    ///
    /// Used by the idle guard in [`AppState::trigger_model_switch`]: a switch
    /// is blocked while the model has been active within `model_idle_timeout_secs`.
    pub last_request_at: Arc<Mutex<Instant>>,
}

impl AppState {
    /// Clone the current `Arc<LlamaProcess>` without holding the lock longer
    /// than necessary. Use for short-lived read access in request handlers.
    pub async fn process_snapshot(&self) -> Arc<LlamaProcess> {
        self.process.read().await.clone()
    }

    /// Clone the current `Arc<AppConfig>`.
    pub async fn config_snapshot(&self) -> Arc<AppConfig> {
        self.config.read().await.clone()
    }

    /// Record that an inference request just completed.
    ///
    /// Call this once per successful chat request (before returning the response)
    /// so the idle guard knows the model is actively in use.
    pub fn touch_last_request(&self) {
        if let Ok(mut t) = self.last_request_at.lock() {
            *t = Instant::now();
        }
    }

    /// Short name of the currently-loaded model (file stem of `model_path`).
    pub async fn current_model_name(&self) -> String {
        let cfg = self.config_snapshot().await;
        cfg.model
            .model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("local-model")
            .to_string()
    }

    /// Attempt to switch to the named model.
    ///
    /// Returns `true` when a switch was triggered (caller should return 503).
    /// Returns `false` when the name cannot be resolved (caller proceeds normally).
    ///
    /// The actual model load runs on a background Tokio task; `switching` is
    /// reset to `false` once the new process is ready (or on failure).
    pub async fn trigger_model_switch(&self, name: &str) -> bool {
        let config = self.config_snapshot().await;

        let Some(def) = ModelRegistry::resolve(name, &config) else {
            tracing::warn!(model = %name, "model not found in registry or models_dir — ignoring switch request");
            return false;
        };

        // Idle guard: refuse the switch while the model has been recently used.
        let timeout_secs = config.model.model_idle_timeout_secs;
        if timeout_secs > 0 {
            let idle = self
                .last_request_at
                .lock()
                .map(|t| t.elapsed())
                .unwrap_or_default();
            if idle.as_secs() < timeout_secs {
                tracing::info!(
                    model = %name,
                    idle_secs = idle.as_secs(),
                    required_idle_secs = timeout_secs,
                    "model switch blocked — model active within idle timeout"
                );
                return false;
            }
        }

        // CAS: set switching false→true. If already true, a switch is in flight.
        if self
            .switching
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            tracing::info!(model = %name, "model switch already in progress");
            return true;
        }

        // Build updated config from the definition, falling back to base values.
        let mut new_config = (*config).clone();
        new_config.model.model_path = def.path;
        if let Some(ctx) = def.context_size {
            new_config.model.context_size = ctx;
        }
        if let Some(layers) = def.n_gpu_layers {
            new_config.model.n_gpu_layers = layers;
        }
        if let Some(gpu) = def.main_gpu {
            new_config.model.main_gpu = gpu;
        }
        if let Some(batch) = def.batch_size {
            new_config.model.batch_size = batch;
        }
        if let Some(split) = def.tensor_split {
            new_config.model.tensor_split = split;
        }
        let new_config = Arc::new(new_config);

        let state = self.clone();
        let new_config_clone = Arc::clone(&new_config);

        tokio::spawn(async move {
            let model_display = new_config_clone.model.model_path.display().to_string();
            tracing::info!(model = %model_display, "model switch started");

            {
                let old_proc = state.process_snapshot().await;
                old_proc.kill().await;
            }

            // Update config now so status endpoint reflects the incoming model.
            *state.config.write().await = Arc::clone(&new_config_clone);

            match LlamaProcess::start(new_config_clone).await {
                Ok(new_proc) => {
                    *state.process.write().await = Arc::new(new_proc);
                    tracing::info!(model = %model_display, "model switch complete");
                }
                Err(e) => {
                    tracing::error!(error = %e, model = %model_display, "model switch failed");
                }
            }

            state.switching.store(false, Ordering::SeqCst);
        });

        true
    }
}

/// Build the axum router with all API routes mounted.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // ── Inference ─────────────────────────────────────────────────────────
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/chat/completions", post(routes::openai::chat_completions))
        .route("/v1/messages", post(routes::anthropic::create_message))
        // ── Observability ─────────────────────────────────────────────────────
        .route("/health", get(routes::metrics::health))
        .route("/metrics", get(routes::metrics::prometheus_metrics))
        .route("/v1/admin/stats", get(routes::metrics::admin_stats))
        // ── Admin ─────────────────────────────────────────────────────────────
        .route("/v1/admin/status", get(routes::admin::status))
        .route("/v1/admin/restart", post(routes::admin::restart))
        .route("/v1/admin/load", post(routes::admin::load))
        .with_state(state)
}

/// Start the llama-server subprocess, build the router, and serve until interrupted.
pub async fn serve(config: AppConfig) -> Result<()> {
    let host = config.server.host.clone();
    let port = config.server.port;

    info!(host = %host, port, "starting TurboQuantLoader");

    let config = Arc::new(config);
    let process = LlamaProcess::start(Arc::clone(&config))
        .await
        .context("failed to start llama-server subprocess")?;

    info!(internal_port = config.backend.internal_port, "llama-server ready; starting API server");

    let metrics = MetricsCollector::start();

    let state = AppState {
        process: Arc::new(RwLock::new(Arc::new(process))),
        config: Arc::new(RwLock::new(config)),
        metrics,
        switching: Arc::new(AtomicBool::new(false)),
        last_request_at: Arc::new(Mutex::new(Instant::now())),
    };

    let router = build_router(state);

    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("failed to bind to {bind_addr}"))?;

    info!(addr = %listener.local_addr().unwrap(), "API server listening");

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            info!("shutdown signal received");
        })
        .await
        .context("server error")?;

    Ok(())
}
