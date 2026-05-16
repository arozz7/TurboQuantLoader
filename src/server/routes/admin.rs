//! Agent control API — hot-swap model and query backend lifecycle.
//!
//! | Method | Path                  | Description                       |
//! |--------|-----------------------|-----------------------------------|
//! | GET    | `/v1/admin/status`    | Backend pid, state, model, uptime |
//! | POST   | `/v1/admin/restart`   | Graceful restart with same config |
//! | POST   | `/v1/admin/load`      | Restart with a new model path     |

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde_json::json;

use crate::server::llama_process::LlamaProcess;
use crate::server::AppState;

// ── GET /v1/admin/status ──────────────────────────────────────────────────────

/// `GET /v1/admin/status` — backend process snapshot.
pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    let proc = state.process_snapshot().await;
    let cfg = state.config_snapshot().await;

    let backend_state = proc.state().await;
    let pid = proc.pid().await;
    let model_name = cfg
        .model
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    Json(json!({
        "state": format!("{backend_state:?}"),
        "pid": pid,
        "model": model_name,
        "variant": format!("{:?}", cfg.backend.variant),
        "uptime_secs": state.metrics.uptime_secs(),
        "total_requests": state.metrics.total_requests.load(std::sync::atomic::Ordering::Relaxed),
        "total_errors": state.metrics.total_errors.load(std::sync::atomic::Ordering::Relaxed),
    }))
}

// ── POST /v1/admin/restart ────────────────────────────────────────────────────

/// `POST /v1/admin/restart` — gracefully restart llama-server with the current config.
pub async fn restart(State(state): State<AppState>) -> impl IntoResponse {
    tracing::info!("admin/restart: restarting llama-server");

    let config = state.config_snapshot().await;

    match LlamaProcess::start(Arc::clone(&config)).await {
        Ok(new_proc) => {
            *state.process.write().await = Arc::new(new_proc);
            tracing::info!("admin/restart: llama-server restarted successfully");
            (StatusCode::OK, Json(json!({"status": "restarted"}))).into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "admin/restart: failed to restart llama-server");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

// ── POST /v1/admin/load ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct LoadRequest {
    /// Absolute path to the new GGUF model file.
    pub model_path: String,
    /// Optional: override context size for the new model.
    pub context_size: Option<u32>,
    /// Optional: override n_gpu_layers.
    pub n_gpu_layers: Option<i32>,
}

/// `POST /v1/admin/load` — hot-swap the model without restarting the whole server.
///
/// Kills the current llama-server process, updates the config, and spawns a new
/// subprocess pointing at the new model. Returns 202 immediately; the backend
/// becomes available again once `/health` returns `ok`.
pub async fn load(
    State(state): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> impl IntoResponse {
    tracing::info!(model_path = %req.model_path, "admin/load: loading new model");

    // Build updated config.
    let base_config = state.config_snapshot().await;
    let mut new_config = (*base_config).clone();
    new_config.model.model_path = req.model_path.clone().into();
    if let Some(ctx) = req.context_size {
        new_config.model.context_size = ctx;
    }
    if let Some(layers) = req.n_gpu_layers {
        new_config.model.n_gpu_layers = layers;
    }
    let new_config = Arc::new(new_config);

    // Kill the old process before spawning the new one.
    {
        let old_proc = state.process_snapshot().await;
        old_proc.kill().await;
    }

    // Update stored config first so subsequent requests see the new model.
    *state.config.write().await = Arc::clone(&new_config);

    // Spawn new process.
    match LlamaProcess::start(new_config).await {
        Ok(new_proc) => {
            *state.process.write().await = Arc::new(new_proc);
            tracing::info!("admin/load: new model ready");
            (
                StatusCode::OK,
                Json(json!({
                    "status": "loaded",
                    "model_path": req.model_path,
                })),
            )
                .into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "admin/load: failed to start llama-server with new model");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": e.to_string(),
                    "model_path": req.model_path,
                })),
            )
                .into_response()
        }
    }
}
