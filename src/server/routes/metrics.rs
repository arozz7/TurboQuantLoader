//! Observability endpoints.
//!
//! | Method | Path                | Format     |
//! |--------|---------------------|------------|
//! | GET    | `/health`           | JSON       |
//! | GET    | `/metrics`          | Prometheus |
//! | GET    | `/v1/admin/stats`   | JSON       |

use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

use crate::server::llama_process::ProcessState;
use crate::server::AppState;

// ── /health ───────────────────────────────────────────────────────────────────

/// `GET /health` — rich JSON health response.
pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let proc = state.process_snapshot().await;
    let cfg = state.config_snapshot().await;
    let backend_state = proc.state().await;
    let pid = proc.pid().await;
    let gpu_stats = state.metrics.gpu_stats.read().await.clone();
    let (tps_p50, tps_p95, tps_p99) = state.metrics.tps_percentiles().await;
    let (ttft_p50, ttft_p95, ttft_p99) = state.metrics.ttft_percentiles().await;
    let (gen_p50, gen_p95, gen_p99) = state.metrics.generation_ms_percentiles().await;
    let (stop_count, length_count, tool_calls_count) = state.metrics.finish_reason_counts().await;

    let status = if backend_state == ProcessState::Ready {
        "ok"
    } else {
        "degraded"
    };

    let model_name = cfg
        .model
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let gpus: Vec<serde_json::Value> = gpu_stats
        .iter()
        .map(|g| {
            json!({
                "device": g.device_index,
                "name": g.name,
                "vram_used_mb": g.vram_used_mb,
                "vram_total_mb": g.vram_total_mb,
                "utilization_pct": g.utilization_pct,
            })
        })
        .collect();

    Json(json!({
        "status": status,
        "backend": {
            "state": format!("{backend_state:?}"),
            "pid": pid,
            "variant": format!("{:?}", cfg.backend.variant),
            "model": model_name,
            "context_size": cfg.model.context_size,
            "uptime_secs": state.metrics.uptime_secs(),
        },
        "inference": {
            "total_requests": state.metrics.total_requests.load(std::sync::atomic::Ordering::Relaxed),
            "total_errors": state.metrics.total_errors.load(std::sync::atomic::Ordering::Relaxed),
            "active_requests": state.metrics.active_requests.load(std::sync::atomic::Ordering::Relaxed),
            "total_tokens_generated": state.metrics.total_tokens_generated.load(std::sync::atomic::Ordering::Relaxed),
            "total_prompt_tokens": state.metrics.total_prompt_tokens.load(std::sync::atomic::Ordering::Relaxed),
            "finish_reasons": {
                "stop": stop_count,
                "length": length_count,
                "tool_calls": tool_calls_count,
            },
        },
        "performance": {
            "tps_p50": tps_p50,
            "tps_p95": tps_p95,
            "tps_p99": tps_p99,
            "ttft_p50_ms": ttft_p50,
            "ttft_p95_ms": ttft_p95,
            "ttft_p99_ms": ttft_p99,
            "generation_p50_ms": gen_p50,
            "generation_p95_ms": gen_p95,
            "generation_p99_ms": gen_p99,
        },
        "gpus": gpus,
    }))
}

// ── /metrics (Prometheus) ─────────────────────────────────────────────────────

/// `GET /metrics` — Prometheus text format exposition.
pub async fn prometheus_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let total_req = state
        .metrics
        .total_requests
        .load(std::sync::atomic::Ordering::Relaxed);
    let total_err = state
        .metrics
        .total_errors
        .load(std::sync::atomic::Ordering::Relaxed);
    let total_tokens = state
        .metrics
        .total_tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    let total_prompt = state
        .metrics
        .total_prompt_tokens
        .load(std::sync::atomic::Ordering::Relaxed);
    let active = state
        .metrics
        .active_requests
        .load(std::sync::atomic::Ordering::Relaxed);
    let (tps_p50, tps_p95, tps_p99) = state.metrics.tps_percentiles().await;
    let (ttft_p50, ttft_p95, ttft_p99) = state.metrics.ttft_percentiles().await;
    let (gen_p50, gen_p95, gen_p99) = state.metrics.generation_ms_percentiles().await;
    let (stop_count, length_count, tool_calls_count) = state.metrics.finish_reason_counts().await;
    let gpu_stats = state.metrics.gpu_stats.read().await.clone();

    let mut out = String::with_capacity(2048);

    out.push_str("# HELP tql_requests_total Total inference requests received\n");
    out.push_str("# TYPE tql_requests_total counter\n");
    out.push_str(&format!("tql_requests_total {total_req}\n\n"));

    out.push_str("# HELP tql_errors_total Total inference errors\n");
    out.push_str("# TYPE tql_errors_total counter\n");
    out.push_str(&format!("tql_errors_total {total_err}\n\n"));

    out.push_str("# HELP tql_active_requests Requests currently in flight\n");
    out.push_str("# TYPE tql_active_requests gauge\n");
    out.push_str(&format!("tql_active_requests {active}\n\n"));

    out.push_str("# HELP tql_tokens_generated_total Cumulative completion tokens produced\n");
    out.push_str("# TYPE tql_tokens_generated_total counter\n");
    out.push_str(&format!("tql_tokens_generated_total {total_tokens}\n\n"));

    out.push_str("# HELP tql_prompt_tokens_total Cumulative prompt tokens processed\n");
    out.push_str("# TYPE tql_prompt_tokens_total counter\n");
    out.push_str(&format!("tql_prompt_tokens_total {total_prompt}\n\n"));

    out.push_str("# HELP tql_finish_reason_total Requests by finish reason (recent window)\n");
    out.push_str("# TYPE tql_finish_reason_total gauge\n");
    out.push_str(&format!(
        "tql_finish_reason_total{{reason=\"stop\"}} {stop_count}\n"
    ));
    out.push_str(&format!(
        "tql_finish_reason_total{{reason=\"length\"}} {length_count}\n"
    ));
    out.push_str(&format!(
        "tql_finish_reason_total{{reason=\"tool_calls\"}} {tool_calls_count}\n\n"
    ));

    out.push_str("# HELP tql_tps_p50 Token generation throughput p50 (tokens/s)\n");
    out.push_str("# TYPE tql_tps_p50 gauge\n");
    out.push_str(&format!("tql_tps_p50 {tps_p50:.2}\n\n"));

    out.push_str("# HELP tql_tps_p95 Token generation throughput p95 (tokens/s)\n");
    out.push_str("# TYPE tql_tps_p95 gauge\n");
    out.push_str(&format!("tql_tps_p95 {tps_p95:.2}\n\n"));

    out.push_str("# HELP tql_tps_p99 Token generation throughput p99 (tokens/s)\n");
    out.push_str("# TYPE tql_tps_p99 gauge\n");
    out.push_str(&format!("tql_tps_p99 {tps_p99:.2}\n\n"));

    out.push_str("# HELP tql_ttft_p50_ms Time to first token p50 (ms)\n");
    out.push_str("# TYPE tql_ttft_p50_ms gauge\n");
    out.push_str(&format!("tql_ttft_p50_ms {ttft_p50}\n\n"));

    out.push_str("# HELP tql_ttft_p95_ms Time to first token p95 (ms)\n");
    out.push_str("# TYPE tql_ttft_p95_ms gauge\n");
    out.push_str(&format!("tql_ttft_p95_ms {ttft_p95}\n\n"));

    out.push_str("# HELP tql_ttft_p99_ms Time to first token p99 (ms)\n");
    out.push_str("# TYPE tql_ttft_p99_ms gauge\n");
    out.push_str(&format!("tql_ttft_p99_ms {ttft_p99}\n\n"));

    out.push_str("# HELP tql_generation_p50_ms Total generation time p50 (ms)\n");
    out.push_str("# TYPE tql_generation_p50_ms gauge\n");
    out.push_str(&format!("tql_generation_p50_ms {gen_p50}\n\n"));

    out.push_str("# HELP tql_generation_p95_ms Total generation time p95 (ms)\n");
    out.push_str("# TYPE tql_generation_p95_ms gauge\n");
    out.push_str(&format!("tql_generation_p95_ms {gen_p95}\n\n"));

    out.push_str("# HELP tql_generation_p99_ms Total generation time p99 (ms)\n");
    out.push_str("# TYPE tql_generation_p99_ms gauge\n");
    out.push_str(&format!("tql_generation_p99_ms {gen_p99}\n\n"));

    out.push_str("# HELP tql_uptime_seconds Server uptime in seconds\n");
    out.push_str("# TYPE tql_uptime_seconds counter\n");
    out.push_str(&format!(
        "tql_uptime_seconds {}\n\n",
        state.metrics.uptime_secs()
    ));

    if !gpu_stats.is_empty() {
        out.push_str("# HELP tql_gpu_vram_used_mb GPU VRAM used (MB)\n");
        out.push_str("# TYPE tql_gpu_vram_used_mb gauge\n");
        for g in &gpu_stats {
            out.push_str(&format!(
                "tql_gpu_vram_used_mb{{device=\"{}\",name=\"{}\"}} {}\n",
                g.device_index, g.name, g.vram_used_mb
            ));
        }
        out.push('\n');

        out.push_str("# HELP tql_gpu_vram_total_mb GPU VRAM total (MB)\n");
        out.push_str("# TYPE tql_gpu_vram_total_mb gauge\n");
        for g in &gpu_stats {
            out.push_str(&format!(
                "tql_gpu_vram_total_mb{{device=\"{}\",name=\"{}\"}} {}\n",
                g.device_index, g.name, g.vram_total_mb
            ));
        }
        out.push('\n');

        out.push_str("# HELP tql_gpu_utilization_pct GPU compute utilization (%)\n");
        out.push_str("# TYPE tql_gpu_utilization_pct gauge\n");
        for g in &gpu_stats {
            out.push_str(&format!(
                "tql_gpu_utilization_pct{{device=\"{}\",name=\"{}\"}} {}\n",
                g.device_index, g.name, g.utilization_pct
            ));
        }
        out.push('\n');
    }

    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        out,
    )
}

// ── /v1/admin/stats ───────────────────────────────────────────────────────────

/// `GET /v1/admin/stats` — full histogram JSON for the last 100 requests.
pub async fn admin_stats(State(state): State<AppState>) -> impl IntoResponse {
    let recent = state.metrics.recent_snapshot().await;
    let (tps_p50, tps_p95, tps_p99) = state.metrics.tps_percentiles().await;
    let (ttft_p50, ttft_p95, ttft_p99) = state.metrics.ttft_percentiles().await;
    let (gen_p50, gen_p95, gen_p99) = state.metrics.generation_ms_percentiles().await;
    let (stop_count, length_count, tool_calls_count) = state.metrics.finish_reason_counts().await;

    let requests: Vec<serde_json::Value> = recent
        .iter()
        .map(|r| {
            json!({
                "ttft_ms": r.ttft_ms,
                "generation_ms": r.generation_ms,
                "tokens_per_second": r.tokens_per_second,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "finish_reason": r.finish_reason,
            })
        })
        .collect();

    Json(json!({
        "total_requests": state.metrics.total_requests.load(std::sync::atomic::Ordering::Relaxed),
        "total_errors": state.metrics.total_errors.load(std::sync::atomic::Ordering::Relaxed),
        "active_requests": state.metrics.active_requests.load(std::sync::atomic::Ordering::Relaxed),
        "total_tokens_generated": state.metrics.total_tokens_generated.load(std::sync::atomic::Ordering::Relaxed),
        "total_prompt_tokens": state.metrics.total_prompt_tokens.load(std::sync::atomic::Ordering::Relaxed),
        "uptime_secs": state.metrics.uptime_secs(),
        "finish_reasons": {
            "stop": stop_count,
            "length": length_count,
            "tool_calls": tool_calls_count,
        },
        "percentiles": {
            "tps_p50": tps_p50,
            "tps_p95": tps_p95,
            "tps_p99": tps_p99,
            "ttft_p50_ms": ttft_p50,
            "ttft_p95_ms": ttft_p95,
            "ttft_p99_ms": ttft_p99,
            "generation_p50_ms": gen_p50,
            "generation_p95_ms": gen_p95,
            "generation_p99_ms": gen_p99,
        },
        "recent_requests": requests,
    }))
}
