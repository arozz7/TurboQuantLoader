//! Rolling request metrics and GPU telemetry collection.
//!
//! [`MetricsCollector`] is cheaply cloneable (`Arc` internally). A background
//! task polls GPU stats every 2 seconds. Request metrics are recorded after
//! each streaming or non-streaming inference request completes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use crate::gpu_stats::GpuStats;

// ── Per-request record ────────────────────────────────────────────────────────

/// Timing and token counts for a single completed request.
#[derive(Debug, Clone)]
pub struct RequestMetrics {
    /// Milliseconds from request start to first token received.
    pub ttft_ms: u64,
    /// Total milliseconds from request start to generation complete.
    pub generation_ms: u64,
    /// Tokens generated per second during this request.
    pub tokens_per_second: f32,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Why generation stopped: `"stop"`, `"length"`, `"tool_calls"`.
    pub finish_reason: String,
}

// ── Collector ─────────────────────────────────────────────────────────────────

/// Shared, thread-safe rolling metrics store.
pub struct MetricsCollector {
    pub total_requests: AtomicU64,
    pub total_errors: AtomicU64,
    /// Cumulative tokens produced across all completed requests.
    pub total_tokens_generated: AtomicU64,
    /// Cumulative prompt tokens processed across all completed requests.
    pub total_prompt_tokens: AtomicU64,
    /// Number of requests currently in flight (generation in progress).
    pub active_requests: AtomicI64,
    /// Last 100 completed requests.
    recent: RwLock<VecDeque<RequestMetrics>>,
    /// Latest GPU snapshot, refreshed every 2 s.
    pub gpu_stats: RwLock<Vec<GpuStats>>,
    pub started_at: Instant,
}

impl MetricsCollector {
    /// Create the collector and spawn the background GPU poller.
    pub fn start() -> Arc<Self> {
        let collector = Arc::new(Self {
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            active_requests: AtomicI64::new(0),
            recent: RwLock::new(VecDeque::with_capacity(100)),
            gpu_stats: RwLock::new(Vec::new()),
            started_at: Instant::now(),
        });

        let c = Arc::clone(&collector);
        tokio::spawn(async move {
            loop {
                let stats = crate::gpu_stats::query_all_gpus();
                *c.gpu_stats.write().await = stats;
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });

        collector
    }

    pub fn inc_requests(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_errors(&self) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn record(&self, m: RequestMetrics) {
        self.total_tokens_generated
            .fetch_add(m.completion_tokens as u64, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(m.prompt_tokens as u64, Ordering::Relaxed);
        let mut recent = self.recent.write().await;
        if recent.len() >= 100 {
            recent.pop_front();
        }
        recent.push_back(m);
    }

    pub fn uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// p50 / p95 / p99 tokens-per-second across the recent window.
    pub async fn tps_percentiles(&self) -> (f32, f32, f32) {
        let recent = self.recent.read().await;
        if recent.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut vals: Vec<f32> = recent.iter().map(|r| r.tokens_per_second).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        percentiles_f32(&vals)
    }

    /// p50 / p95 / p99 time-to-first-token (ms) across the recent window.
    pub async fn ttft_percentiles(&self) -> (u64, u64, u64) {
        let recent = self.recent.read().await;
        if recent.is_empty() {
            return (0, 0, 0);
        }
        let mut vals: Vec<u64> = recent.iter().map(|r| r.ttft_ms).collect();
        vals.sort_unstable();
        percentiles_u64(&vals)
    }

    /// p50 / p95 / p99 total generation time (ms) across the recent window.
    pub async fn generation_ms_percentiles(&self) -> (u64, u64, u64) {
        let recent = self.recent.read().await;
        if recent.is_empty() {
            return (0, 0, 0);
        }
        let mut vals: Vec<u64> = recent.iter().map(|r| r.generation_ms).collect();
        vals.sort_unstable();
        percentiles_u64(&vals)
    }

    /// Count of requests in the recent window that ended with each finish reason.
    pub async fn finish_reason_counts(&self) -> (u64, u64, u64) {
        let recent = self.recent.read().await;
        let mut stop = 0u64;
        let mut length = 0u64;
        let mut tool_calls = 0u64;
        for r in recent.iter() {
            match r.finish_reason.as_str() {
                "length" => length += 1,
                "tool_calls" | "tool_use" => tool_calls += 1,
                _ => stop += 1,
            }
        }
        (stop, length, tool_calls)
    }

    /// Clone the recent window (for admin/stats JSON).
    pub async fn recent_snapshot(&self) -> Vec<RequestMetrics> {
        self.recent.read().await.iter().cloned().collect()
    }
}

fn percentiles_f32(sorted: &[f32]) -> (f32, f32, f32) {
    let n = sorted.len();
    let p = |pct: f64| sorted[((pct / 100.0) * n as f64).min((n - 1) as f64) as usize];
    (p(50.0), p(95.0), p(99.0))
}

fn percentiles_u64(sorted: &[u64]) -> (u64, u64, u64) {
    let n = sorted.len();
    let p = |pct: f64| sorted[((pct / 100.0) * n as f64).min((n - 1) as f64) as usize];
    (p(50.0), p(95.0), p(99.0))
}
