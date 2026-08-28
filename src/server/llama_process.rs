//! Managed `llama-server` subprocess.
//!
//! [`LlamaProcess`] spawns and supervises a `llama-server` child process,
//! forwarding stdout/stderr to `tracing`, polling `/health` until the backend
//! is ready, and providing graceful shutdown on drop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::config::{AppConfig, BackendVariant};

// ── Process state ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessState {
    Starting,
    Ready,
    Crashed,
    Stopped,
}

/// True if a llama-server stderr line reports a fatal GPU/backend error that
/// leaves the process running but permanently unable to serve requests.
fn is_fatal_backend_error(line: &str) -> bool {
    line.contains("device lost") || line.contains("ErrorDeviceLost")
}

/// Fail fast if `port` is already bound by some other process.
///
/// Without this check, spawning a new llama-server child whose bind fails
/// (port already held by a stray/orphaned process from a previous run) can
/// still pass health-checking — `/health` on that port answers, just from
/// the wrong process — leaving two model instances loaded and traffic
/// routed unpredictably between them. Bailing here turns that silent,
/// hard-to-diagnose failure mode into an immediate, clear error instead.
async fn check_port_available(port: u16) -> Result<()> {
    match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(listener) => {
            drop(listener);
            Ok(())
        }
        Err(e) => {
            bail!(
                "port {port} is already in use (likely a stray llama-server process from a \
                 previous run) — refusing to start a second instance against it; free the port \
                 (check `tasklist`/`Get-Process llama-server`) and retry: {e}"
            )
        }
    }
}

// ── LlamaProcess ─────────────────────────────────────────────────────────────

/// A supervised `llama-server` subprocess.
///
/// Spawn with [`LlamaProcess::start`]; the call blocks (async) until the
/// backend `/health` endpoint returns 200 or the startup timeout expires.
pub struct LlamaProcess {
    config: Arc<AppConfig>,
    state: Arc<Mutex<ProcessState>>,
    child: Arc<Mutex<Option<Child>>>,
    /// Reqwest client reused for health polling.
    health_client: reqwest::Client,
    /// Reqwest client used for proxying requests (long timeout).
    proxy_client: reqwest::Client,
}

impl LlamaProcess {
    /// Spawn the backend and wait until it is ready.
    pub async fn start(config: Arc<AppConfig>) -> Result<Self> {
        let health_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .context("failed to build reqwest health client")?;

        let proxy_client = reqwest::Client::builder()
            .build()
            .context("failed to build reqwest proxy client")?;

        let state = Arc::new(Mutex::new(ProcessState::Starting));
        let child = Arc::new(Mutex::new(None::<Child>));

        let proc = LlamaProcess {
            config,
            state,
            child,
            health_client,
            proxy_client,
        };
        proc.spawn_child().await?;
        proc.wait_until_ready().await?;

        Ok(proc)
    }

    // ── Internal spawn ────────────────────────────────────────────────────────

    async fn spawn_child(&self) -> Result<()> {
        let port = self.config.backend.internal_port;
        check_port_available(port).await?;

        let args = build_args(&self.config);
        let binary = &self.config.backend.binary_path;

        tracing::info!(
            binary = %binary.display(),
            args = ?args,
            "spawning llama-server subprocess"
        );

        let mut cmd = Command::new(binary);
        cmd.args(&args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // Prevent the subprocess from inheriting the parent's Ctrl-C handler
            // on Windows so we can manage shutdown ourselves.
            .kill_on_drop(false);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn {}", binary.display()))?;

        // Forward stdout lines to tracing at DEBUG level.
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "llama_server", "{}", line);
                }
            });
        }

        // Forward stderr lines to tracing at INFO level (llama-server progress goes there).
        if let Some(stderr) = child.stderr.take() {
            let state_ref = Arc::clone(&self.state);
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::info!(target: "llama_server", "{}", line);

                    // A lost GPU device (e.g. Vulkan `device lost`) leaves the
                    // process running with its HTTP server still up but every
                    // decode failing — stderr never closes, so the
                    // exit-based detection below would never fire and the
                    // watchdog would see a "healthy" process forever. Flag it
                    // the same way a real exit is flagged so the existing
                    // watchdog restart logic picks it up.
                    if is_fatal_backend_error(&line) {
                        let mut s = state_ref.lock().await;
                        if *s == ProcessState::Ready || *s == ProcessState::Starting {
                            *s = ProcessState::Crashed;
                            tracing::warn!(
                                "llama-server backend reported a fatal GPU error — marking crashed"
                            );
                        }
                        break;
                    }
                }
                // When stderr closes the process has exited.
                let mut s = state_ref.lock().await;
                if *s == ProcessState::Ready || *s == ProcessState::Starting {
                    *s = ProcessState::Crashed;
                    tracing::warn!("llama-server subprocess exited unexpectedly");
                }
            });
        }

        *self.child.lock().await = Some(child);
        *self.state.lock().await = ProcessState::Starting;
        Ok(())
    }

    // ── Health polling ────────────────────────────────────────────────────────

    async fn wait_until_ready(&self) -> Result<()> {
        let port = self.config.backend.internal_port;
        let timeout_secs = self.config.backend.startup_timeout_secs;
        let health_url = format!("http://127.0.0.1:{port}/health");

        tracing::info!(url = %health_url, timeout = timeout_secs, "waiting for llama-server to become ready");

        let deadline = Instant::now() + Duration::from_secs(timeout_secs);

        loop {
            // Check if process died during startup.
            if *self.state.lock().await == ProcessState::Crashed {
                bail!("llama-server subprocess crashed during startup");
            }

            match self.health_client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    *self.state.lock().await = ProcessState::Ready;
                    tracing::info!("llama-server is ready");
                    return Ok(());
                }
                Ok(resp) => {
                    tracing::debug!(status = %resp.status(), "health poll: not yet ready");
                }
                Err(e) => {
                    tracing::debug!(error = %e, "health poll: connection refused (still loading)");
                }
            }

            if Instant::now() >= deadline {
                bail!(
                    "llama-server did not become ready within {timeout_secs}s — check binary path and model path"
                );
            }

            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Returns the current process state.
    pub async fn state(&self) -> ProcessState {
        self.state.lock().await.clone()
    }

    /// Returns `true` if the subprocess is running and healthy.
    #[allow(dead_code)]
    pub async fn is_healthy(&self) -> bool {
        if *self.state.lock().await != ProcessState::Ready {
            return false;
        }
        let port = self.config.backend.internal_port;
        let url = format!("http://127.0.0.1:{port}/health");
        self.health_client
            .get(&url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Returns the OS PID of the subprocess, if running.
    pub async fn pid(&self) -> Option<u32> {
        self.child.lock().await.as_ref().and_then(|c| c.id())
    }

    /// Kill the current subprocess and respawn with the same (or updated) config.
    ///
    /// `new_config`, if provided, replaces the stored config before respawning.
    pub async fn restart(&self, new_config: Option<Arc<AppConfig>>) -> Result<()> {
        tracing::info!("restarting llama-server subprocess");
        self.kill().await;

        if let Some(cfg) = new_config {
            // Safety: we only write the config pointer here, behind the lock.
            // SAFETY: AppConfig is Clone but Arc is not reassignable on &self.
            // We work around by cloning into a local and updating the process.
            // The config field is not directly settable on &self; callers that
            // need a different config must drop this LlamaProcess and start a new one.
            // This path is reserved for future use.
            let _ = cfg; // suppress unused warning
        }

        self.spawn_child().await?;
        self.wait_until_ready().await
    }

    /// Send a kill signal to the subprocess without waiting for it to exit.
    pub async fn kill(&self) {
        let mut guard = self.child.lock().await;
        if let Some(ref mut child) = *guard {
            let _ = child.start_kill();
            tracing::info!("kill signal sent to llama-server (pid={:?})", child.id());
        }
        *guard = None;
        *self.state.lock().await = ProcessState::Stopped;
    }

    /// The internal HTTP base URL for the subprocess (e.g. `http://127.0.0.1:7433`).
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.config.backend.internal_port)
    }

    /// A reference to the reqwest client, for reuse by the proxy layer.
    pub fn http_client(&self) -> &reqwest::Client {
        &self.proxy_client
    }
}

impl Drop for LlamaProcess {
    fn drop(&mut self) {
        // best-effort: send kill signal synchronously.
        if let Ok(mut guard) = self.child.try_lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.start_kill();
            }
        }
    }
}

// ── CLI argument builder ──────────────────────────────────────────────────────

/// Build the llama-server CLI argument list from `config`.
fn build_args(config: &AppConfig) -> Vec<String> {
    let m = &config.model;
    let b = &config.backend;

    let mut args: Vec<String> = Vec::new();

    // Core model and network flags.
    args.extend([
        "--model".into(),
        m.model_path.to_string_lossy().into_owned(),
    ]);
    args.extend(["--host".into(), "127.0.0.1".into()]);
    args.extend(["--port".into(), b.internal_port.to_string()]);

    // GPU flags.
    args.extend(["--n-gpu-layers".into(), m.n_gpu_layers.to_string()]);
    if m.main_gpu >= 0 {
        args.extend(["--main-gpu".into(), m.main_gpu.to_string()]);
    }
    if !m.tensor_split.is_empty() && m.tensor_split != [1.0_f32] {
        let split: String = m
            .tensor_split
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        args.extend(["--tensor-split".into(), split]);
    } else if m.main_gpu >= 0 {
        // If a main GPU is specified but no split weights are given, restrict the model entirely to the main GPU.
        args.extend(["--split-mode".into(), "none".into()]);
    }

    // Context and batching.
    args.extend(["--ctx-size".into(), m.context_size.to_string()]);
    args.extend(["--batch-size".into(), m.batch_size.to_string()]);
    args.extend(["--threads".into(), m.threads.to_string()]);

    // Optional mmproj (vision).
    if let Some(ref mmproj) = m.mmproj_path {
        args.extend(["--mmproj".into(), mmproj.to_string_lossy().into_owned()]);
    }

    // KV Cache settings for native llama-server.
    if b.variant == BackendVariant::LlamaServer {
        args.extend([
            "--cache-type-k".into(),
            config.kv_cache.type_k.as_cli_str().into(),
        ]);
        args.extend([
            "--cache-type-v".into(),
            config.kv_cache.type_v.as_cli_str().into(),
        ]);
    }

    // Prompt-cache RAM budget (llama-server's `--cache-ram`, host RAM used to
    // store slot checkpoints for fast prefix reuse — separate from the
    // active on-GPU KV cache). Left unset, llama-server applies its own
    // 8192 MiB default; set explicitly here once a value is configured so
    // it's a deliberate choice rather than a silent default.
    if let Some(mb) = config.kv_cache.memory_budget_mb {
        args.extend(["--cache-ram".into(), mb.to_string()]);
    }

    // TurboQuant variant: append the custom KV cache type flag.
    if b.variant == BackendVariant::TurboQuant {
        args.extend(["--cache-type-k".into(), "turbo3".into()]);
    }

    // Speculative decoding and chat-template flags. `[backend]` fields already
    // hold the effective per-model values — the caller merges `[models.load]`
    // overrides onto a cloned `BackendConfig` before starting the process, so
    // this function only ever reads one flat set of fields.
    if let Some(ref spec_type) = b.spec_type {
        args.extend(["--spec-type".into(), spec_type.clone()]);
        if let Some(n_max) = b.spec_draft_n_max {
            args.extend(["--spec-draft-n-max".into(), n_max.to_string()]);
        }
    }
    if let Some(ref draft_model) = b.draft_model {
        args.extend([
            "--spec-draft-model".into(),
            draft_model.to_string_lossy().into_owned(),
        ]);
    }
    if let Some(ref kwargs) = b.chat_template_kwargs {
        args.extend(["--chat-template-kwargs".into(), kwargs.to_string()]);
    }

    // User-supplied extra flags (verbatim).
    args.extend(b.extra_flags.iter().cloned());

    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BackendConfig, BackendVariant, ModelConfig};
    use std::path::PathBuf;

    fn test_config() -> AppConfig {
        AppConfig {
            model: ModelConfig {
                model_path: PathBuf::from("/models/test.gguf"),
                n_gpu_layers: -1,
                main_gpu: 1,
                tensor_split: vec![16.0, 32.0],
                context_size: 8192,
                batch_size: 512,
                threads: 4,
                ..ModelConfig::default()
            },
            backend: BackendConfig {
                variant: BackendVariant::LlamaServer,
                binary_path: PathBuf::from("llama-server"),
                internal_port: 7433,
                startup_timeout_secs: 30,
                restart_on_crash: true,
                extra_flags: vec!["--flash-attn".into()],
                ..BackendConfig::default()
            },
            ..AppConfig::default()
        }
    }

    #[tokio::test]
    async fn check_port_available_succeeds_on_free_port() {
        // Bind port 0 to let the OS hand back a free ephemeral port, then
        // release it immediately so the check has something free to find.
        let probe = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);

        assert!(check_port_available(port).await.is_ok());
    }

    #[tokio::test]
    async fn check_port_available_fails_when_port_held() {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();

        let err = check_port_available(port).await.unwrap_err();
        assert!(err.to_string().contains("already in use"));

        drop(listener);
    }

    #[test]
    fn is_fatal_backend_error_detects_device_lost() {
        assert!(is_fatal_backend_error(
            "ggml_vulkan: device lost on Vulkan1"
        ));
        assert!(is_fatal_backend_error(
            "srv update_slots: decode() failed: vk::Device::getFenceStatus: ErrorDeviceLost"
        ));
        assert!(!is_fatal_backend_error("srv update_slots: processing task"));
    }

    #[test]
    fn cache_ram_flag_omitted_when_unset() {
        let cfg = test_config();
        let args = build_args(&cfg);
        assert!(!args.iter().any(|a| a == "--cache-ram"));
    }

    #[test]
    fn cache_ram_flag_included_when_set() {
        let mut cfg = test_config();
        cfg.kv_cache.memory_budget_mb = Some(4096);
        let args = build_args(&cfg);
        assert!(args.windows(2).any(|w| w == ["--cache-ram", "4096"]));
    }

    #[test]
    fn build_args_includes_core_flags() {
        let cfg = test_config();
        let args = build_args(&cfg);
        assert!(args
            .windows(2)
            .any(|w| w == ["--model", "/models/test.gguf"]));
        assert!(args.windows(2).any(|w| w == ["--port", "7433"]));
        assert!(args.windows(2).any(|w| w == ["--n-gpu-layers", "-1"]));
        assert!(args.windows(2).any(|w| w == ["--ctx-size", "8192"]));
        assert!(args.windows(2).any(|w| w == ["--main-gpu", "1"]));
        assert!(args.windows(2).any(|w| w == ["--tensor-split", "16,32"]));
        assert!(args.iter().any(|a| a == "--flash-attn"));
    }

    #[test]
    fn turbo_quant_adds_cache_flag() {
        let mut cfg = test_config();
        cfg.backend.variant = BackendVariant::TurboQuant;
        let args = build_args(&cfg);
        assert!(args.windows(2).any(|w| w == ["--cache-type-k", "turbo3"]));
    }

    #[test]
    fn spec_decoding_flags_omitted_when_unset() {
        let cfg = test_config();
        let args = build_args(&cfg);
        assert!(!args.iter().any(|a| a == "--spec-type"));
        assert!(!args.iter().any(|a| a == "--spec-draft-n-max"));
        assert!(!args.iter().any(|a| a == "--spec-draft-model"));
    }

    #[test]
    fn spec_decoding_flags_included_when_set() {
        let mut cfg = test_config();
        cfg.backend.spec_type = Some("draft-mtp".into());
        cfg.backend.spec_draft_n_max = Some(2);
        let args = build_args(&cfg);
        assert!(args.windows(2).any(|w| w == ["--spec-type", "draft-mtp"]));
        assert!(args.windows(2).any(|w| w == ["--spec-draft-n-max", "2"]));
    }

    #[test]
    fn draft_model_flag_included_when_set() {
        let mut cfg = test_config();
        cfg.backend.spec_type = Some("draft-dspark".into());
        cfg.backend.draft_model = Some(PathBuf::from("/models/dspark-drafter.gguf"));
        let args = build_args(&cfg);
        assert!(args
            .windows(2)
            .any(|w| w == ["--spec-draft-model", "/models/dspark-drafter.gguf"]));
    }

    #[test]
    fn chat_template_kwargs_serialized_as_json() {
        let mut cfg = test_config();
        cfg.backend.chat_template_kwargs = Some(serde_json::json!({"reasoning_effort": "medium"}));
        let args = build_args(&cfg);
        assert!(args.windows(2).any(
            |w| w[0] == "--chat-template-kwargs" && w[1] == r#"{"reasoning_effort":"medium"}"#
        ));
    }

    #[test]
    fn no_main_gpu_when_negative() {
        let mut cfg = test_config();
        cfg.model.main_gpu = -1;
        let args = build_args(&cfg);
        assert!(!args.iter().any(|a| a == "--main-gpu"));
    }
}
