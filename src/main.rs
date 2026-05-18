mod config;
mod conversation_log;
mod gpu_stats;
mod inference;
mod kv_cache;
mod metrics;
mod model;
mod server;
mod tui;

use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{fmt, EnvFilter};

use config::{load_from_file, AppConfig, CliOverrides, KvBits, KvCacheConfig, LoggingConfig};
use inference::engine::{ChatMessage, ChatRequest, InferenceEngine};
use model::backend::{GenerateEvent, SamplerParams};
use model::registry::ModelRegistry;

// ─── CLI Definition ───────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "turboquant-loader",
    version,
    about = "Local LLM inference server with TurboQuant KV cache compression and OpenAI-compatible API"
)]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the OpenAI-compatible HTTP API server on the configured host:port.
    Serve(ServeArgs),
    /// Open an interactive terminal chat session.
    Run(RunArgs),
    /// Benchmark context sizes versus KV cache configurations.
    Bench(BenchArgs),
    /// List all GGUF models discovered in the configured models directory.
    List,
}

#[derive(Parser)]
struct ServeArgs {
    /// Override the model file to load (overrides `config.toml`).
    #[arg(short, long)]
    model: Option<PathBuf>,
    /// Override the HTTP listen port (overrides `config.toml`).
    #[arg(short, long)]
    port: Option<u16>,
    /// Override the context window size in tokens (overrides `config.toml`).
    #[arg(long)]
    context_size: Option<u32>,
    /// Override the number of GPU layers to offload; `-1` offloads all (overrides `config.toml`).
    #[arg(long)]
    n_gpu_layers: Option<i32>,
    /// Override the KV cache bit-width: `2`, `3`, `4`, or `8` (overrides `config.toml`).
    #[arg(long)]
    kv_bits: Option<u8>,
}

#[derive(Parser)]
struct RunArgs {
    /// Override the model file to load (overrides `config.toml`).
    #[arg(short, long)]
    model: Option<PathBuf>,
    /// Override the context window size in tokens (overrides `config.toml`).
    #[arg(long)]
    context_size: Option<u32>,
}

#[derive(Parser)]
struct BenchArgs {
    /// Comma-separated context sizes to benchmark (e.g. `1024,8192,32768`).
    #[arg(long, default_value = "1024,8192,32768")]
    context_sizes: String,
    /// Comma-separated KV bit-widths to benchmark (e.g. `4,8`).
    #[arg(long, default_value = "4,8")]
    bits: String,
    /// Write benchmark results to this JSON file.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

// ─── Entry Point ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Disable llama.cpp CUDA Graphs globally to prevent WDDM VRAM fragmentation deadlocks
    // when executing deeply contextual batches on Windows.
    std::env::set_var("LLAMA_CUDA_NO_GRAPHS", "1");

    let cli = Cli::parse();

    // Load config before initialising tracing so the file appender knows where to write.
    let mut app_config = if cli.config.exists() {
        load_from_file(&cli.config)?
    } else {
        eprintln!(
            "warn: config file not found at '{}' — using built-in defaults",
            cli.config.display()
        );
        AppConfig::default()
    };

    // Initialise dual-subscriber tracing (stdout + rolling file).
    // _log_guard must stay alive for the duration of main — dropping it flushes the
    // non-blocking writer and terminates the background logging thread.
    let _log_guard = init_tracing(&app_config.logging)?;

    match cli.command {
        Command::Serve(args) => {
            let overrides = serve_overrides(&args)?;
            app_config.apply_cli_overrides(&overrides);
            cmd_serve(app_config).await
        }
        Command::Run(args) => {
            let overrides = run_overrides(&args);
            app_config.apply_cli_overrides(&overrides);
            cmd_run(app_config).await
        }
        Command::Bench(args) => {
            app_config.apply_cli_overrides(&CliOverrides::default());
            cmd_bench(app_config, args).await
        }
        Command::List => cmd_list(&app_config),
    }
}

// ─── Override builders ────────────────────────────────────────────────────────

fn serve_overrides(args: &ServeArgs) -> Result<CliOverrides> {
    let kv_bits = args
        .kv_bits
        .map(KvBits::try_from)
        .transpose()
        .map_err(|e| anyhow::anyhow!("--kv-bits: {e}"))?;

    Ok(CliOverrides {
        model_path: args.model.clone(),
        port: args.port,
        context_size: args.context_size,
        n_gpu_layers: args.n_gpu_layers,
        kv_bits,
    })
}

fn run_overrides(args: &RunArgs) -> CliOverrides {
    CliOverrides {
        model_path: args.model.clone(),
        context_size: args.context_size,
        ..CliOverrides::default()
    }
}

// ─── Command handlers ─────────────────────────────────────────────────────────

async fn cmd_serve(config: AppConfig) -> Result<()> {
    server::serve(config).await
}

/// Interactive terminal chat session.
///
/// Loads the model on a blocking thread, then drives a ChatML REPL until the
/// user types `/quit`, `/exit`, or sends EOF (Ctrl-D).
async fn cmd_run(config: AppConfig) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    println!("Loading model: {}", config.model.model_path.display());
    println!("Context size:  {} tokens", config.model.context_size);
    println!("Type /quit or press Ctrl-D to exit.\n");

    // Model loading is blocking; run it off the async executor.
    let engine = tokio::task::spawn_blocking(move || InferenceEngine::new(config))
        .await
        .map_err(|e| anyhow::anyhow!("inference thread panicked: {e}"))??;

    println!(
        "Model ready: {} ({} ctx)\n",
        engine.model_name(),
        engine.context_size()
    );

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut history: Vec<ChatMessage> = Vec::new();

    loop {
        // Print prompt and flush.
        print!("> ");
        std::io::stdout().flush()?;

        let line = match lines.next_line().await? {
            Some(l) => l,
            None => break, // EOF (Ctrl-D)
        };
        let line = line.trim().to_string();

        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }

        history.push(ChatMessage {
            role: "user".into(),
            content: line,
        });

        let req = ChatRequest {
            messages: history.clone(),
            max_tokens: 2048,
            sampler: SamplerParams::default(),
        };

        let mut stream = engine.chat(req)?;
        let mut response = String::new();

        loop {
            match stream.next_event().await {
                Some(GenerateEvent::Token(tok)) => {
                    print!("{tok}");
                    std::io::stdout().flush()?;
                    response.push_str(&tok);
                }
                Some(GenerateEvent::Done(summary)) => {
                    println!();
                    info!(
                        tokens = summary.tokens_generated,
                        tps = format!("{:.1}", summary.tokens_per_second),
                        ctx = summary.context_tokens,
                        "generation complete"
                    );
                    break;
                }
                Some(GenerateEvent::Error(e)) => {
                    eprintln!("\nerror: {e}");
                    break;
                }
                None => break,
            }
        }

        history.push(ChatMessage {
            role: "assistant".into(),
            content: response,
        });
        println!();
    }

    Ok(())
}

/// Benchmark (context_size × kv_bits) combinations against a fixed prompt.
///
/// The model is loaded once; each combination only rebuilds the `LlamaContext`
/// (~50 ms). Results are printed as a table and optionally written to JSON.
async fn cmd_bench(config: AppConfig, args: BenchArgs) -> Result<()> {
    // ── Parse arguments ───────────────────────────────────────────────────────
    let context_sizes: Vec<u32> = args
        .context_sizes
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("invalid context size: '{}'", s.trim()))
        })
        .collect::<Result<_>>()?;

    let kv_bits_list: Vec<KvBits> = args
        .bits
        .split(',')
        .map(|s| {
            let n: u8 = s
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid kv_bits: '{}'", s.trim()))?;
            KvBits::try_from(n).map_err(|e| anyhow::anyhow!("{e}"))
        })
        .collect::<Result<_>>()?;

    // ── Load bench prompt ─────────────────────────────────────────────────────
    let prompt_path = std::path::Path::new("docs/bench_prompt.txt");
    let prompt = if prompt_path.exists() {
        std::fs::read_to_string(prompt_path)
            .map_err(|e| anyhow::anyhow!("failed to read bench prompt: {e}"))?
    } else {
        "Explain the transformer attention mechanism in detail.".to_string()
    };

    // ── Load model ────────────────────────────────────────────────────────────
    println!(
        "Benchmarking: {} context sizes × {} kv_bits = {} configurations\n",
        context_sizes.len(),
        kv_bits_list.len(),
        context_sizes.len() * kv_bits_list.len()
    );
    println!("Loading model: {}", config.model.model_path.display());

    let engine = tokio::task::spawn_blocking(move || InferenceEngine::new(config))
        .await
        .map_err(|e| anyhow::anyhow!("inference thread panicked: {e}"))??;

    println!("Model ready: {}\n", engine.model_name());

    // ── Print table header ────────────────────────────────────────────────────
    println!(
        "{:<12} {:<8} {:>8} {:>10} {:>10}",
        "ctx_size", "kv_bits", "tokens", "tps", "ctx_used"
    );
    println!("{}", "-".repeat(52));

    #[derive(serde::Serialize)]
    struct BenchRow {
        ctx_size: u32,
        kv_bits: u8,
        tokens_generated: u32,
        tokens_per_second: f32,
        context_tokens: u32,
    }
    let mut rows: Vec<BenchRow> = Vec::new();

    // ── Run combinations ──────────────────────────────────────────────────────
    for &ctx_size in &context_sizes {
        for &bits in &kv_bits_list {
            let kv_cfg = KvCacheConfig {
                bits,
                ..KvCacheConfig::default()
            };

            // Reconfigure is blocking — move off the async executor.
            let reconf_result = {
                let engine_ref = &engine;
                let kv_cfg_ref = &kv_cfg;
                tokio::task::block_in_place(|| engine_ref.reconfigure_context(ctx_size, kv_cfg_ref))
            };
            if let Err(e) = reconf_result {
                eprintln!(
                    "  skip ctx={ctx_size} bits={}: reconfigure failed: {e}",
                    u8::from(bits)
                );
                continue;
            }

            let req = ChatRequest {
                messages: vec![ChatMessage {
                    role: "user".into(),
                    content: prompt.clone(),
                }],
                max_tokens: 256,
                sampler: SamplerParams {
                    temperature: 0.1,
                    ..SamplerParams::default()
                },
            };

            let mut stream = engine.chat(req)?;
            let mut summary_opt = None;

            loop {
                match stream.next_event().await {
                    Some(GenerateEvent::Token(_)) => {}
                    Some(GenerateEvent::Done(s)) => {
                        summary_opt = Some(s);
                        break;
                    }
                    Some(GenerateEvent::Error(e)) => {
                        eprintln!("  error ctx={ctx_size} bits={}: {e}", u8::from(bits));
                        break;
                    }
                    None => break,
                }
            }

            if let Some(summary) = summary_opt {
                println!(
                    "{:<12} {:<8} {:>8} {:>10.1} {:>10}",
                    ctx_size,
                    u8::from(bits),
                    summary.tokens_generated,
                    summary.tokens_per_second,
                    summary.context_tokens,
                );
                rows.push(BenchRow {
                    ctx_size,
                    kv_bits: u8::from(bits),
                    tokens_generated: summary.tokens_generated,
                    tokens_per_second: summary.tokens_per_second,
                    context_tokens: summary.context_tokens,
                });
            }
        }
    }

    // ── Optional JSON output ──────────────────────────────────────────────────
    if let Some(ref path) = args.output {
        let json = serde_json::to_string_pretty(&rows)
            .map_err(|e| anyhow::anyhow!("JSON serialization failed: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| anyhow::anyhow!("failed to write output file: {e}"))?;
        println!("\nResults written to: {}", path.display());
    }

    Ok(())
}

/// List all GGUF models discovered in `config.model.models_dir`.
fn cmd_list(config: &AppConfig) -> Result<()> {
    let dir = &config.model.models_dir;

    info!(models_dir = %dir.display(), "scanning for models");

    let entries = ModelRegistry::scan(dir)?;

    if entries.is_empty() {
        println!("No models found in: {}", dir.display());
        return Ok(());
    }

    println!("Found {} model(s) in: {}\n", entries.len(), dir.display());
    println!("{:<60}  {:>10}  {}", "Name", "Size", "Flags");
    println!("{}", "-".repeat(80));

    for entry in &entries {
        let size = format_size(entry.size_bytes);
        let flags = if entry.has_mmproj { "[mmproj]" } else { "" };
        println!("{:<60}  {:>10}  {}", entry.name, size, flags);
    }

    Ok(())
}

// ─── Logging ──────────────────────────────────────────────────────────────────

/// Initialise a dual-subscriber tracing setup:
///
/// - **stdout** — colored, human-readable; level from `RUST_LOG` env var or
///   `cfg.stdout_log_level`.
/// - **file** — plain text, daily rolling in `cfg.log_dir`; level from
///   `cfg.file_log_level`. The file name pattern is
///   `turboquant.<YYYY-MM-DD>.log`.
///
/// The returned [`WorkerGuard`] must be kept alive for the duration of the
/// program. Dropping it flushes the non-blocking writer buffer.
fn init_tracing(cfg: &LoggingConfig) -> Result<WorkerGuard> {
    // Ensure log directory exists.
    std::fs::create_dir_all(&cfg.log_dir)
        .with_context(|| format!("failed to create log directory: {}", cfg.log_dir.display()))?;

    // Purge files older than the retention window before opening a new one.
    if cfg.log_retention_days > 0 {
        cleanup_old_logs(&cfg.log_dir, cfg.log_retention_days);
    }

    // Daily rolling file appender.
    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("turboquant")
        .filename_suffix("log")
        .build(&cfg.log_dir)
        .context("failed to create rolling file appender")?;

    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // Stdout layer — colored, level from RUST_LOG or config.
    let stdout_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cfg.stdout_log_level));

    let stdout_layer = fmt::layer().with_target(false).with_filter(stdout_filter);

    // File layer — no ANSI, full target path, independent level.
    let file_filter = EnvFilter::new(&cfg.file_log_level);

    let file_layer = fmt::layer()
        .with_ansi(false)
        .with_target(true)
        .with_writer(non_blocking)
        .with_filter(file_filter);

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(file_layer)
        .init();

    Ok(guard)
}

/// Delete log files in `log_dir` whose last-modified time is older than
/// `retention_days` days. Skips files it cannot stat or remove, logging a
/// warning instead of aborting.
fn cleanup_old_logs(log_dir: &PathBuf, retention_days: u32) {
    let cutoff = match SystemTime::now()
        .checked_sub(Duration::from_secs(u64::from(retention_days) * 86_400))
    {
        Some(t) => t,
        None => return,
    };

    let Ok(entries) = std::fs::read_dir(log_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        // Only touch files that look like our log files.
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.starts_with("turboquant") || !name.ends_with(".log") {
            continue;
        }

        let modified = match entry.metadata().and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("warn: could not stat log file {}: {e}", path.display());
                continue;
            }
        };

        if modified < cutoff {
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!(
                    "warn: could not remove old log file {}: {e}",
                    path.display()
                );
            } else {
                eprintln!("info: removed old log file: {}", path.display());
            }
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Format a byte count as a human-readable string (GB / MB / KB).
fn format_size(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;
    const KB: u64 = 1_024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}
