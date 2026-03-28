mod config;
mod inference;
mod kv_cache;
mod model;
mod server;
mod tui;

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use tracing::info;

use config::{load_from_file, AppConfig, CliOverrides, KvBits};
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

/// Arguments shared by commands that load a model.
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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Load config, falling back to defaults when the file is absent.
    let mut app_config = if cli.config.exists() {
        load_from_file(&cli.config)?
    } else {
        tracing::warn!(
            path = %cli.config.display(),
            "config file not found — using built-in defaults"
        );
        AppConfig::default()
    };

    match cli.command {
        Command::Serve(args) => {
            let overrides = serve_overrides(&args)?;
            app_config.apply_cli_overrides(&overrides);
            cmd_serve(app_config)
        }
        Command::Run(args) => {
            let overrides = run_overrides(&args);
            app_config.apply_cli_overrides(&overrides);
            cmd_run(app_config)
        }
        Command::Bench(args) => {
            app_config.apply_cli_overrides(&CliOverrides::default());
            cmd_bench(app_config, args)
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

// ─── Command handlers ────────────────────────────────────────────────────────

fn cmd_serve(_config: AppConfig) -> Result<()> {
    bail!("serve command not yet implemented — coming in Phase 4");
}

fn cmd_run(_config: AppConfig) -> Result<()> {
    bail!("run command not yet implemented — coming in Phase 2");
}

fn cmd_bench(_config: AppConfig, _args: BenchArgs) -> Result<()> {
    bail!("bench command not yet implemented — coming in Phase 3");
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
    println!(
        "{:<60}  {:>10}  {}",
        "Name", "Size", "Flags"
    );
    println!("{}", "-".repeat(80));

    for entry in &entries {
        let size = format_size(entry.size_bytes);
        let flags = if entry.has_mmproj { "[mmproj]" } else { "" };
        println!("{:<60}  {:>10}  {}", entry.name, size, flags);
    }

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

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
