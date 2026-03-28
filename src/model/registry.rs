use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::warn;

/// Metadata for a discovered GGUF model file.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Short display name derived from the file stem (no extension).
    pub name: String,
    /// Absolute path to the GGUF file.
    pub path: PathBuf,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Model architecture string read from the GGUF header.
    ///
    /// `None` in Phase 1 — populated in Phase 2 when the GGUF header is parsed.
    pub arch: Option<String>,
    /// Quantization type string, e.g. `"IQ3_XXS"`.
    ///
    /// `None` in Phase 1 — populated in Phase 2 when the GGUF header is parsed.
    pub quant_type: Option<String>,
    /// `true` when a sibling `mmproj-*.gguf` vision projector exists in the
    /// same directory as this model.
    pub has_mmproj: bool,
}

/// Discovers and indexes GGUF model files on disk.
pub struct ModelRegistry;

impl ModelRegistry {
    /// Recursively scan `dir` for `*.gguf` files and return model metadata.
    ///
    /// Vision projector files (`mmproj-*.gguf`) are excluded from the returned
    /// list but their presence sets [`ModelEntry::has_mmproj`] on the primary
    /// model entry in the same directory.
    pub fn scan(dir: &Path) -> Result<Vec<ModelEntry>> {
        let mut paths: Vec<PathBuf> = Vec::new();
        collect_gguf_files(dir, &mut paths)?;

        let (mmproj_paths, model_paths): (Vec<_>, Vec<_>) =
            paths.into_iter().partition(|p| is_mmproj(p));

        let entries = model_paths
            .into_iter()
            .map(|path| {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let size_bytes = std::fs::metadata(&path)
                    .map(|m| m.len())
                    .unwrap_or_else(|e| {
                        warn!(path = %path.display(), error = %e, "failed to stat model file");
                        0
                    });

                let has_mmproj = mmproj_paths
                    .iter()
                    .any(|mp| mp.parent() == path.parent());

                ModelEntry {
                    name,
                    path,
                    size_bytes,
                    arch: None,
                    quant_type: None,
                    has_mmproj,
                }
            })
            .collect();

        Ok(entries)
    }

    /// Find the first entry whose name contains `query` (case-insensitive).
    pub fn find_by_name<'a>(entries: &'a [ModelEntry], query: &str) -> Option<&'a ModelEntry> {
        let lower = query.to_lowercase();
        entries.iter().find(|e| e.name.to_lowercase().contains(&lower))
    }
}

/// `true` when the path's filename begins with `"mmproj-"`.
fn is_mmproj(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.starts_with("mmproj-"))
        .unwrap_or(false)
}

/// Recursively collect all `*.gguf` paths under `dir`, skipping unreadable
/// entries with a warning rather than aborting the scan.
fn collect_gguf_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    let read_dir = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read directory: {}", dir.display()))?;

    for entry in read_dir {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!(dir = %dir.display(), error = %e, "skipping unreadable directory entry");
                continue;
            }
        };

        let path = entry.path();

        if path.is_dir() {
            collect_gguf_files(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            out.push(path);
        }
    }

    Ok(())
}
