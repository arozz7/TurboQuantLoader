use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::warn;

use crate::config::{AppConfig, ModelDefinition};

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
    #[allow(dead_code)]
    pub arch: Option<String>,
    /// Quantization type string, e.g. `"IQ3_XXS"`.
    ///
    /// `None` in Phase 1 — populated in Phase 2 when the GGUF header is parsed.
    #[allow(dead_code)]
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

        let (mmproj_paths, remaining): (Vec<_>, Vec<_>) =
            paths.into_iter().partition(|p| is_mmproj(p));
        let (_drafter_paths, remaining): (Vec<_>, Vec<_>) =
            remaining.into_iter().partition(|p| is_drafter(p));
        // Multi-part GGUFs (`*-00002-of-00004.gguf`, etc.) are loaded by
        // llama-server as a set once given the first shard's path — list only
        // that first shard so split models don't show up as N separate,
        // individually-unloadable entries.
        let (_continuation_shards, model_paths): (Vec<_>, Vec<_>) =
            remaining.into_iter().partition(|p| is_split_continuation(p));

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

                let has_mmproj = mmproj_paths.iter().any(|mp| mp.parent() == path.parent());

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
        entries
            .iter()
            .find(|e| e.name.to_lowercase().contains(&lower))
    }

    /// Resolve a model `name` to a [`ModelDefinition`] using `config`.
    ///
    /// Resolution order:
    /// 1. Named registry (`config.models`) — exact match, then substring match.
    /// 2. `models_dir` filesystem scan — substring match on file stem.
    ///
    /// Returns `None` when the name is unknown in both sources.
    pub fn resolve(name: &str, config: &AppConfig) -> Option<ModelDefinition> {
        let lower = name.to_lowercase();

        // 1. Exact name match in registry.
        if let Some(def) = config
            .models
            .iter()
            .find(|m| m.name.to_lowercase() == lower)
        {
            return Some(def.clone());
        }

        // 2. Substring match in registry.
        if let Some(def) = config
            .models
            .iter()
            .find(|m| m.name.to_lowercase().contains(&lower))
        {
            return Some(def.clone());
        }

        // 3. Filesystem scan fallback — returns a synthetic definition with no overrides.
        if let Ok(entries) = Self::scan(&config.model.models_dir) {
            if let Some(entry) = Self::find_by_name(&entries, name) {
                return Some(ModelDefinition {
                    name: entry.name.clone(),
                    path: entry.path.clone(),
                    context_size: None,
                    n_gpu_layers: None,
                    main_gpu: None,
                    batch_size: None,
                    tensor_split: None,
                    load: None,
                });
            }
        }

        None
    }

    /// Returns `true` when `requested` refers to the currently-loaded model.
    ///
    /// Treats `"local"` and empty strings as wildcards (always matches). Otherwise
    /// checks for case-insensitive equality or whether the current model name
    /// contains the requested string, so `"qwen3"` matches `"Qwen3.6-27B-Q4_K_S"`.
    pub fn matches_current(requested: &str, current: &str) -> bool {
        if requested.is_empty() || requested.eq_ignore_ascii_case("local") {
            return true;
        }
        let req = requested.to_lowercase();
        let cur = current.to_lowercase();
        cur == req || cur.contains(&req)
    }
}

/// `true` when the path's filename begins with `"mmproj-"`.
fn is_mmproj(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.starts_with("mmproj-"))
        .unwrap_or(false)
}

/// `true` when the path's filename begins with `"dspark-"` — the naming
/// convention for external speculative-decoding drafter GGUFs (e.g.
/// DeepSeek-V4-Flash's DSpark drafter, loaded via `--spec-draft-model`).
/// Excluded from `scan` results so it doesn't show up as a selectable model.
fn is_drafter(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.starts_with("dspark-"))
        .unwrap_or(false)
}

/// Parses the shard index out of a multi-part GGUF filename stem ending in
/// `-<index>-of-<total>` (e.g. `"model-00002-of-00004"` → `Some(2)`).
/// Returns `None` for filenames that don't match this convention.
fn split_shard_index(path: &Path) -> Option<u32> {
    let stem = path.file_stem()?.to_str()?;
    let of_pos = stem.rfind("-of-")?;
    let total = &stem[of_pos + 4..];
    if total.is_empty() || !total.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let before = &stem[..of_pos];
    let dash_pos = before.rfind('-')?;
    let index = &before[dash_pos + 1..];
    if index.is_empty() {
        return None;
    }
    index.parse().ok()
}

/// `true` for every shard of a multi-part GGUF except the first — llama-server
/// loads the whole set from the first shard's path.
fn is_split_continuation(path: &Path) -> bool {
    matches!(split_shard_index(path), Some(idx) if idx != 1)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_drafter_matches_dspark_prefix() {
        assert!(is_drafter(Path::new(
            "/models/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
        )));
        assert!(!is_drafter(Path::new(
            "/models/DeepSeek-V4-Flash-0731-UD-IQ3_XXS.gguf"
        )));
    }

    #[test]
    fn is_mmproj_matches_mmproj_prefix() {
        assert!(is_mmproj(Path::new("/models/mmproj-F16.gguf")));
        assert!(!is_mmproj(Path::new("/models/Qwen3.8-27B-Q4_K_S.gguf")));
    }

    #[test]
    fn split_shard_index_parses_multi_part_filenames() {
        assert_eq!(
            split_shard_index(Path::new(
                "/models/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
            )),
            Some(1)
        );
        assert_eq!(
            split_shard_index(Path::new(
                "/models/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00002-of-00004.gguf"
            )),
            Some(2)
        );
        assert_eq!(
            split_shard_index(Path::new("/models/Qwen3.8-27B-Q4_K_S.gguf")),
            None
        );
    }

    #[test]
    fn is_split_continuation_keeps_only_first_shard() {
        assert!(!is_split_continuation(Path::new("/models/x-00001-of-00004.gguf")));
        assert!(is_split_continuation(Path::new("/models/x-00002-of-00004.gguf")));
        assert!(is_split_continuation(Path::new("/models/x-00004-of-00004.gguf")));
        assert!(!is_split_continuation(Path::new("/models/single-file.gguf")));
    }
}
