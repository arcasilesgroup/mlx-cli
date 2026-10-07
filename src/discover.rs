//! MLX model discovery.
//!
//! A model is a directory containing `config.json` + at least one
//! `*.safetensors`. Scans the HuggingFace hub cache (resolving the
//! `snapshots/<hash>` layout) plus any extra dirs from config.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::config::Config;
use crate::error::{MlxError, Result};

/// A discovered MLX model.
#[derive(Debug, Clone, Serialize)]
pub struct Model {
    /// Short name used on the command line (dir name).
    pub name: String,
    /// Weights directory (what mlx-vlm/mlx-lm load).
    pub path: PathBuf,
    /// Sum of *.safetensors on disk, bytes - identity across copies.
    pub weights_bytes: u64,
    /// Total size of all files on disk, bytes.
    pub size_bytes: u64,
    pub params: Option<String>,
    /// Architecture from config.json, e.g. "qwen2_5_vl" or "llama".
    pub architecture: Option<String>,
    /// True when the model has an image processor (vision config) and must
    /// be served by mlx_vlm; false means plain text -> mlx_lm.
    pub vision: bool,
    /// Where the weights live on disk (human-shortened home).
    pub location: String,
}

impl Model {
    /// Name without org prefix, for identity comparison across origins.
    pub fn short_name(&self) -> &str {
        self.name.rsplit('/').next().unwrap_or(&self.name)
    }
}

/// Roots to scan.
pub fn scan_roots(cfg: &Config) -> Vec<PathBuf> {
    cfg.model_dirs()
}

/// True if the directory looks like a loadable MLX model.
pub fn is_model_dir(dir: &Path) -> bool {
    dir.join("config.json").is_file() && has_safetensors(dir)
}

fn has_safetensors(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        })
        .unwrap_or(false)
}

/// Discover every model across all roots. The same model (same weights) in
/// several places is one row; the first hit (HF cache first) wins for path.
pub fn discover_all(cfg: &Config) -> Vec<Model> {
    let mut models: Vec<Model> = Vec::new();
    for root in scan_roots(cfg) {
        let is_hf_cache = root.ends_with(".cache/huggingface/hub");
        for model in scan_root(root, is_hf_cache) {
            if !models.iter().any(|m| {
                m.short_name() == model.short_name() && m.weights_bytes == model.weights_bytes
            }) {
                models.push(model);
            }
        }
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

/// Scan one root. HF cache root inspects `models--org--name/snapshots/<hash>`.
fn scan_root(root: PathBuf, is_hf_cache: bool) -> Vec<Model> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(&root) else {
        return out;
    };
    for entry in entries.filter_map(std::result::Result::ok) {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let candidates: Vec<PathBuf> = if is_hf_cache {
            snapshot_dirs(&path)
        } else {
            vec![path.clone()]
        };
        for cand in candidates {
            if is_model_dir(&cand) {
                let dir_name = if is_hf_cache {
                    hf_model_name(&path)
                } else {
                    path.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                };
                // Prefer the recorded hub origin (`org/model`) over the dir
                // name, so renamed installs still identify their source.
                let name = origin_of(&cand).unwrap_or(dir_name);
                if let Some(model) = build_model(name, cand) {
                    out.push(model);
                }
            }
        }
    }
    out
}

/// For `models--org--name`, return each `snapshots/<hash>` dir that holds weights.
fn snapshot_dirs(hub_dir: &Path) -> Vec<PathBuf> {
    let snap = hub_dir.join("snapshots");
    fs::read_dir(snap)
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default()
}

/// `models--Qwen--Qwen3-8B` -> `Qwen/Qwen3-8B`.
fn hf_model_name(hub_dir: &Path) -> String {
    hub_dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .strip_prefix("models--")
        .unwrap_or_default()
        .replace("--", "/")
}

/// Hub id recorded by `mlx download` in the model dir's `.mlx-origin`.
fn origin_of(model_dir: &Path) -> Option<String> {
    let raw = fs::read_to_string(model_dir.join(".mlx-origin")).ok()?;
    let repo = raw.trim().to_string();
    (!repo.is_empty() && repo.contains('/')).then_some(repo)
}

/// Assemble a `Model`, reading size and metadata. Missing config.json fields
/// are fine (both are optional).
fn build_model(name: String, path: PathBuf) -> Option<Model> {
    let size_bytes = dir_size(&path);
    let weights_bytes = weights_size(&path);
    let (params, architecture, vision) = read_config(&path);
    // HF snapshot paths carry a commit hash - show the hub dir instead.
    let display_path = path
        .ancestors()
        .find(|p| {
            p.file_name()
                .is_some_and(|f| f.to_string_lossy().starts_with("models--"))
        })
        .unwrap_or(&path);
    let location = shorten_home(display_path);
    Some(Model {
        name,
        location,
        path,
        weights_bytes,
        size_bytes,
        params,
        architecture,
        vision,
    })
}

/// "/Users/soy/..." -> "~/..." for display.
fn shorten_home(p: &Path) -> String {
    let s = p.display().to_string();
    dirs::home_dir()
        .map(|h| s.replacen(&h.display().to_string(), "~", 1))
        .unwrap_or(s)
}

/// Sum of *.safetensors sizes - stable identity of a model copy.
fn weights_size(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Recursive directory size. Skips unreadable entries.
pub fn dir_size(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .map(|e| {
                    let p = e.path();
                    if p.is_dir() {
                        dir_size(&p)
                    } else {
                        e.metadata().map(|m| m.len()).unwrap_or(0)
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

/// Pull `num_parameters`-style hints and `model_type` + vision flag from
/// config.json / processor files.
fn read_config(path: &Path) -> (Option<String>, Option<String>, bool) {
    let vision =
        |v: &serde_json::Value| v.get("vision_config").is_some() || v.get("image_config").is_some();
    let mut params = None;
    let mut arch = None;
    let mut is_vision = path.join("preprocessor_config.json").is_file()
        || path.join("processor_config.json").is_file();
    if let Ok(raw) = fs::read_to_string(path.join("config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            is_vision = is_vision || vision(&v);
            params = v
                .get("num_parameters")
                .and_then(serde_json::Value::as_u64)
                .map(human_params);
            arch = v
                .get("model_type")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
        }
    }
    (params, arch, is_vision)
}

/// 8_127_000_000 -> "8.1B"
fn human_params(n: u64) -> String {
    let b = n as f64 / 1e9;
    if b >= 1.0 {
        format!("{b:.1}B")
    } else {
        format!("{:.0}M", n as f64 / 1e6)
    }
}

/// Resolve a CLI model name to a `Model`. Accepts `name`, `org/name`, or a
/// full path. Errors listing what was searched.
pub fn resolve(name: &str, cfg: &Config) -> Result<Model> {
    let models = discover_all(cfg);
    models
        .iter()
        .find(|m| {
            m.name == name
                || m.name.ends_with(&format!("/{name}"))
                || m.path.file_name().is_some_and(|f| f == name)
        })
        .cloned()
        .ok_or_else(|| MlxError::ModelNotFound {
            name: name.to_string(),
            searched: cfg
                .model_dirs()
                .iter()
                .map(|p| p.display().to_string())
                .collect(),
        })
        .map_err(anyhow::Error::from)
}
