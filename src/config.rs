//! Application configuration: discovery roots and server command.
//!
//! Loaded from `<config_dir>/mlx-cli/config.toml`. Everything optional with
//! sane defaults so the CLI works with zero configuration.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const DEFAULT_PORT_BASE: u16 = 8080;

/// Path to the config file: `<config_dir>/mlx-cli/config.toml`.
pub fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("mlx-cli")
        .join("config.toml")
}

/// User configuration, all optional.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Directories to scan for MLX models, in addition to the defaults
    /// (HuggingFace cache and ~/.mlx-models/models).
    pub model_dirs: Option<Vec<PathBuf>>,
    /// Fixed port per model short name: { qwen = 8081 }. Deterministic
    /// across stop/start so downstream tools can rely on it.
    pub ports: Option<std::collections::HashMap<String, u16>>,
    /// First port tried for models without a fixed port.
    pub port_base: Option<u16>,
    /// Command used to launch servers (a python with mlx-lm / mlx-vlm installed).
    pub server: Option<ServerConfig>,
    /// Extra CLI args passed to the server module, per model:
    /// `"Qwen3-8B-4bit" = ["--kv-bits", "4"]`. Keys match the short or
    /// full (org/name) model name, exactly like `ports`. mlx-cli does not
    /// interpret these; the server module validates them.
    pub start_args: Option<std::collections::HashMap<String, Vec<String>>>,
    /// Extra environment variables for the server process, per model:
    /// `"Qwen3-8B-4bit" = { KV_BITS = "4" }`. Same key matching as above.
    pub start_env:
        Option<std::collections::HashMap<String, std::collections::BTreeMap<String, String>>>,
}

/// Server launch configuration.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// e.g. "/opt/homebrew/bin/python3" or ["uv", "run", "python"]
    pub command: StringOrVec,
}

impl ServerConfig {
    /// Full argv to launch the server.
    pub fn argv(&self) -> Vec<String> {
        self.command.to_vec()
    }
}

/// Accepts either a single string or a list of strings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StringOrVec::One(s) => vec![s.clone()],
            StringOrVec::Many(v) => v.clone(),
        }
    }
}

impl Config {
    /// (a typo should not be silently ignored).
    pub fn load() -> anyhow::Result<Self> {
        let path = config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)?;
        let cfg = toml::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("invalid config at {}:\n{e}", path.display()))?;
        Ok(cfg)
    }

    /// Discovery roots: HuggingFace cache, ~/.mlx-models/models, then extras.
    pub fn model_dirs(&self) -> Vec<PathBuf> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let mut dirs = vec![
            home.join(".cache/huggingface/hub"),
            home.join(".mlx-models/models"),
        ];
        if let Some(extra) = &self.model_dirs {
            dirs.extend(extra.iter().cloned());
        }
        dirs
    }

    /// Persist config to its file, creating the directory. Overwrites.
    pub fn save(&self) -> anyhow::Result<PathBuf> {
        let path = config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = toml::to_string_pretty(self)?;
        std::fs::write(&path, &body)?;
        Ok(path)
    }

    /// Extra server args pinned for a model, if any.
    pub fn start_args(&self, model: &crate::discover::Model) -> Vec<String> {
        let Some(map) = self.start_args.as_ref() else {
            return Vec::new();
        };
        map.get(model.short_name())
            .or_else(|| map.get(&model.name))
            .cloned()
            .unwrap_or_default()
    }

    /// Extra environment variables pinned for a model, if any.
    pub fn start_env(&self, model: &crate::discover::Model) -> Vec<(String, String)> {
        let Some(map) = self.start_env.as_ref() else {
            return Vec::new();
        };
        map.get(model.short_name())
            .or_else(|| map.get(&model.name))
            .map(|env| env.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    }

    /// Port pinned in config for a model, if any.
    pub fn pinned_port(&self, model: &crate::discover::Model) -> Option<u16> {
        let ports = self.ports.as_ref()?;
        let short = model.short_name();
        ports.get(short).or_else(|| ports.get(&model.name)).copied()
    }

    /// Port for a model: fixed from `ports` map (short or full name match),
    /// else first free from port_base.
    pub fn port_for(&self, model: &crate::discover::Model) -> u16 {
        self.pinned_port(model)
            .unwrap_or_else(|| crate::server::first_free_port(self.port_base()))
    }

    /// First port tried when starting a server.
    pub fn port_base(&self) -> u16 {
        self.port_base.unwrap_or(DEFAULT_PORT_BASE)
    }
}
