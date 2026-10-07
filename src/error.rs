use std::fmt;

/// Errors with actionable, user-facing messages.
///
/// Every variant tells the user what to do next instead of just what broke.
#[derive(Debug)]
pub enum MlxError {
    /// Model name matches nothing on disk; includes what was searched.
    ModelNotFound { name: String, searched: Vec<String> },
    /// Server claims to run but refuses connections.
    PortNotListening { port: u16, pid: u32 },
    /// Server binary/interpreter configured but not found.
    ServerCommandNotFound { command: String },
}

impl fmt::Display for MlxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlxError::ModelNotFound { name, searched } => {
                write!(f, "model '{name}' not found\n  searched:\n")?;
                for dir in searched {
                    writeln!(f, "    - {dir}")?;
                }
                write!(
                    f,
                    "  run 'mlx list' to see discovered models, or add extra directories in mlx-cli config.toml"
                )
            }
            MlxError::PortNotListening { port, pid } => {
                write!(f, "server pid {pid} is not listening on port {port}\n  the server may still be loading weights; check 'mlx status' in a few seconds")
            }
            MlxError::ServerCommandNotFound { command } => {
                write!(f, "server command '{command}' not found\n  set server.command in mlx-cli config.toml to your python with mlx-vlm/mlx-lm installed")
            }
        }
    }
}

impl std::error::Error for MlxError {}

/// Crate-wide alias: caller-facing errors are anyhow; typed errors auto-convert.
pub type Result<T, E = anyhow::Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_not_found_message_lists_searched_dirs() {
        let err = MlxError::ModelNotFound {
            name: "foo".into(),
            searched: vec!["/a".into(), "/b".into()],
        };
        let msg = err.to_string();
        assert!(msg.contains("foo") && msg.contains("/a") && msg.contains("/b"));
    }
}
