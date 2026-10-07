//! `mlx delete [model]`: remove an installed model from disk.
//!
//! Only model dirs discovered by `discover` may be deleted, and a running
//! model is refused (stop it first). HF-cache copies remove the whole
//! `models--org--name` dir so no partial snapshots or blobs are left behind.

use crate::config::Config;
use crate::discover::{self, Model};
use crate::error::Result;
use crate::table::human_bytes;

/// Find the HF hub dir that owns `model.path` (snapshots are symlinks into
/// `../../blobs`, so deleting the snapshot alone would leak blob storage).
fn hub_owner_dir(path: &std::path::Path) -> Option<std::path::PathBuf> {
    path.ancestors()
        .find(|p| {
            p.file_name()
                .is_some_and(|f| f.to_string_lossy().starts_with("models--"))
        })
        .map(std::path::Path::to_path_buf)
}

fn prompt_yes_no(prompt: &str, default: bool) -> bool {
    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() {
        return default;
    }
    dialoguer::Confirm::new()
        .with_prompt(prompt)
        .default(default)
        .interact()
        .map_err(|e| anyhow::anyhow!("{e}"))
        .unwrap_or(default)
}

/// Entry point for `mlx delete`. Interactive picker already resolved `name`.
pub fn cmd_delete(cfg: &Config, name: &str, yes: bool) -> Result<()> {
    let model: Model = discover::resolve(name, cfg)?;
    let running = crate::server::running();
    if running.iter().any(|s| {
        s.model == model.name || model.name.ends_with(&s.model) || s.model.ends_with(&model.name)
    }) {
        anyhow::bail!(
            "'{}' is running - stop it first with 'mlx stop {}'",
            model.name,
            model.name
        );
    }

    let hub_dir = hub_owner_dir(&model.path);
    let target = hub_dir.as_ref().unwrap_or(&model.path);
    println!(
        "deleting '{}' ({} at {})",
        model.name,
        human_bytes(model.size_bytes),
        model.location
    );
    if !yes {
        let confirmed = prompt_yes_no("Permanently delete this model?", false);
        if !confirmed {
            println!("aborted");
            return Ok(());
        }
    }

    if !target.starts_with(dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."))) {
        anyhow::bail!(
            "refusing to delete outside the home directory: {}",
            target.display()
        );
    }
    let freed = discover::dir_size(target);
    std::fs::remove_dir_all(target)?;
    println!("deleted {} (freed {})", model.name, human_bytes(freed));
    Ok(())
}
