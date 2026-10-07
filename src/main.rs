//! CLI entry: clap definitions + command dispatch.

mod config;
mod delete;
mod discover;
mod download;
mod error;
mod server;
mod setup;
mod table;

use clap::{Parser, Subcommand};
use dialoguer::Confirm;

use crate::config::Config;
use crate::error::Result;

#[derive(Parser)]
#[command(
    name = "mlx",
    version,
    about = "Manage local MLX model servers on Apple Silicon",
    after_help = "Models are auto-discovered from the HuggingFace cache\nand ~/.mlx-models/models. Add extra directories in mlx-cli/config.toml."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List discovered models with details and running state.
    List,
    /// Start a model's server (interactive picker if no model given).
    Start {
        /// Model name (see mlx list); omit to pick interactively.
        model: Option<String>,
        /// Skip the tight-memory confirmation.
        #[arg(long)]
        yes: bool,
    },
    /// Stop a model's server (interactive picker if no model given).
    Stop {
        /// Model name (see mlx list); omit to pick among running ones.
        model: Option<String>,
    },
    /// Stop every running server.
    StopAll,
    /// Pin the port of a model (persists in config).
    SetPort {
        /// Model name (see mlx list); omit to pick interactively.
        model: Option<String>,
        /// New fixed port.
        port: u16,
    },
    /// Download a model from HuggingFace into ~/.mlx-models/models/.
    Download {
        /// Repo id (org/name) or a huggingface.co URL.
        repo: String,
        /// Re-download even if the model seems already installed.
        #[arg(long)]
        force: bool,
        /// Assume yes; do not prompt for overwrites.
        #[arg(long)]
        yes: bool,
    },
    /// Delete an installed model from disk (picker if no model given).
    Delete {
        /// Model name (see mlx list); omit to pick interactively.
        model: Option<String>,
        /// Do not ask for confirmation.
        #[arg(long)]
        yes: bool,
    },
    /// Show per-model state and system memory.
    Status,
}

fn main() {
    // Piping into `head` etc. closes stdout early; Rust's default panic on
    // EPIPE is noise for a table-printing CLI. Restore unix default.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let cli = Cli::parse();
    let mut cfg = Config::load()?;
    match cli.cmd {
        Cmd::List => cmd_list(&cfg),
        Cmd::Start { model, yes } => {
            setup::ensure_server(&mut cfg)?;
            let name = match model {
                Some(n) => n,
                None => pick_model(&cfg, false)?.expect("start picker always returns Some"),
            };
            cmd_start(&cfg, &name, yes)
        }
        Cmd::Stop { model } => {
            let name = match model {
                Some(n) => n,
                None => match pick_model(&cfg, true)? {
                    Some(n) => n,
                    None => return Ok(()),
                },
            };
            server::stop(&name)
        }
        Cmd::StopAll => {
            let stopped = server::stop_all();
            if stopped.is_empty() {
                println!("no servers running");
            } else {
                for m in stopped {
                    println!("stopped {m}");
                }
            }
            Ok(())
        }
        Cmd::SetPort { model, port } => {
            let name = match model {
                Some(n) => n,
                None => pick_model(&cfg, false)?.expect("start picker always returns Some"),
            };
            cmd_set_port(&mut cfg, &name, port)
        }
        Cmd::Download { repo, force, yes } => download::cmd_download(&repo, force, yes),
        Cmd::Delete { model, yes } => {
            let name = match model {
                Some(n) => n,
                None => match pick_model(&cfg, false)? {
                    Some(n) => n,
                    None => return Ok(()),
                },
            };
            delete::cmd_delete(&cfg, &name, yes)
        }
        Cmd::Status => cmd_status(&cfg),
    }
}

fn cmd_list(cfg: &Config) -> Result<()> {
    let models = discover::discover_all(cfg);
    let running = server::running();
    let table = table::list_table(&models, &running, &|m| {
        cfg.pinned_port(m).unwrap_or_else(|| cfg.port_base())
    });
    println!("{table}");
    Ok(())
}

/// Interactive model picker. Shows the list table first. When `running_only`
/// (stop), only running models are offered. Returns None when there is
/// nothing to pick (message already printed) - not an error.
fn pick_model(cfg: &Config, running_only: bool) -> Result<Option<String>> {
    let models = discover::discover_all(cfg);
    let running = server::running();
    let candidates: Vec<discover::Model> = models
        .into_iter()
        .filter(|m| {
            !running_only
                || running
                    .iter()
                    .any(|s| s.model == m.name || s.model.ends_with(m.short_name()))
        })
        .collect();
    if candidates.is_empty() {
        if running_only {
            println!("no models running");
            return Ok(None);
        }
        anyhow::bail!("no models discovered\n  download one with 'mlx download <org/model>'");
    }
    println!(
        "{}\n",
        table::list_table(&candidates, &running, &|m| cfg
            .pinned_port(m)
            .unwrap_or_else(|| cfg.port_base()))
    );
    let items: Vec<String> = candidates
        .iter()
        .map(|m| {
            let port = cfg.pinned_port(m).unwrap_or_else(|| cfg.port_base());
            let state = if running.iter().any(|s| s.model == m.name) {
                " [running]"
            } else {
                ""
            };
            format!("{m} (:{port}){state}", m = m.name)
        })
        .collect();
    let sel = dialoguer::Select::new()
        .with_prompt(if running_only {
            "Stop which model"
        } else {
            "Which model"
        })
        .items(&items)
        .default(0)
        .interact()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(Some(candidates[sel].name.clone()))
}

/// Pin a model's port and persist the config.
fn cmd_set_port(cfg: &mut Config, name: &str, port: u16) -> Result<()> {
    let model = discover::resolve(name, cfg)?;
    let ports = cfg.ports.get_or_insert_with(Default::default);
    ports.insert(model.short_name().to_string(), port);
    ports.insert(model.name.clone(), port);
    let path = cfg.save()?;
    println!(
        "pinned {} to port {} (saved to {})",
        model.name,
        port,
        path.display()
    );
    Ok(())
}

fn cmd_start(cfg: &Config, name: &str, yes: bool) -> Result<()> {
    let model = discover::resolve(name, cfg)?;
    let headroom = server::headroom_after(model.size_bytes);
    // Warn when free memory after load drops below ~8 GiB.
    if headroom < 8 * 1024 * 1024 * 1024 && !yes {
        eprintln!(
            "warning: starting '{}' ({} on disk) leaves only ~{} of free memory",
            model.name,
            table::human_bytes(model.size_bytes),
            table::human_bytes(headroom)
        );
        let confirmed = Confirm::new()
            .with_prompt("Continue anyway?")
            .default(false)
            .interact()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if !confirmed {
            println!("aborted");
            return Ok(());
        }
    }
    let state = server::start(&model, cfg)?;
    println!(
        "started {} on port {} (pid {})",
        state.model, state.port, state.pid
    );
    Ok(())
}

fn cmd_status(cfg: &Config) -> Result<()> {
    let models = discover::discover_all(cfg);
    let running = server::running();
    let mut t = table::Table::new(&["MODEL", "STATE", "PID", "PORT", "UPTIME", "MEM"]);
    for m in &models {
        match running.iter().find(|s| s.model == m.name) {
            Some(s) => {
                let mem = server::physical_footprint(s.pid)
                    .map(table::human_bytes)
                    .unwrap_or_else(|| "-".into());
                t.row(vec![
                    m.name.clone(),
                    "running".into(),
                    s.pid.to_string(),
                    s.port.to_string(),
                    server::uptime_since(s.started_at),
                    mem,
                ]);
            }
            None => t.row(vec![
                m.name.clone(),
                "stopped".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
            ]),
        };
    }
    println!("{t}");

    // System memory
    let sys = sysinfo::System::new_all();
    let total = sys.total_memory();
    let used = sys.used_memory();
    println!(
        "\nmemory: {} / {} used ({} free)",
        table::human_bytes(used),
        table::human_bytes(total),
        table::human_bytes(total - used)
    );
    Ok(())
}
