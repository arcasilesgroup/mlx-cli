//! Server lifecycle: start, stop, stop-all, state in pidfiles.
//!
//! State dir: `<data_local_dir>/mlx-cli/`. One JSON pidfile per running
//! server (`<model-sanitized>.json`) holding pid, port, model path and start
//! time. A pidfile whose pid is dead is stale and cleaned on sight.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::discover::{self, Model};
use crate::error::{MlxError, Result};

/// Live state of one server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerState {
    pub pid: u32,
    pub port: u16,
    /// Model name as shown by `mlx list`.
    pub model: String,
    /// Weights directory passed to the server.
    pub model_path: String,
    /// When the server was started.
    pub started_at: SystemTime,
}

/// Directory holding pidfiles.
pub fn state_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("mlx-cli")
}

fn pidfile(model: &str) -> PathBuf {
    state_dir().join(format!("{}.json", model.replace('/', "__")))
}

/// Read one model's state, cleaning stale pidfiles.
pub fn state_of(model: &str) -> Option<ServerState> {
    let path = pidfile(model);
    let raw = fs::read_to_string(&path).ok()?;
    let state: ServerState = serde_json::from_str(&raw).ok()?;
    if process_alive(state.pid) {
        Some(state)
    } else {
        let _ = fs::remove_file(&path);
        None
    }
}

/// All running servers (stale entries removed).
pub fn running() -> Vec<ServerState> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(state_dir()) else {
        return out;
    };
    for e in entries.filter_map(std::result::Result::ok) {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "json") {
            if let Some(state) = read_pidfile(&p) {
                out.push(state);
            }
        }
    }
    out.sort_by(|a, b| a.model.cmp(&b.model));
    out
}

fn read_pidfile(path: &Path) -> Option<ServerState> {
    let raw = fs::read_to_string(path).ok()?;
    let state: ServerState = serde_json::from_str(&raw).ok()?;
    if process_alive(state.pid) {
        Some(state)
    } else {
        let _ = fs::remove_file(path);
        None
    }
}

/// true if a process with this pid exists (macOS: kill 0).
fn process_alive(pid: u32) -> bool {
    // kill(pid, 0) returns Ok for existing, permitted processes. EPERM means
    // it exists but is owned by someone else - still "alive" for our purposes.
    let pid = pid as i32;
    unsafe {
        libc::kill(pid, 0) == 0
            || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
}

/// First free TCP port at or after `base`.
pub fn first_free_port(base: u16) -> u16 {
    for port in base..base + 100 {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    base
}

/// Estimated free memory headroom if we load `weight_bytes` more, in bytes.
pub fn headroom_after(weight_bytes: u64) -> u64 {
    let sys = sysinfo::System::new_all();
    let total = sys.total_memory();
    let used = sys.used_memory();
    total.saturating_sub(used + weight_bytes)
}

/// Pick the serving module for a model: `mlx_vlm.server` when the model has
/// an image processor / vision config, `mlx_lm.server` for text-only.
fn server_module(model: &Model) -> &'static str {
    if model.vision {
        "mlx_vlm.server"
    } else {
        "mlx_lm.server"
    }
}

/// Refuse to spawn when the configured python cannot serve this model kind,
/// telling the user exactly what to install.
fn ensure_serves(cfg: &Config, model: &Model) -> Result<()> {
    let Some(rt) = crate::setup::configured_runtime(cfg) else {
        return Err(anyhow::Error::msg(crate::setup::install_guidance(
            "mlx_vlm",
        )));
    };
    if rt.serves(model.vision) {
        return Ok(());
    }
    let missing = if model.vision { "mlx_vlm" } else { "mlx_lm" };
    Err(anyhow::Error::msg(crate::setup::install_guidance(missing)))
}

/// Spawn the mlx server for `model` on the first free port from config.
///
/// Returns the state after confirming the port accepts connections.
pub fn start(model: &Model, cfg: &Config) -> Result<ServerState> {
    ensure_serves(cfg, model)?;
    let module = server_module(model);
    if let Some(existing) = state_of(&model.name) {
        return Ok(existing);
    }
    let port = cfg.port_for(model);
    let argv = cfg
        .server
        .as_ref()
        .ok_or_else(|| MlxError::ServerCommandNotFound {
            command: "<unset: config.server.command>".into(),
        })?
        .argv();

    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .arg("-m")
        .arg(module)
        .arg("--model")
        .arg(&model.path)
        .arg("--port")
        .arg(port.to_string())
        .args(cfg.start_args(model))
        .stdout(Stdio::null())
        // Truncate per start; read back on early death (unknown flag,
        // import error, OOM). A file cannot deadlock the way a full
        // unread pipe would stall a chatty server.
        .stderr(Stdio::from(
            fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(server_log(port))
                .map_err(|e| anyhow::Error::msg(format!("cannot open server log: {e}")))?,
        ));
    for (k, v) in cfg.start_env(model) {
        cmd.env(k, v);
    }
    let child = cmd.spawn().map_err(|_| MlxError::ServerCommandNotFound {
        command: argv.join(" "),
    })?;
    let pid = child.id();

    let state = ServerState {
        pid,
        port,
        model: model.name.clone(),
        model_path: model.path.display().to_string(),
        started_at: SystemTime::now(),
    };
    fs::create_dir_all(state_dir())?;
    fs::write(pidfile(&model.name), serde_json::to_string(&state)?)?;

    // Wait until the port accepts connections or the child dies. On death,
    // surface the server's own stderr tail (argparse errors on unknown
    // start_args, import failures, OOMs) so config mistakes are fixable
    // without rerunning by hand.
    if !wait_port(port, pid, child, Duration::from_secs(120)) {
        let _ = fs::remove_file(pidfile(&model.name));
        return Err(anyhow::Error::msg(match server_log_tail(port) {
            Some(err) => format!(
                "server for '{}' died before serving on port {port}:\n{err}",
                model.name
            ),
            None => MlxError::PortNotListening { port, pid }.to_string(),
        }));
    }
    Ok(state)
}

/// Path of the stderr log `start` keeps for a server on `port`.
fn server_log(port: u16) -> PathBuf {
    state_dir().join(format!("serve-{port}.log"))
}

/// Last lines of the stderr log for `port`, if any.
fn server_log_tail(port: u16) -> Option<String> {
    let raw = fs::read_to_string(server_log(port)).ok()?;
    let mut tail: Vec<&str> = raw.lines().rev().take(15).collect();
    tail.reverse();
    Some(tail.join("\n"))
}

/// Poll the port until it answers AND is owned by our child, the child
/// exits, or `timeout`.
fn wait_port(port: u16, pid: u32, mut child: Child, timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        // Child first: another process may already own the port (stale
        // server, SO_REUSEADDR) - a connect success means nothing if our
        if let Ok(Some(_)) = child.try_wait() {
            return false;
        }
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            // Connect success only counts when the listen socket belongs
            // to our child or its direct descendant (uvicorn with an
            // import-string app spawns a worker process; mlx_lm binds
            // in-process). Anything else is a pre-existing server making
            // our spawn look healthy.
            if let Some(listener) = listening_pid(port) {
                if listener == pid || ppid(listener) == Some(pid) {
                    return true;
                }
            }
        }
        if std::time::Instant::now() > deadline || !process_alive(pid) {
            let _ = child.kill();
            return false;
        }
        thread::sleep(Duration::from_millis(500));
    }
}

/// Parent PID of `pid`, if the process exists.
fn ppid(pid: u32) -> Option<u32> {
    let out = Command::new("ps")
        .args(["-o", "ppid=", &pid.to_string()])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// PID listening on `port` (macOS: lsof), if any.
fn listening_pid(port: u16) -> Option<u32> {
    let out = Command::new("lsof")
        // "-iTCP:<port>" must be ONE token; lsof 4.91 rejects it split.
        .args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-t"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Stop one model's server: SIGTERM, wait, SIGKILL if needed. Cleans pidfile.
pub fn stop(model_name: &str) -> Result<()> {
    let cfg = crate::config::Config::load()?;
    let Some(state) = state_of(model_name) else {
        // Model may exist but not be running - that's fine.
        let exists = discover::discover_all(&cfg)
            .iter()
            .any(|m| m.name == model_name || m.name.ends_with(&format!("/{model_name}")));
        if exists {
            return Ok(());
        }
        return Err(anyhow::Error::from(MlxError::ModelNotFound {
            name: model_name.to_string(),
            searched: cfg.model_dirs()[..]
                .iter()
                .map(|p| p.display().to_string())
                .collect(),
        }));
    };
    send_signal(state.pid, libc::SIGTERM);
    for _ in 0..50 {
        if !process_alive(state.pid) {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    if process_alive(state.pid) {
        send_signal(state.pid, libc::SIGKILL);
    }
    let _ = fs::remove_file(pidfile(model_name));
    Ok(())
}
/// True memory of a pid, bytes: the task's physical footprint (weights
/// mapped for the GPU count here). RSS is meaningless for MLX servers -
/// Metal keeps mapped weights out of resident pages, so a 15 GB model
/// reports ~40 MB of RSS. `vmmap` exposes the footprint; slow (~1 s) but
/// `status` only runs it for live servers.
pub fn physical_footprint(pid: u32) -> Option<u64> {
    let out = Command::new("vmmap")
        .args(["--summary", &pid.to_string()])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("Physical footprint: ") {
            return parse_footprint(rest.trim());
        }
    }
    None
}

/// "15.5G" / "812.3M" / "65536K" -> bytes.
fn parse_footprint(text: &str) -> Option<u64> {
    let (num, unit) = text.split_at(text.find(|c: char| !c.is_ascii_digit() && c != '.')?);
    let v: f64 = num.parse().ok()?;
    let mult = match unit.trim().to_ascii_uppercase().as_str() {
        "B" | "" => 1.0,
        "K" | "KB" => 1024.0,
        "M" | "MB" => 1024.0 * 1024.0,
        "G" | "GB" => 1024.0 * 1024.0 * 1024.0,
        "T" | "TB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((v * mult) as u64)
}

/// Stop every running server. Returns the models stopped.
pub fn stop_all() -> Vec<String> {
    running()
        .iter()
        .map(|s| s.model.clone())
        .collect::<Vec<_>>()
        .iter()
        .filter_map(|m| stop(m).ok().map(|_| m.clone()))
        .collect()
}

fn send_signal(pid: u32, sig: i32) {
    unsafe {
        libc::kill(pid as i32, sig);
    }
}

/// For `status`: list each running server with uptime.
pub fn uptime_since(started: SystemTime) -> String {
    let secs = started.elapsed().unwrap_or(Duration::ZERO).as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{}s", secs / 60, secs % 60)
    } else {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    }
}
