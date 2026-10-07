//! Which python can serve what: detect `mlx_vlm` and `mlx_lm` per candidate
//! python, so `mlx start` can pick `mlx_vlm.server` for vision models and
//! `mlx_lm.server` for text models, and tell the user exactly what to install
//! when the needed package is missing.

use std::process::Command;

use crate::config::{Config, ServerConfig, StringOrVec};

/// A python interpreter and the MLX server packages it can import.
#[derive(Debug, Clone)]
pub struct Runtime {
    /// Full path or name of the interpreter (argv[0] for the server).
    pub python: String,
    /// True when `import mlx_vlm` succeeds (serves vision-language models).
    pub vlm: bool,
    /// True when `import mlx_lm` succeeds (serves text-only models).
    pub lm: bool,
}

impl Runtime {
    /// True if this interpreter can serve a model of the given kind.
    pub fn serves(&self, vision: bool) -> bool {
        if vision {
            self.vlm
        } else {
            self.vlm || self.lm
        }
    }
}

/// Candidate pythons, best first. The local venv used by the common
/// mlx setups is checked before the system interpreters.
fn candidates() -> Vec<String> {
    let home = dirs::home_dir().unwrap_or_default();
    let mut out = vec![home.join(".mlx-venv/bin/python").display().to_string()];
    if let Ok(path) = std::env::var("VIRTUAL_ENV") {
        out.push(format!("{path}/bin/python"));
    }
    out.push("python3".into());
    out.push("python".into());
    out
}

/// Import-check both packages in one interpreter call. `python` may be a
/// bare interpreter path or a wrapper script/shell command; wrappers that
/// only accept server args fail the direct probe, so on failure we retry
/// their shebang/`python`-like interpreter or fall back to a venv guess.
/// A missing interpreter yields `None`, a probed one always a `Runtime`.
fn probe(argv0: &str) -> Option<Runtime> {
    let code = "import importlib.util as u;\n\
def has(n):\n    try: return u.find_spec(n) is not None\n    except Exception: return False\n\
print('OK', int(has('mlx_vlm')), int(has('mlx_lm')))";
    if let Some(rt) = import_check(argv0, code) {
        return Some(rt);
    }
    // Wrapper-script fallbacks: its declared interpreter, then the local
    // venv python (the usual backing interpreter for mlx wrappers).
    let shebang = std::fs::read_to_string(argv0)
        .ok()
        .and_then(|s| s.lines().next()?.strip_prefix("#!").map(str::to_string));
    if let Some(interp) = shebang {
        if let Some(rt) = import_check(&interp, code) {
            return Some(rt);
        }
    }
    if let Some(home) = dirs::home_dir() {
        let venv = home.join(".mlx-venv/bin/python");
        if let Some(rt) = import_check(&venv.display().to_string(), code) {
            return Some(rt);
        }
    }
    None
}

/// Run `interp -c <code>` and parse "OK <vlm> <lm>".
fn import_check(interp: &str, code: &str) -> Option<Runtime> {
    let out = Command::new(interp).arg("-c").arg(code).output().ok()?;
    let line = String::from_utf8_lossy(&out.stdout);
    let line = line.lines().find(|l| l.starts_with("OK "))?;
    let mut it = line[3..].split_whitespace();
    let vlm = matches!(it.next()?.parse::<u8>().ok()?, 1);
    let lm = matches!(it.next()?.parse::<u8>().ok()?, 1);
    Some(Runtime {
        python: interp.to_string(),
        vlm,
        lm,
    })
}

/// Probe every candidate python, keeping candidate order.
pub fn detect() -> Vec<Runtime> {
    candidates().iter().filter_map(|p| probe(p)).collect()
}

/// Capabilities of the interpreter configured in `[server]`, if any.
pub fn configured_runtime(cfg: &Config) -> Option<Runtime> {
    let server = cfg.server.as_ref()?;
    let argv = server.command.to_vec();
    let (python, _) = argv.split_first()?;
    probe(python)
}

/// Actionable message when no interpreter can serve the model: exact install
/// commands plus the config.toml escape hatch.
pub fn install_guidance(missing: &str) -> String {
    let pkg = missing.replace('_', "-"); // mlx_vlm -> mlx-vlm
    format!(
        "no python on this machine can serve this model: `import {missing}` failed everywhere.\n  \
         Install it first - one command creates the venv with everything:\n    \
         python3 -m venv ~/.mlx-venv && ~/.mlx-venv/bin/pip install -U 'mlx[vlm]' mlx-vlm mlx-lm\n  \
         or, minimal for this model only (inside your venv/python):\n    \
         pip install {pkg}\n  \
         then re-run 'mlx start'. To use another interpreter, set in {}:\n    \
         [server]\n    command = \"/path/to/python\"",
        crate::config::config_path().display()
    )
}

/// Ensure `cfg.server` is set, detecting and persisting it on first use.
/// Mutates cfg in place; prints what it did. Prefers a python that can serve
/// both model kinds; ties keep candidate order.
pub fn ensure_server(cfg: &mut Config) -> anyhow::Result<()> {
    if cfg.server.is_some() {
        return Ok(());
    }
    let runtimes = detect();
    // Prefer an interpreter that serves both kinds; ties keep candidate
    // order. A python with neither package is not a server at all.
    let best = runtimes
        .iter()
        .rev()
        .max_by_key(|r| (r.vlm as u8) + (r.lm as u8))
        .filter(|r| r.vlm || r.lm)
        .map(|r| r.python.clone());
    let Some(best) = best else {
        anyhow::bail!(install_guidance("mlx_vlm"));
    };
    println!(
        "first run: detected server command '{}', saved to {}",
        best,
        crate::config::config_path().display()
    );
    cfg.server = Some(ServerConfig {
        command: StringOrVec::Many(vec![best]),
    });
    cfg.save()?;
    Ok(())
}
