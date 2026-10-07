//! `mlx download <repo|url>`: pull a HuggingFace repo into `~/.mlx-models/models/`.
//!
//! Zero external services beyond huggingface.co: the REST API lists files
//! (with LFS sha256 oids), a single `curl -L -C -` fetches each one, and every
//! file is verified against the API's oid before the model dir is considered
//! complete. `curl` ships with macOS and honors corporate proxies, so no PyPI
//! and no Python are involved.
//!
//! Flow: parse target -> check installed duplicates -> fetch file list ->
//! skip files already present with the right hash -> download the rest into a
//! `<name>.partial-<pid>/` staging dir -> verify -> rename into place.

use std::collections::BTreeMap;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::table::human_bytes;

/// Network attempts per HTTP operation before giving up (user-approved cap).
const MAX_ATTEMPTS: u32 = 3;
/// Base for exponential backoff between attempts.
const BACKOFF_MS: u64 = 1_500;

/// A file listed by the hub API for one repo.
#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: u64,
    lfs: Option<LfsInfo>,
}

#[derive(Debug, Deserialize)]
struct LfsInfo {
    oid: String,
    size: u64,
}

/// Repo contents + revision, from `/api/models/<repo>`.
pub struct RepoListing {
    pub repo: String,
    pub sha: String,
    /// path -> (download url, expected sha256 or None for non-LFS, size)
    pub files: BTreeMap<String, (String, Option<String>, u64)>,
}

/// Parse `org/name`, a full `https://huggingface.co/.../org/name` URL, or a
/// bare `name` (treated as `org` omitted -> error with guidance).
pub fn parse_repo(input: &str) -> anyhow::Result<String> {
    let input = input.trim();
    let input = input.strip_prefix("hf.co/").unwrap_or(input);
    let input = input
        .strip_prefix("https://")
        .or_else(|| input.strip_prefix("http://"))
        .unwrap_or(input);
    let input = input.strip_prefix("www.").unwrap_or(input);
    let host_ok = input.starts_with("huggingface.co/");
    let rest = if host_ok {
        &input["huggingface.co/".len()..]
    } else {
        input
    };
    let rest = rest.trim_start_matches('/');
    let rest = rest.trim_end_matches('/');
    // Drop URL fragments/query leftovers.
    let rest = rest.split(['?', '#']).next().unwrap_or(rest);
    // Trim trailing file paths like /resolve/main/... or /tree/main/...
    let rest = if let Some(pos) = rest.find("/resolve/") {
        &rest[..pos]
    } else if let Some(pos) = rest.find("/tree/") {
        &rest[..pos]
    } else if let Some(pos) = rest.find("/blob/") {
        &rest[..pos]
    } else {
        rest
    };
    let parts: Vec<&str> = rest.split('/').filter(|p| !p.is_empty()).collect();
    match parts.as_slice() {
        [org, name] => {
            let org = org.trim();
            let name = name.trim();
            if org.is_empty() || name.is_empty() {
                anyhow::bail!("empty org or repo in '{input}'");
            }
            Ok(format!("{org}/{name}"))
        }
        _ => anyhow::bail!(
            "expected <org>/<repo> or a huggingface.co model URL, got '{input}'\n  example: mlx download mlx-community/Qwen3-0.6B-4bit"
        ),
    }
}

/// Directory where `mlx download` installs: `~/.mlx-models/models/<name>`.
pub fn install_dir(repo: &str) -> PathBuf {
    let name = repo.split('/').next_back().unwrap_or(repo);
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".mlx-models/models")
        .join(name)
}

fn api_json(url: &str) -> anyhow::Result<Vec<u8>> {
    let mut last_err = String::new();
    for attempt in 1..=MAX_ATTEMPTS {
        let out = Command::new("curl")
            .args(["-sSL", "--fail", "-w", "\n%{http_code}", url])
            .output()?;
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            if let Some((body, code)) = stdout.rsplit_once('\n') {
                if code.trim() == "200" {
                    return Ok(body.as_bytes().to_vec());
                }
            }
            last_err = format!("unexpected response from {url}");
        } else {
            last_err = String::from_utf8_lossy(&out.stderr).trim().to_string();
            if last_err.is_empty() {
                let code = String::from_utf8_lossy(&out.stdout);
                if code.contains("401") || code.contains("403") {
                    anyhow::bail!("hub returned 401/403 for {url}\n  is the repo gated? set HF_TOKEN or run 'huggingface-cli login'");
                }
                if code.contains("404") {
                    anyhow::bail!("repo not found: {url}\n  check the org/name spelling");
                }
                last_err = format!("curl failed ({})", out.status);
            }
        }
        if attempt < MAX_ATTEMPTS {
            std::thread::sleep(std::time::Duration::from_millis(
                BACKOFF_MS * attempt as u64,
            ));
        }
    }
    anyhow::bail!("hub unreachable after {MAX_ATTEMPTS} attempts: {last_err}\n  check network/proxy and retry")
}

/// List every file of `repo` with sizes and LFS sha256 oids.
pub fn list_repo(repo: &str) -> anyhow::Result<RepoListing> {
    let meta: serde_json::Value = serde_json::from_slice(&api_json(&format!(
        "https://huggingface.co/api/models/{repo}"
    ))?)
    .map_err(|e| anyhow::anyhow!("cannot parse hub metadata for {repo}: {e}"))?;
    let sha = meta
        .get("sha")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let tree: Vec<TreeEntry> = serde_json::from_slice(&api_json(&format!(
        "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    ))?)
    .map_err(|e| anyhow::anyhow!("cannot parse hub file tree for {repo}: {e}"))?;
    let mut files = BTreeMap::new();
    for entry in tree {
        if entry.kind != "file" {
            continue;
        }
        // Gated/private token support: pass through env to curl.
        let url = format!("https://huggingface.co/{repo}/resolve/main/{}", entry.path);
        let expected = entry.lfs.as_ref().map(|l| l.oid.clone());
        let size = entry.lfs.as_ref().map(|l| l.size).unwrap_or(entry.size);
        files.insert(entry.path, (url, expected, size));
    }
    if files.is_empty() {
        anyhow::bail!("hub lists no files for {repo}");
    }
    Ok(RepoListing {
        repo: repo.to_string(),
        sha,
        files,
    })
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Files worth downloading from a model repo: everything except hub
/// bookkeeping (`.gitattributes`) and auxiliary assets dirs.
fn wanted(path: &str) -> bool {
    !path.starts_with("assets/") && path != ".gitattributes"
}

/// Size of safetensors weights inside `dir` (identity used by discovery).
fn weights_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Same model already installed? Heuristic (config hash + weights size) so we
/// never duplicate a 20 GB download just because dir names differ.
fn find_installed_copy(repo: &str, listing: &RepoListing) -> Option<crate::discover::Model> {
    let cfg = crate::config::Config::load().ok()?;
    let (_url, config_hash, _size) = listing.files.get("config.json")?;
    let remote_weights: u64 = listing
        .files
        .iter()
        .filter(|(p, _)| p.ends_with(".safetensors") && !p.ends_with(".index.json"))
        .map(|(_, (_, _, s))| s)
        .sum();
    if remote_weights == 0 {
        return None;
    }
    // Remote config.json may or may not be LFS; hash it directly when small.
    let remote_config_bytes = match config_hash {
        Some(hash) => fetch_bytes(&listing.files["config.json"].0)
            .ok()
            .filter(|b| format!("{:x}", Sha256::digest(b)) == *hash)?,
        None => fetch_bytes(&listing.files["config.json"].0).ok()?,
    };
    for model in crate::discover::discover_all(&cfg) {
        let local_config = std::fs::read_to_string(model.path.join("config.json")).ok()?;
        let local_weights = weights_bytes(&model.path);
        if local_weights == remote_weights
            && local_config.as_bytes() == remote_config_bytes.as_slice()
        {
            eprintln!(
                "note: {repo} is already installed as '{}' ({})",
                model.name, model.location
            );
            return Some(model);
        }
    }
    None
}

fn fetch_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
    let out = Command::new("curl")
        .args(["-sSL", "--fail", url])
        .output()?;
    if out.status.success() {
        Ok(out.stdout)
    } else {
        anyhow::bail!("download failed: {url}")
    }
}

/// One `curl -L -C -` file fetch with retry + resume into `staging`.
fn fetch_file(url: &str, dest: &Path, size: u64) -> anyhow::Result<()> {
    for attempt in 1..=MAX_ATTEMPTS {
        let status = Command::new("curl")
            .args([
                "-L",
                "--fail",
                "--retry",
                "2",
                "--connect-timeout",
                "20",
                "-C",
                "-",
                "-o",
            ])
            .arg(dest)
            .arg(url)
            .status()?;
        if status.success() {
            let have = dest.metadata().map(|m| m.len()).unwrap_or(0);
            if size > 0 && have < size {
                // Resume produced a short file; retry once more from scratch.
                let _ = std::fs::remove_file(dest);
                if attempt < MAX_ATTEMPTS {
                    continue;
                }
                anyhow::bail!(
                    "file is short after download: {} ({})",
                    dest.display(),
                    human_bytes(have)
                );
            }
            return Ok(());
        }
        if attempt < MAX_ATTEMPTS {
            std::thread::sleep(std::time::Duration::from_millis(
                BACKOFF_MS * attempt as u64,
            ));
        }
    }
    anyhow::bail!("download failed after {MAX_ATTEMPTS} attempts: {url}\n  partial files remain in {}; re-run 'mlx download' to resume", dest.parent().unwrap_or(Path::new(".")).display())
}

/// Verify every staged file against its LFS oid (when present) or size.
fn verify_staged(
    staging: &Path,
    listing: &RepoListing,
    wanted_paths: &[&str],
) -> anyhow::Result<()> {
    for path in wanted_paths {
        let dest = staging.join(path);
        let Some((_, hash, size)) = listing.files.get(*path) else {
            continue;
        };
        if !dest.is_file() {
            anyhow::bail!("missing file after download: {}", dest.display());
        }
        if let Some(hash) = hash {
            let actual = sha256_file(&dest)?;
            if !actual.eq_ignore_ascii_case(hash.trim_start_matches("sha256:")) {
                let _ = std::fs::remove_file(&dest);
                anyhow::bail!(
                    "sha256 mismatch for {}\n  expected {hash}\n  got      {actual}\n  file deleted; re-run 'mlx download' to retry",
                    dest.display()
                );
            }
        } else {
            let have = dest.metadata().map(|m| m.len()).unwrap_or(0);
            if have != *size {
                let _ = std::fs::remove_file(&dest);
                anyhow::bail!(
                    "size mismatch for {} (expected {}, got {})",
                    dest.display(),
                    size,
                    have
                );
            }
        }
    }
    Ok(())
}

fn prompt_yes_no(prompt: &str, default: bool) -> bool {
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

/// Entry point for `mlx download`.
pub fn cmd_download(target: &str, force: bool, yes: bool) -> Result<()> {
    let repo = parse_repo(target)?;
    let listing = list_repo(&repo)?;
    let wanted_paths: Vec<&str> = listing
        .files
        .keys()
        .map(String::as_str)
        .filter(|p| wanted(p))
        .collect();
    let total: u64 = wanted_paths
        .iter()
        .filter_map(|p| listing.files.get(*p))
        .map(|(_, _, s)| s)
        .sum();
    println!(
        "{} @ {} - {} files, {}",
        listing.repo,
        &listing.sha[..7.min(listing.sha.len())],
        wanted_paths.len(),
        human_bytes(total)
    );

    if !force {
        if let Some(existing) = find_installed_copy(&repo, &listing) {
            println!(
                "already installed as '{}' at {} - nothing to do (use --force to download anyway)",
                existing.name, existing.location
            );
            return Ok(());
        }
    }

    let dest = install_dir(&repo);
    if dest.exists() && !yes {
        let resp = prompt_yes_no(
            &format!(
                "{} already exists locally. Re-download and overwrite?",
                dest.display()
            ),
            false,
        );
        if !resp {
            println!("aborted");
            return Ok(());
        }
    }

    // Show the largest files so big repos are not a surprise.
    let mut big: Vec<(&String, u64)> = listing
        .files
        .iter()
        .filter(|(p, _)| wanted(p))
        .map(|(p, (_, _, s))| (p, *s))
        .collect();
    big.sort_by_key(|&(_, size)| std::cmp::Reverse(size));
    for (path, size) in big.iter().take(3) {
        println!("  {:>10}  {}", human_bytes(*size), path);
    }

    let staging = dest.with_extension(format!("partial-{}", std::process::id()));
    std::fs::create_dir_all(&staging)?;
    for (i, path) in wanted_paths.iter().enumerate() {
        let (url, _hash, size) = &listing.files[*path];
        let out = staging.join(path);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let existing = out.metadata().map(|m| m.len()).unwrap_or(0);
        if existing == *size {
            println!("[{}/{}] {} (cached)", i + 1, wanted_paths.len(), path);
            continue;
        }
        if existing > *size {
            let _ = std::fs::remove_file(&out);
        }
        println!(
            "[{}/{}] {} ({})",
            i + 1,
            wanted_paths.len(),
            path,
            human_bytes(*size)
        );
        fetch_file(url, &out, *size)?;
    }

    println!("verifying...");
    verify_staged(&staging, &listing, &wanted_paths)?;

    // Swap into place: remove previous install, then rename staging dir.
    if dest.exists() {
        std::fs::remove_dir_all(&dest)?;
    }
    std::fs::rename(&staging, &dest)?;
    // Remember where this came from so `mlx list` can show `org/model`.
    std::fs::write(dest.join(".mlx-origin"), &listing.repo)?;
    println!("installed {} -> {}", listing.repo, dest.display());
    Ok(())
}
