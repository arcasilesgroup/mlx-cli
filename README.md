# mlx-cli

Manage local [MLX](https://github.com/ml-explore/mlx) model servers on Apple Silicon.

A single static Rust binary to `list`, `start`, `stop` and check the `status` of
models served with `mlx-vlm` / `mlx-lm`. Nothing is hardcoded: it discovers
whatever models you have on disk.

## Install

```sh
cargo install --path .
```

Requirements: Rust 1.70+, macOS on Apple Silicon, and a Python environment with
`mlx-vlm` or `mlx-lm` installed (see configuration below).

## Usage

```sh
mlx list                                  # discovered models + running state + URL
mlx status                                # per-model state + system memory
mlx start                                 # interactive picker (shows the list first)
mlx start mlx-community/Qwen3-8B          # direct start
mlx stop                                  # interactive picker among running
mlx stop <model>                          # direct stop
mlx stop-all                              # stop every running server
mlx set-port <model> <port>               # pin a model's port (persists)
mlx download <org/name|hf-url>            # download a model from HuggingFace
mlx delete                                # interactive picker among installed
mlx delete <model>                        # delete a model from disk
```

`start` reads your system RAM and the model's size on disk: if loading it
would leave less than ~8 GiB free, it asks for confirmation first. Several
models can run at the same time if memory allows; each gets the first free
port.

## Vision vs text models

`mlx start` picks the serving module automatically: models with an image
processor or `vision_config` in `config.json` are served by
`mlx_vlm.server`, text-only models by `mlx_lm.server`. It probes which of
the two packages your configured python can actually import and refuses to
spawn with an install command (`pip install mlx-vlm` / `pip install mlx-lm`)
when the required one is missing - so a text-only setup never needs
`mlx-vlm` installed.


## Downloading and deleting models

`mlx download` pulls any HuggingFace model repo into the standard location
`~/.mlx-models/models/<repo-name>/` - no Python, no PyPI, no `huggingface-cli`
needed. It talks to the hub REST API and uses the `curl` that ships with macOS,
so it works behind corporate proxies. Both forms are accepted:

```sh
mlx download mlx-community/Qwen3-0.6B-4bit
mlx download https://huggingface.co/mlx-community/Qwen3-0.6B-4bit
```

Integrity: every LFS file is SHA-256-verified against the hub API's `oid`
before the model counts as installed; files already on disk with the right
size are skipped, and interrupted runs resume on re-run. If the repo is
already installed under a different folder name (same `config.json` + same
weights size), it refuses to duplicate it - pass `--force` to download anyway.
Gated repos work by exporting `HF_TOKEN`.

`mlx delete` removes an installed model and frees its disk space. Without an
argument it shows the interactive picker (same as `start`/`stop`); with a
model name it deletes directly after a confirmation prompt (`--yes` skips
it). Deleting a running model is refused - stop it first. HF-cache copies
remove the whole `models--org--name` directory so no blobs are left behind.

## How models are discovered

Two places are scanned by default: `~/.cache/huggingface/hub` (where
`huggingface-cli` / `mlx-vlm` download) and `~/.mlx-models/models` (the
convention of many local MLX setups). In the HF cache it resolves each
`models--org--name/snapshots/<hash>` directory that contains `config.json` +
`*.safetensors`; elsewhere any directory with those two files counts. Size
comes from disk; parameter count from `config.json` when available.

To add your own directories, see the config below.

## Server setup (automatic)

`mlx start` needs a python that can serve MLX models. On first use it looks
for one automatically - `~/.mlx-venv/bin/python`, then `$VIRTUAL_ENV`, then
`python3`/`python`, verifying that `mlx_vlm` or `mlx_lm` actually imports -
and saves the winner to the config file. From then on it just works; if
detection fails, the error tells you how to configure `server.command`.

## Configuration

Optional file at `~/Library/Application Support/mlx-cli/config.toml` (Linux:
`~/.config/mlx-cli/config.toml`):

```toml
# Extra directories to scan for models
model_dirs = ["/path/to/models"]

# First port tried when starting a server
port_base = 8080

# Command that launches servers: a python with mlx-vlm / mlx-lm installed.
# Accepts a string or an argv list.
[server]
command = "/path/to/python"
# or: command = ["uv", "run", "--project", "/path/to/venv", "python"]
```

The `[server]` block is written automatically on first `mlx start` (see
"Server setup" above); you only edit it if you want a different interpreter.
Discovery works with no config file at all.

### Fixed ports

By default a model gets the first free port from 8080 - which can drift
across stop/start cycles. If a client expects a stable port, pin it - by
editing the config or with:

```sh
mlx set-port Tiel-Coder-35B-A3B 8082
```

which writes the equivalent of:

```toml
[ports]
"Tiel-Coder-35B-A3B" = 8082
```

Keys match the short or full (`org/name`) model name. Pinned ports are used
exactly; unpinned models fall back to first-free-from-`port_base`.

### Per-model server args and environment

Every server module accepts tuning flags - `mlx_lm.server` / `mlx_vlm.server`
document theirs via `--help` (KV-cache quantization, batching width, context
caps, ...). mlx-cli stays agnostic: list extra args in `[start_args]` and
extra environment variables in `[start_env]`, keyed by model exactly like
`[ports]`:

```toml
[start_args]
"Tiel-Coder-35B-A3B" = [
  "--kv-bits", "4",
  "--kv-quant-scheme", "turboquant",
  "--max-num-seqs", "2",
  "--prefill-step-size", "4096",
  "--max-kv-size", "262144",
]

[start_env]
"Tiel-Coder-35B-A3B" = { KV_BITS = "4", APC_ENABLED = "1" }
```

mlx-cli does not interpret these - it appends the args to the server module
and injects the variables into its environment. An unknown flag makes the
server exit at startup; `mlx start` detects the death and prints the
server's own error so you can fix the config. A pre-existing server on the
pinned port is never mistaken for the newly spawned one.

`mlx list` shows the resulting OpenAI-compatible base URL per model
(`http://localhost:<port>/v1`). Point any OpenAI-speaking tool at it -
copilot BYOK, claude code, cursor, aider, opencode, omp, curl - nothing in
mlx-cli knows or cares which client you use.

### Model names (`org/model`)

`mlx download` records the hub id in `.mlx-origin` inside the model dir, and
`mlx list` / `mlx status` show it - so renamed installs still identify their
source. Models copied by hand keep their directory name unless you drop a
`.mlx-origin` file containing the `org/model` id.

### Memory (`mlx status`)

The MEM column is the process's **physical footprint** (`vmmap`), not RSS:
Metal maps model weights outside resident pages, so RSS shows ~40 MB while
the model really holds ~15 GB. Footprint is the number that matches system
memory pressure.

### Speculative decoding (MTP / draft models)

`mlx_lm.server` accepts `--draft-model` and `--num-draft-tokens`; pass them
through `[start_args]` for a text model - no special support needed:

```toml
[start_args]
"My-Model-4bit" = [
  "--draft-model", "/path/to/drafter-4bit",
  "--num-draft-tokens", "4",
]
```

`mlx_vlm.server` does not expose server-side draft flags yet; for vision
models speculative decoding is not available through this CLI.

### Other runtimes

mlx-cli spawns one process per model with `python -m <server module>` -
matching `mlx-vlm` / `mlx-lm`. Servers with a different shape (e.g. oMLX:
one daemon serving many models with load/unload via HTTP API) are separate
tools; mlx-cli deliberately does not wrap them.

## Design

- KISS / YAGNI: five commands, one config file, no daemon.
- Errors tell you what to do next, not just what broke.
- No unit tests by choice; the acceptance path is running the real commands.


## License

MIT
