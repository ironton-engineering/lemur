# Lemur

Lemur is a local desktop control panel for [llama.cpp](https://github.com/ggml-org/llama.cpp) and optional [vLLM](https://github.com/vllm-project/vllm). It finds local models, assigns them to NVIDIA GPUs, starts model servers, and provides a small chat interface.

Lemur does not include model files and does not send telemetry.

## Supported Systems

- Ubuntu 22.04, Ubuntu 24.04, or Ubuntu 26.04
- x86-64
- NVIDIA GeForce RTX 30-, 40-, or 50-series GPU
- NVIDIA driver 570.26 or newer on Ubuntu 22.04 and 24.04
- NVIDIA driver 580 or newer on Ubuntu 26.04
- At least 15 GiB of free disk space

Lemur does not support CPU-only inference. The NVIDIA driver must work before installation. The installer does not install or change the driver.

## Install

```bash
curl -fsSL https://github.com/ironton-engineering/lemur/releases/download/v0.1.0/install.sh | bash
```

Pre-releases use a versioned installer URL. After the first stable release, you can use `releases/latest/download/install.sh` instead.

The installer shows each Ubuntu package command before it requests `sudo`. It can install CUDA Toolkit 12.8 on Ubuntu 22.04 and 24.04 or CUDA Toolkit 13.3 on Ubuntu 26.04. It does not install the NVIDIA driver. It builds a pinned llama.cpp release for the GPUs in the machine. The build can take several minutes.

To inspect the installer before execution:

```bash
curl -fsSLo install-lemur.sh https://github.com/ironton-engineering/lemur/releases/download/v0.1.0/install.sh
less install-lemur.sh
bash install-lemur.sh
```

Open **Lemur** from the Ubuntu application menu, or run `lemur`. If `~/.local/bin` is not in `PATH`, run `~/.local/bin/lemur`.

## Optional vLLM

vLLM is not part of the default install. Install it later with:

```bash
lemur install-vllm
```

You can also install it with Lemur:

```bash
curl -fsSL https://github.com/ironton-engineering/lemur/releases/download/v0.1.0/install.sh | bash -s -- --with-vllm
```

The pinned vLLM wheel uses CUDA 12.9 and needs NVIDIA driver 575.51.03 or newer. A failed optional vLLM install does not remove the llama.cpp backend.

## First Use

Lemur scans the home directory for GGUF files and supported Hugging Face model directories. The scan excludes common cache, source, and application directories.

For llama.cpp, use a GGUF model from a source that you trust. Put all parts of a sharded GGUF in one directory. Put a compatible `mmproj*.gguf` file beside a vision model.

For native vLLM NVFP4 models, install the optional vLLM backend first. Lemur does not download a model for you. You are responsible for the license and use terms of each model.

Then:

1. Open the full model list.
2. Select a model.
3. Select a GPU and context size.
4. Start the model.
5. Open chat or use the displayed API address.

## Main Commands

```text
lemur                 Start the desktop application.
lemur doctor          Check the system, GPU, driver, backend, and paths.
lemur update          Install the latest checked release.
lemur rollback        Activate the prior working release.
lemur install-vllm    Install the optional vLLM backend.
lemur uninstall       Remove Lemur but keep settings and logs.
```

To remove settings and logs too, run `lemur uninstall --remove-user-data`. The uninstall command shows each path before removal. It never removes model files.

## Files and Logs

- Application releases: `~/.local/share/llm-hub/releases/`
- Backends: `~/.local/share/llm-hub/backends/`
- Active release: `~/.local/share/llm-hub/current`
- Settings and favorites: `~/.config/lemur/state.json`
- Server log: `~/.config/lemur/server.log`
- Launch log: `~/.config/lemur/launch.log`

See [the install layout](docs/INSTALL_LAYOUT.md) for the full file ownership rules.

## Features

- Local GGUF and native NVFP4 model discovery
- Sharded GGUF grouping
- NVIDIA GPU and VRAM display
- Single-GPU and multi-GPU llama.cpp placement
- RAM fallback controls
- Multiple model servers
- Saved launch favorites
- Streaming chat
- OpenAI-compatible chat and responses routes
- Codex profile creation
- Optional local-network access

## Codex

Lemur writes a separate Codex profile at `~/.codex/lemur.config.toml`. It does not change the main Codex configuration file.

After a model starts, use the command shown by Lemur. A typical command is:

```bash
codex --profile lemur -m MODEL_ID
```

## Local-Network Access

Lemur binds to `127.0.0.1` by default. The LAN control can restart active model servers on `0.0.0.0`.

LAN model endpoints have no authentication. Use LAN mode only on a trusted network. Do not expose these ports to the public internet.

## Diagnose a Problem

Run `lemur doctor`. Then inspect:

```bash
tail -n 200 ~/.config/lemur/launch.log
tail -n 200 ~/.config/lemur/server.log
```

If an update fails, the current release stays active. If a new release does not work, run `lemur rollback`.

## API

The Lemur control service listens at `http://127.0.0.1:9000`.

| Endpoint | Purpose |
|---|---|
| `GET /api/models` | List found models |
| `POST /api/models/refresh` | Start a new model scan |
| `GET /api/gpus` | List supported GPUs |
| `GET /api/servers` | List managed model servers |
| `POST /api/servers` | Start a model server |
| `DELETE /api/servers/{id}` | Stop a model server |
| `GET /api/servers/{id}/logs` | Read recent model-server logs |
| `GET/POST /api/favorites` | List or create favorites |
| `GET/PUT /api/settings` | Read or change settings |
| `POST /api/chat` | Use the streaming chat proxy |

Model servers provide OpenAI-compatible `/v1/chat/completions` routes. Lemur also provides `/v1/responses` through its local proxy.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.lock
.venv/bin/python -m pytest -q
.venv/bin/uvicorn server.main:app --host 127.0.0.1 --port 9000
```

To test the installer from a checkout on a supported machine:

```bash
LEMUR_SOURCE_DIR="$PWD" ./install.sh
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [the release guide](docs/RELEASING.md).

## Security, Privacy, and Licenses

Read [SECURITY.md](SECURITY.md) before you use LAN mode or untrusted models. Read [PRIVACY.md](PRIVACY.md) for the local data policy.

Lemur uses the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). llama.cpp uses the MIT License. vLLM uses the Apache License 2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for pinned versions and source links.
