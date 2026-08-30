# Lemur

Local web control panel for [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` and native vLLM NVFP4 models. Discover models on your machine, assign them to GPUs, run multiple servers at once, and chat with a running instance.

## Requirements

- Ubuntu (or Linux) with NVIDIA GPUs
- Python 3.10+
- Built `llama-server` binary (auto-detected from `PATH` or common `~/llama.cpp/build*/bin/` locations; set explicitly in Settings)
- `nvidia-smi` in PATH

## Quick start

```bash
cd /path/to/llm-hub
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.main:app --host 127.0.0.1 --port 9000
```

Open [http://127.0.0.1:9000](http://127.0.0.1:9000).

## Desktop icon

Install a launcher that always runs from this repo (live code via `--reload`):

```bash
./scripts/install-desktop.sh
```

Double-click **Lemur** on the desktop or find it in the app menu. Opens a native GTK/WebKit window (not a browser tab). Logs: `~/.config/lemur/server.log` and `~/.config/lemur/launch.log`.

Sharded GGUFs (e.g. `*-00001-of-00003.gguf`) are grouped as one model. HuggingFace text model dirs are discovered and converted to GGUF on first launch (cached under `~/.config/lemur/converted/`).

## Codex integration

Lemur exposes OpenAI-compatible `/v1/chat/completions` and `/v1/responses` at `http://127.0.0.1:9000/v1`, and writes a Codex profile at `~/.codex/lemur.config.toml` (your main `config.toml` is left alone).

1. Start a model in Lemur
2. Run Codex:

```bash
codex --profile lemur -m Qwen3-Coder-Next-UD-IQ3_XXS
```

Or click **copy codex** / **codex default** on a running server card.

## Features

- **Model discovery** — scans `$HOME` for `.gguf` files and HF text model directories (excludes caches, vocab stubs, mmproj, ComfyUI trees)
- **Favorite presets** — save a running model's GPU, context, spill, MTP, vision, and GPU-layer settings for one-click launch; favorites are the primary boot view
- **Native NVFP4** — compressed Hugging Face NVFP4 checkpoints run through vLLM without conversion, with Blackwell FP4 kernels, FP8 KV cache, vision, tools, reasoning, and optional MTP
- **NVFP4 GGUF** — Qwen3.8 NVFP4 GGUFs run through a recent `llama-server`; Lemur applies the publisher's MTP, sampling, vision, and long-context profile
- **Sharded GGUFs** — multi-part models appear as one entry; loads via shard `00001`
- **Codex** — `/v1` proxy + auto sync of `lemur` Codex profile
- **Multi-GPU** — pick GPU per server via `CUDA_VISIBLE_DEVICES`
- **Parameters** — context length, port, GPU layers (`-ngl`)
- **Multiple servers** — run several models simultaneously on different GPUs/ports
- **LAN toggle** — restart loaded models on the same ports for access from trusted devices on your local network
- **Chat** — simple streaming chat against a running server (via API proxy)

## Qwen3.8 profiles for this machine

The four saved favorites show the main ways to run Qwen3.8 on the RTX 5090.
Choose a profile by workload, not only by peak single-request speed.

| Favorite | Backend | Context | MTP | Measured output speed | Best use | Main tradeoff |
|---|---|---:|---:|---:|---|---|
| `Qwen3.8-27B-NVFP4` | vLLM | 128K | N=3 | 113 tok/s single; 413 tok/s with four requests | Concurrent agents, API users, and parallel chat requests | Native checkpoint uses almost all 32 GiB of VRAM |
| `Qwen3.8-27B-Uncensored-NVFP4-MTP` | llama.cpp GGUF | 128K | N=3 | 118 tok/s single; 103–110 tok/s total when queued | Fast uncensored single-agent chat or Codex work | MTP uses one server slot, so concurrent requests wait |
| `Qwen3.8-27B-Uncensored-NVFP4-MTP` | llama.cpp GGUF | 256K | N=3 | Not measured | Very large repositories, documents, or conversation history | More KV-cache use and lower long-context speed |
| `Qwen3.8-27B-UD-Q6_K_XL` | llama.cpp GGUF | 128K | N=2 | Not measured | Higher-weight-precision work where quality is more important than memory use | Larger weights and one MTP server slot |

The native NVFP4 favorite is the throughput profile. vLLM uses continuous
batching, FP8 KV cache, prefix caching, 94% GPU memory, four active sequences,
and a 4,096-token scheduler batch. In the 2026-08-22 test, four concurrent
256-token requests produced about 413 output tokens/s after warmup. One request
produced about 113 tokens/s. Eight incoming requests did not improve total
throughput because four requests waited for an active sequence.

| Native vLLM request load | Total output speed | Completion time for 256 tokens per request |
|---:|---:|---:|
| 1 | 113 tok/s | 2.26 s |
| 2 concurrent | 209 tok/s | 2.45 s total |
| 4 concurrent | 413 tok/s | 2.48 s total |
| 8 incoming, 4 active | 421 tok/s | 4.86 s total |

These are short, warm-cache generation tests. They measure serving throughput,
not model quality or retrieval accuracy. Prompt length, generated length, MTP
acceptance, and current context use can change the result. The two profiles
marked **Not measured** must not use the older Qwen3.6 values as substitutes.

The 128K context value is a per-request limit, not a promise that four requests
can each fill 128K at the same time. The GPU KV cache can hold one nearly full
128K request, or several smaller agent contexts. Use the native vLLM favorite
for multi-agent or multi-user work. Use a GGUF favorite when one active request,
GGUF portability, an uncensored model, a 256K limit, or Q6 weights are more
important than concurrent throughput.

For the uncensored NVFP4 GGUF, put `mmproj-BF16.gguf` beside the model when
vision is required. Lemur uses the embedded MTP head, `draft_n=3`,
`--spec-draft-p-split 0.2`, and Q8 target/draft KV caches at 128K or more.
Both native NVFP4 and NVFP4 GGUF execution require a recent Blackwell-capable
runtime and an RTX 50-series or RTX PRO Blackwell GPU.

## Qwen3.6 profile for this machine

Measured 2026-07-16 with llama.cpp b9967, 8K context, and
`Qwen3.6-27B-UD-Q6_K_XL_MTP.gguf`:

| Placement | MTP | Generation | Memory |
|---|---:|---:|---:|
| RTX 5090 | off | 56.8 tok/s | 25.3 GiB VRAM |
| RTX 5090 | N=2 | 112.0 tok/s | 26.7 GiB VRAM |
| RTX 5090 + RAM fallback | N=2 | 112.5 tok/s | 26.7 GiB VRAM; no RAM spill |
| RTX 5090 + RTX 3060 | off | 25.5 tok/s | 17.8 + 7.9 GiB VRAM |
| RTX 5090 + RTX 3060 | N=2 | 43.8 tok/s | 18.7 + 8.5 GiB VRAM |
| Both GPUs + RAM fallback | N=2 | 47.4 tok/s | 19.5 + 7.7 GiB VRAM |
| RTX 3060 + RAM fallback | N=2 | 6.23 tok/s | 7.6 GiB VRAM + ~20 GiB RSS |

Use GPU 0 (RTX 5090), GPU-only placement, MTP N=2, and RAM fallback only
when extra capacity is required. The RTX 3060 has no peer-to-peer path to the
5090 and runs at PCIe x4, so adding it reduces token generation speed. Full
offload to the 3060 is rejected before CUDA starts because the model cannot fit.

Long-context results use MTP N=2 and Q8 target/draft KV caches. The prompt was
filled to 122,880 tokens at 128K and 245,760 tokens at 256K:

| Context | Fastest placement | Prompt fill | Near-full generation | Peak VRAM |
|---|---|---:|---:|---:|
| 128K | RTX 5090 + RAM fallback | 1,660 tok/s (74.0 s) | 69.4 tok/s | 30.0 GiB |
| 256K | RTX 5090 + RTX 3060 + RAM fallback | 636 tok/s (386.4 s) | 22.6 tok/s | 29.1 + 7.9 GiB |

At 256K, a single-GPU Q5 cache looked faster with an empty context but fell to
95 prompt tok/s once the context was populated. The two-GPU Q8 profile is the
practical winner. Reusing the 245K-token prefix reduced the next prompt pass to
a 516-token suffix evaluation that took 1.6 seconds. In the launch form, choose
**RAM** spill for 128K and **both** for 256K. Lemur applies Q8 KV caches
automatically for large Qwen3.5/3.6 models at 128K+.

For production stability, Lemur uses a separate CUDA 12.8 build and disables
CUDA graphs and PDL for long-context launches. At 128K it uses `-ub 256` so the
Q8 cache, MTP context, and full model offload fit beside the desktop; 256K keeps
automatic fallback. This follows an observed RTX 5090 Xid 8 watchdog timeout
under a variable-shape Codex request; the peak benchmark rates above were
measured before the safety change.

For Qwen3.5/3.6 metadata, Lemur uses Qwen's precise-coding sampling defaults
(`temperature=0.6`, `top_k=20`, `top_p=0.95`, `min_p=0`) when a request does not
override them. MTP also forces flash attention and one server slot, as required
by the [model publisher's llama.cpp recipe](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF#to-run-in-llamacpp).

## Settings

Stored in `~/.config/lemur/state.json`. Configure from the gear icon:

- `llama-server` binary path
- `diffusion-visual` binary (`llama-diffusion-gemma-visual-server` from Unsloth /
  [llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423)) —
  auto-used for `diffusion-gemma` GGUFs such as DiffusionGemma
- Scan root (default: home directory)
- Minimum model size filter (default: 50 MB)
- Default context length

### DiffusionGemma

Standard `llama-server` cannot load `architecture=diffusion-gemma`. When Lemur
detects a diffusion GGUF it starts a small OpenAI-compatible shim that drives
Unsloth’s visual decoder (same path as Unsloth Studio). Chat and `/v1` keep
working as usual.

### Dream / Dream-Coder

Dream is a discrete diffusion LM that uses Hugging Face Transformers
(`diffusion_generate`), not `llama-server`. When Lemur detects a Dream HF
folder (e.g. `Dream-org/Dream-Coder-v0-Instruct-7B`) it starts a Transformers
OpenAI shim — no GGUF conversion. Prefer a ≥16 GB GPU for the BF16 weights.
Lemur spill / MTP / ngl settings are ignored; **ctx** maps to `max_new_tokens`
(capped at 2048).

Requires a Python with `torch` + `transformers==4.46.x` (Dream’s tested stack;
5.x breaks RoPE). Lemur prefers `~/.config/lemur/dream-venv` when present.
Create it once with:

```bash
python3 -m venv --system-site-packages ~/.config/lemur/dream-venv
~/.config/lemur/dream-venv/bin/pip install 'transformers==4.46.2'
```

(Or set **dream-python** / `DREAM_PYTHON` to any interpreter that has CUDA torch
and transformers 4.46.)

## Local network access

Click **lan:off** above the running-model list. Lemur restarts loaded models
bound to all network interfaces; the button changes to **lan:on**. Connect from
another device using `http://<this-machine's-LAN-IP>:<model-port>/v1`.

The model endpoints have no authentication. Use this only on a trusted network,
and allow the model ports through the host firewall if it is enabled. Click
**lan:on** to return the models to localhost-only access.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/models` | List discovered models |
| `POST /api/models/refresh` | Force rescan |
| `GET /api/gpus` | List GPUs |
| `GET /api/servers` | Running servers |
| `POST /api/servers` | Start server |
| `DELETE /api/servers/{id}` | Stop server |
| `GET /api/servers/{id}/logs` | Log tail |
| `GET/POST /api/favorites` | List presets / save a running server preset |
| `POST /api/favorites/{id}/start` | Start a saved preset |
| `DELETE /api/favorites/{id}` | Delete a saved preset |
| `GET/PUT /api/settings` | Settings |
| `POST /api/chat` | Chat proxy (streaming) |
