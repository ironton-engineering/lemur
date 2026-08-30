import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from server import gguf_meta
from server import gpu as gpu_mod
from server import models as model_scan
from server.aliases import model_alias
from server.codex import codex_command
from server.config import CONFIG_DIR, CONVERTED_DIR, ensure_config_dir, get_settings
_HF_CONVERT_PY_CACHE: dict[str, str | None] = {}
_HF_CONVERT_PY_PROBE: dict[str, bool] = {}


def _python_torch_ok(python: str) -> bool:
    """True if interpreter can import torch (convert_hf_to_gguf.py needs it)."""
    if python in _HF_CONVERT_PY_PROBE:
        return _HF_CONVERT_PY_PROBE[python]
    try:
        r = subprocess.run(
            [python, "-c", "import torch"],
            capture_output=True,
            timeout=45,
        )
        _HF_CONVERT_PY_PROBE[python] = r.returncode == 0
        return _HF_CONVERT_PY_PROBE[python]
    except (OSError, subprocess.TimeoutExpired):
        _HF_CONVERT_PY_PROBE[python] = False
        return False


def resolve_hf_convert_python(configured: str | None = None) -> str | None:
    """Pick a Python that can import torch for HF → GGUF conversion.

    Use an explicit HF_CONVERT_PYTHON when the application environment does
    not contain torch.
    """
    key = (configured or "").strip() or os.environ.get("HF_CONVERT_PYTHON", "") or ""
    if key in _HF_CONVERT_PY_CACHE:
        return _HF_CONVERT_PY_CACHE[key]

    if key:
        p = str(Path(key).expanduser())
        if Path(p).is_file() and _python_torch_ok(p):
            _HF_CONVERT_PY_CACHE[key] = p
            return p

    cands: list[str] = []
    if configured and str(configured).strip():
        cands.append(str(Path(configured).expanduser()))
    for env_key in ("HF_CONVERT_PYTHON",):
        env = os.environ.get(env_key)
        if env:
            cands.append(str(Path(env).expanduser()))
    cands.extend([sys.executable, "python3"])

    seen: set[str] = set()
    for raw in cands:
        p = str(Path(raw).expanduser()) if raw not in ("python3", "python") else raw
        if p in seen:
            continue
        seen.add(p)
        if raw not in ("python3", "python") and not Path(p).is_file():
            continue
        if _python_torch_ok(p):
            _HF_CONVERT_PY_CACHE[key] = p
            return p

    _HF_CONVERT_PY_CACHE[key] = None
    return None

# Overflow when primary GPU VRAM is not enough.
# none = primary only; ram = CPU/RAM; gpu = other GPUs; both = other GPUs + RAM
SPILL_MODES = ("none", "ram", "gpu", "both")
HF_CONVERTER_OUTTYPES = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}
LAST_SERVER_LOG = CONFIG_DIR / "last-llama-server.log"


class PlacementError(RuntimeError):
    """The requested CUDA placement is known to be invalid or unsafe."""


class CapacityWarning(PlacementError):
    """The requested placement can exceed the currently free VRAM."""


@dataclass
class ServerInstance:
    id: str
    model_path: str
    model_name: str
    gpu: int
    port: int
    ctx: int
    ngl: int
    host: str
    alias: str = ""
    format: str = "gguf"
    spill: str = "none"
    mtp: bool = False
    mtp_draft_n: int = 2
    vision: bool = False
    vllm: bool = False
    devices: str = ""  # CUDA_VISIBLE_DEVICES actually used
    pid: int | None = None
    status: str = "starting"
    started_at: float = field(default_factory=time.time)
    logs: deque = field(default_factory=lambda: deque(maxlen=2000))
    launch_cmd: str = field(default="", repr=False)
    process: subprocess.Popen | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "alias": self.alias or model_alias(self.model_name),
            "gpu": self.gpu,
            "port": self.port,
            "ctx": self.ctx,
            "ngl": self.ngl,
            "host": self.host,
            "format": self.format,
            "spill": self.spill,
            "mtp": bool(self.mtp),
            "mtp_draft_n": int(self.mtp_draft_n) if self.mtp else 0,
            "vision": bool(self.vision),
            "vllm": bool(self.vllm),
            "devices": self.devices or str(self.gpu),
            "pid": self.pid,
            "status": self.status,
            "started_at": self.started_at,
            "url": f"http://{self.host}:{self.port}",
            "log_path": str(LAST_SERVER_LOG),
            "codex_cmd": codex_command(self.alias or model_alias(self.model_name)),
        }


_lock = threading.Lock()
_servers: dict[str, ServerInstance] = {}


def _clamp_mtp_draft_n(n: int | None) -> int:
    try:
        v = int(n if n is not None else 2)
    except (TypeError, ValueError):
        v = 2
    return max(1, min(6, v))


def list_servers() -> list[ServerInstance]:
    with _lock:
        # Drop finished entries so UI doesn't keep ghosts
        dead = [
            sid
            for sid, s in _servers.items()
            if s.status in ("stopped", "exited")
            and (s.process is None or s.process.poll() is not None)
            and not _pid_alive(s.pid)
        ]
        for sid in dead:
            del _servers[sid]

        for s in list(_servers.values()):
            alive = False
            if s.process is not None:
                alive = s.process.poll() is None
            elif s.pid:
                alive = _pid_alive(s.pid)
            # "starting"/"converting" may not have a pid yet; do not mark dead.
            if not alive and s.status not in (
                "stopped",
                "failed",
                "exited",
                "converting",
                "starting",
            ):
                s.status = "failed" if (s.process and s.process.returncode) else "exited"
        return list(_servers.values())


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _next_free_port(start: int, host: str = "127.0.0.1") -> int:
    port = start
    while port < start + 200:
        if _port_free(port, host):
            return port
        port += 1
    raise RuntimeError("No free ports available")


def _read_output(server: ServerInstance) -> None:
    proc = server.process
    if not proc or not proc.stdout:
        return
    from server.analyzer import note_log_line

    persisted = None
    try:
        ensure_config_dir()
        persisted = LAST_SERVER_LOG.open("w", buffering=1)
        persisted.write(f"started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        persisted.write(f"server_id={server.id}\n")
        persisted.write(f"command={server.launch_cmd}\n")
    except OSError as e:
        server.logs.append(f"could not persist llama log: {e}")

    try:
        for line in proc.stdout:
            text = line.rstrip()
            server.logs.append(text)
            if persisted:
                persisted.write(text + "\n")
            note_log_line(server.id, text)
            lower = text.lower()
            if (
                "listening" in lower
                or "server is listening" in lower
                or "application startup complete" in lower
            ):
                server.status = "running"
    finally:
        if persisted:
            persisted.write(f"exit_code={proc.wait()}\n")
            persisted.close()


def _watch_process(server: ServerInstance) -> None:
    proc = server.process
    if not proc:
        return
    proc.wait()
    if server.status not in ("stopped",):
        if server.status == "converting":
            return
        server.status = "exited" if proc.returncode == 0 else "failed"


def get_server(server_id: str) -> ServerInstance | None:
    with _lock:
        return _servers.get(server_id)


def _convert_script(llama_server: str) -> Path | None:
    # llama-server is .../build/bin/llama-server → repo root has convert_hf_to_gguf.py
    p = Path(llama_server).resolve()
    candidates = [
        p.parents[2] / "convert_hf_to_gguf.py",  # build/bin → repo
        p.parents[1] / "convert_hf_to_gguf.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _converted_gguf_path(hf_dir: Path, outtype: str) -> Path:
    ensure_config_dir()
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(hf_dir.resolve()).encode()).hexdigest()[:10]
    return CONVERTED_DIR / f"{hf_dir.name}-{digest}-{outtype}.gguf"


def _ensure_gguf(server: ServerInstance, source: Path, fmt: str) -> Path:
    """Return a GGUF path ready for llama-server. Converts HF dirs if needed."""
    if fmt == "gguf":
        if not source.is_file():
            raise FileNotFoundError(f"Model not found: {source}")
        return source.resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"HF model directory not found: {source}")

    settings = get_settings()
    requested_outtype = str(settings.get("hf_outtype", "q8_0")).lower()
    outtype = (
        requested_outtype
        if requested_outtype in HF_CONVERTER_OUTTYPES
        else "q8_0"
    )
    if outtype != requested_outtype:
        server.logs.append(
            f"HF outtype {requested_outtype!r} is not supported by "
            "convert_hf_to_gguf.py; using q8_0"
        )
    out = _converted_gguf_path(source, outtype)

    if out.is_file() and out.stat().st_mtime >= source.stat().st_mtime:
        server.logs.append(f"using cached GGUF: {out}")
        return out

    script = _convert_script(settings["llama_server_path"])
    if not script:
        raise FileNotFoundError(
            "convert_hf_to_gguf.py not found next to llama.cpp — cannot load HF models"
        )

    settings_py = settings.get("hf_convert_python")
    py = resolve_hf_convert_python(
        str(settings_py).strip() if settings_py else None
    )
    if not py:
        server.status = "failed"
        raise RuntimeError(
            "No Python with torch found for HF → GGUF conversion. "
            "Install torch, or set hf_convert_python / HF_CONVERT_PYTHON "
            "to an interpreter that has it."
        )

    server.status = "converting"
    server.logs.append(f"converting HF → GGUF ({outtype}): {source} (python={py})")
    cmd = [
        py,
        str(script),
        str(source),
        "--outfile",
        str(out),
        "--outtype",
        outtype,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        server.logs.append(line.rstrip())
    code = proc.wait()
    if code != 0 or not out.is_file():
        server.status = "failed"
        raise RuntimeError(f"HF conversion failed (exit {code})")
    server.logs.append(f"conversion done: {out}")
    return out


def _normalize_spill(spill: str | None) -> str:
    s = (spill or "none").strip().lower()
    return s if s in SPILL_MODES else "none"


def _device_list(primary: int, spill: str) -> list[int]:
    """Ordered CUDA device indices for CUDA_VISIBLE_DEVICES (primary first)."""
    gpus = gpu_mod.list_gpus()
    indexes = [g.index for g in gpus] or [primary]
    if primary not in indexes:
        indexes = [primary] + [i for i in indexes if i != primary]
    devices = [primary]
    if spill in ("gpu", "both"):
        for i in indexes:
            if i != primary and i not in devices:
                devices.append(i)
    return devices


def _effective_spill(spill: str, devices: list[int]) -> str:
    """If no secondary GPU exists, map gpu/both down to none/ram."""
    spill = _normalize_spill(spill)
    if spill in ("gpu", "both") and len(devices) < 2:
        return "ram" if spill == "both" else "none"
    return spill


def _tensor_split_for(devices: list[int]) -> str | None:
    if len(devices) < 2:
        return None
    by_idx = {g.index: max(1, g.memory_total_mib) for g in gpu_mod.list_gpus()}
    parts = [str(by_idx.get(i, 1)) for i in devices]
    return ",".join(parts)


def _fit_target_for(devices: list[int]) -> str:
    """Keep enough free VRAM for request-specific CUDA and MTP graphs."""
    if len(devices) < 2:
        return "2048"

    by_idx = {g.index: g for g in gpu_mod.list_gpus()}
    primary = by_idx.get(devices[0])
    secondary = by_idx.get(devices[1])
    if (
        len(devices) == 2
        and primary is not None
        and secondary is not None
        and primary.memory_total_mib >= 30000
        and secondary.memory_total_mib <= 16000
    ):
        return "1024,3072"
    return ",".join("1024" for _ in devices)


def _validate_cuda_capacity(server: ServerInstance, gguf: Path) -> None:
    """Reject obviously impossible non-RAM placements before CUDA starts."""
    spill = _normalize_spill(server.spill)
    arch = gguf_meta.read_gguf_arch(str(gguf))
    if server.mtp and arch is not None and not arch.get("mtp_capable"):
        raise PlacementError(
            "MTP was requested, but this GGUF has no next-token prediction layer"
        )
    if spill in ("ram", "both"):
        return

    n_layer = int((arch or {}).get("n_layer") or 0)
    if spill == "none" and server.ngl < (n_layer or 999):
        return

    gpus = gpu_mod.list_gpus()
    primary = next((g for g in gpus if g.index == server.gpu), None)
    if not primary:
        raise PlacementError(f"GPU {server.gpu} is not available")

    weights_bytes = gguf.stat().st_size
    est = gguf_meta.estimate_vram_mib(
        weights_bytes=weights_bytes,
        ctx=server.ctx,
        arch=arch,
        mtp=server.mtp,
        mtp_draft_n=server.mtp_draft_n,
        kv_bytes_per_elem=(
            1
            if gguf_meta.uses_qwen35_q8_kv(arch, server.ctx, weights_bytes)
            else 2
        ),
    )
    need_mib = float(est["total_mib"])
    if spill == "none":
        free_mib = float(primary.memory_free_mib)
        if need_mib <= free_mib:
            return

        raise CapacityWarning(
            f"full offload needs about {need_mib / 1024:.1f} GiB but GPU "
            f"{server.gpu} has {free_mib / 1024:.1f} GiB free; choose RAM "
            "fallback, another GPU, a smaller context, or fewer GPU layers"
        )

    devices = _device_list(server.gpu, spill)
    by_idx = {g.index: g for g in gpus}
    totals = [float(by_idx[i].memory_total_mib) for i in devices if i in by_idx]
    total_mib = sum(totals)
    if not totals or total_mib <= 0:
        raise PlacementError("no CUDA devices are available")
    for idx, total in zip(devices, totals):
        free_mib = float(by_idx[idx].memory_free_mib)
        share_mib = need_mib * total / total_mib
        if share_mib > free_mib:
            raise CapacityWarning(
                f"GPU {idx} needs about {share_mib / 1024:.1f} GiB of the "
                f"multi-GPU split but has {free_mib / 1024:.1f} GiB free; "
                "choose RAM fallback, a smaller model/context, or free VRAM"
            )


def _warn_cuda_capacity(server: ServerInstance, gguf: Path) -> None:
    """Record an estimated VRAM problem, but do not stop the launch."""
    try:
        _validate_cuda_capacity(server, gguf)
    except CapacityWarning as exc:
        warning = f"warning: VRAM preflight: {exc}; launch override is enabled"
        if warning not in server.logs:
            server.logs.append(warning)


def _spawn_llama(
    server: ServerInstance,
    gguf: Path,
    binary: str,
    env: dict,
) -> None:
    _warn_cuda_capacity(server, gguf)

    spill = _normalize_spill(server.spill)
    devices = _device_list(server.gpu, spill)
    spill = _effective_spill(spill, devices)
    server.spill = spill
    server.devices = ",".join(str(d) for d in devices)
    env["CUDA_VISIBLE_DEVICES"] = server.devices

    cmd = [
        binary,
        "-m",
        str(gguf),
        "-c",
        str(server.ctx),
        "--host",
        server.host,
        "--port",
        str(server.port),
        "-a",
        server.alias or model_alias(server.model_name),
    ]

    if server.vision:
        mmproj = model_scan.find_sibling_mmproj(gguf)
        if mmproj is None:
            raise FileNotFoundError(
                f"Vision enabled but no mmproj*.gguf found next to {gguf.name}. "
                "Download the model's compatible mmproj GGUF into the same folder "
                "(for this Qwen3.8 model: mmproj-BF16.gguf)."
            )
        cmd.extend(["--mmproj", str(mmproj)])
        server.logs.append(f"mmproj={mmproj.name}")
        # Chat templates for VL models (Qwen3-VL / Qwen3.6, etc.)
        cmd.append("--jinja")

    arch = gguf_meta.read_gguf_arch(str(gguf))
    qwen35_defaults = gguf_meta.is_qwen35_arch(arch)
    qwen38_nvfp4_mtp = gguf_meta.is_qwen38_nvfp4_mtp(gguf, arch)
    qwen35_long_context = (
        server.ctx >= 131072
        and gguf_meta.uses_qwen35_q8_kv(arch, server.ctx, gguf.stat().st_size)
    )
    # Avoid CPU layer placement when the estimated configuration fits the
    # selected GPU with a safety reserve.
    force_full_gpu = False
    try:
        gpus = gpu_mod.list_gpus()
        primary = next((g for g in gpus if g.index == server.gpu), None)
        kv_elem = 1 if qwen35_long_context else 2
        est = gguf_meta.estimate_vram_mib(
            weights_bytes=gguf.stat().st_size,
            ctx=server.ctx,
            arch=arch,
            mtp=server.mtp,
            mtp_draft_n=server.mtp_draft_n,
            kv_bytes_per_elem=kv_elem,
        )
        force_full_gpu = bool(
            qwen35_long_context
            and len(devices) == 1
            and primary is not None
            and float(est["total_mib"]) + 1024.0 <= float(primary.memory_free_mib)
        )
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        pass

    # Spill modes use auto layer placement; gpu-only keeps explicit -ngl.
    # --fit cannot run if -ts/--tensor-split is set (llama.cpp aborts fit).
    use_fit = spill in ("ram", "both") and not force_full_gpu
    fit_target = None
    if spill == "none" or force_full_gpu:
        cmd.extend(["-ngl", str(server.ngl if server.ngl > 0 else 999)])
        cmd.extend(["--fit", "off"])
    else:
        cmd.extend(["-ngl", "auto"])
        # ram/both: fit may leave leftover layers on CPU/RAM
        # gpu: try to stay on GPUs only
        cmd.extend(["--fit", "on" if use_fit else "off"])
        if use_fit:
            fit_target = _fit_target_for(devices)
            cmd.extend(["--fit-target", fit_target])

    if len(devices) > 1:
        cmd.extend(["-sm", "layer", "-mg", "0"])
        if not use_fit:
            ts = _tensor_split_for(devices)
            if ts:
                cmd.extend(["-ts", ts])

    if qwen35_long_context:
        # Use the stable long-context CUDA path for variable request shapes.
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        env["GGML_CUDA_PDL"] = "0"
    if qwen38_nvfp4_mtp:
        # Use the checkpoint publisher's tested general-chat sampling profile.
        cmd.extend(["--temp", "1.0", "--top-p", "0.95", "--top-k", "20"])
    elif qwen35_defaults:
        # Qwen's precise-coding profile. API request values still override it.
        cmd.extend(["--temp", "0.6", "--top-k", "20", "--min-p", "0.0"])

    if qwen35_long_context:
        # Measured winner at 128K and 256K: Q8 preserves more quality than the
        # smaller cache quants while avoiding slow CPU KV/cache spill.
        # The full-GPU 128K profile needs a smaller prompt micro-batch so MTP's
        # final compute buffer fits alongside a desktop on the 32 GiB card.
        if force_full_gpu:
            cmd.extend(["-ub", "256"])
        cmd.extend(
            [
                "-fa",
                "on",
                "-ctk",
                "q8_0",
                "-ctv",
                "q8_0",
            ]
        )
        if server.mtp:
            cmd.extend(["-ctkd", "q8_0", "-ctvd", "q8_0"])

    if server.mtp:
        draft_n = _clamp_mtp_draft_n(server.mtp_draft_n)
        server.mtp_draft_n = draft_n
        # Qwen3.6 MTP currently requires one slot; flash attention is the
        # publisher-recommended CUDA path.
        if not qwen35_long_context:
            cmd.extend(["-fa", "on"])
        cmd.extend(
            [
                "-np",
                "1",
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(draft_n),
            ]
        )
        if qwen38_nvfp4_mtp:
            cmd.extend(["--spec-draft-p-split", "0.2"])

    ngl_arg = cmd[cmd.index("-ngl") + 1] if "-ngl" in cmd else "?"
    fit_arg = cmd[cmd.index("--fit") + 1] if "--fit" in cmd else "?"
    mtp_arg = (
        f" mtp=draft-mtp n={server.mtp_draft_n} fa=on slots=1"
        if server.mtp
        else ""
    )
    vision_arg = " vision=mmproj" if server.vision else ""
    server.logs.append(
        f"spawn spill={spill} devices={server.devices} ngl={ngl_arg} fit={fit_arg}"
        + (" ts=auto" if use_fit and len(devices) > 1 else "")
        + (f" fit_target={fit_target}" if fit_target else "")
        + (
            " sampling=qwen38-nvfp4"
            if qwen38_nvfp4_mtp
            else " sampling=qwen35-coding"
            if qwen35_defaults
            else ""
        )
        + (" kv=q8_0" if qwen35_long_context else "")
        + (" cuda_graphs=off pdl=off" if qwen35_long_context else "")
        + (" full_gpu=1" if force_full_gpu and spill != "none" else "")
        + mtp_arg
        + (" p_split=0.2" if server.mtp and qwen38_nvfp4_mtp else "")
        + vision_arg
    )
    env_prefix = [f"CUDA_VISIBLE_DEVICES={server.devices}"]
    if qwen35_long_context:
        env_prefix.extend(["GGML_CUDA_DISABLE_GRAPHS=1", "GGML_CUDA_PDL=0"])
    server.launch_cmd = " ".join(env_prefix + [shlex.join(cmd)])
    server.status = "starting"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )
    server.process = proc
    server.pid = proc.pid
    threading.Thread(target=_read_output, args=(server,), daemon=True).start()
    threading.Thread(target=_watch_process, args=(server,), daemon=True).start()


def _spawn_vllm(server: ServerInstance, source: Path, binary: str, env: dict) -> None:
    """Serve a native compressed NVFP4 Hugging Face checkpoint with vLLM."""
    if not source.is_dir():
        raise FileNotFoundError(f"vLLM model directory not found: {source}")
    if not Path(binary).is_file():
        raise FileNotFoundError(f"vLLM executable not found: {binary}")

    server.vllm = True
    server.devices = str(server.gpu)
    env["CUDA_VISIBLE_DEVICES"] = server.devices
    # Use vLLM's built-in sampler to avoid a host-toolkit JIT dependency.
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    # The 27B model needs just over 5 GiB of FP8 KV cache for 128K. Keep
    # enough free GPU memory for temporary MTP allocations and the display.
    long_context = server.ctx >= 131072
    gpu_memory_utilization = "0.90"
    max_num_seqs = 4
    max_num_batched_tokens = 4096 if long_context else 2048

    cmd = [
        binary,
        "serve",
        str(source.resolve()),
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--served-model-name",
        server.alias or model_alias(server.model_name),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(server.ctx),
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--attention-backend",
        "TRITON_ATTN",
        "--enable-prefix-caching",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]
    if not server.vision:
        cmd.append("--language-model-only")
    if server.spill == "ram":
        # Keep a CPU KV-cache buffer for contexts that are larger than the
        # GPU KV-cache capacity. This is vLLM's native RAM fallback path.
        cmd.extend(
            [
                "--kv-offloading-backend",
                "native",
                "--kv-offloading-size",
                "8",
            ]
        )
    if server.mtp:
        cmd.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "mtp",
                        "num_speculative_tokens": _clamp_mtp_draft_n(
                            server.mtp_draft_n
                        ),
                        "attention_backend": "TRITON_ATTN",
                    },
                    separators=(",", ":"),
                ),
            ]
        )

    server.logs.append(
        "spawn backend=vllm quant=nvfp4 kv=fp8_e4m3 "
        f"device={server.devices} vision={int(server.vision)} "
        f"mtp={int(server.mtp)} prefix_cache=on max_seqs={max_num_seqs} "
        f"max_batched={max_num_batched_tokens} attn=triton "
        f"gpu_mem={gpu_memory_utilization} "
        f"kv_ram_gib={8 if server.spill == 'ram' else 0}"
    )
    server.launch_cmd = " ".join(
        [
            f"CUDA_VISIBLE_DEVICES={server.devices}",
            "VLLM_USE_FLASHINFER_SAMPLER=0",
            shlex.join(cmd),
        ]
    )
    server.status = "starting"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )
    server.process = proc
    server.pid = proc.pid
    threading.Thread(target=_read_output, args=(server,), daemon=True).start()
    threading.Thread(target=_watch_process, args=(server,), daemon=True).start()


def _boot_server(server: ServerInstance, source: Path, fmt: str) -> None:
    settings = get_settings()
    binary = settings["llama_server_path"]
    env = os.environ.copy()
    try:
        if fmt == "vllm":
            _spawn_vllm(server, source, str(settings.get("vllm_bin") or ""), env)
            return
        gguf = _ensure_gguf(server, source, fmt)
        if server.status == "stopped":
            return
        _spawn_llama(server, gguf, binary, env)
    except Exception as e:
        server.status = "failed"
        server.logs.append(f"error: {e}")


def start_server(
    model_path: str,
    gpu: int,
    ctx: int | None = None,
    port: int | None = None,
    ngl: int | None = None,
    host: str | None = None,
    spill: str | None = None,
    mtp: bool = False,
    mtp_draft_n: int | None = None,
    vision: bool = False,
    format_hint: str | None = None,
) -> ServerInstance:
    settings = get_settings()
    source = Path(model_path)
    info = model_scan.find_model(model_path)
    hinted_format = str(format_hint or "").lower()
    if hinted_format not in ("gguf", "hf", "vllm"):
        hinted_format = ""
    fmt = info.format if info else (
        hinted_format or ("hf" if source.is_dir() else "gguf")
    )
    display_name = info.name if info else source.name
    if not info:
        repo = model_scan.hf_hub_repo_from_path(source)
        if repo:
            display_name = repo

    if fmt == "gguf" and not source.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if fmt in ("hf", "vllm") and not source.is_dir():
        raise FileNotFoundError(f"HF model directory not found: {model_path}")

    arch = gguf_meta.read_gguf_arch(str(source)) if fmt == "gguf" else None
    binary = settings["llama_server_path"]
    if fmt == "vllm":
        vllm_bin = str(settings.get("vllm_bin") or "")
        if not Path(vllm_bin).is_file():
            raise FileNotFoundError(f"vLLM executable not found: {vllm_bin}")
    elif not Path(binary).is_file():
        raise FileNotFoundError(f"llama-server not found: {binary}")

    ctx = ctx if ctx is not None else int(settings["default_ctx"])
    ctx = max(512, min(1048576, int(ctx)))
    ngl = ngl if ngl is not None else int(settings["default_ngl"])
    host = host or settings["default_host"]
    spill = _normalize_spill(spill)
    mtp = bool(mtp)
    draft_n = _clamp_mtp_draft_n(
        mtp_draft_n
        if mtp_draft_n is not None
        else gguf_meta.recommended_mtp_draft_n(source, arch)
    )
    vision = bool(vision)

    if fmt == "vllm":
        # Native vLLM does not use llama.cpp GPU spill. Keep RAM fallback as
        # native CPU KV-cache offload. Treat "both" as RAM on one GPU.
        spill = "ram" if spill in ("ram", "both") else "none"
        vision = bool(vision and info and info.has_mmproj)

    if vision and fmt == "gguf":
        mmproj = model_scan.find_sibling_mmproj(source)
        if mmproj is None:
            raise FileNotFoundError(
                f"Vision enabled but no mmproj*.gguf found next to {source.name}. "
                "Download the model's compatible mmproj GGUF into the same folder "
                "(for this Qwen3.8 model: mmproj-BF16.gguf)."
            )
    if port is None:
        port = _next_free_port(int(settings["port_start"]), host)
    elif not _port_free(port, host):
        port = _next_free_port(int(settings["port_start"]), host)

    server_id = str(uuid.uuid4())[:8]
    # Prefer unique GGUF basename for --alias; HF repo id is matched via path /v1 list
    alias = model_alias(display_name)
    if info and info.alias:
        alias = model_alias(info.alias)
    elif not info:
        repo = model_scan.hf_hub_repo_from_path(source)
        if repo:
            alias = model_alias(repo)
    devices = _device_list(gpu, spill)
    instance = ServerInstance(
        id=server_id,
        model_path=str(source.resolve()),
        model_name=display_name,
        alias=alias,
        gpu=gpu,
        port=port,
        ctx=ctx,
        ngl=ngl,
        host=host,
        format=fmt,
        spill=spill,
        mtp=mtp,
        mtp_draft_n=draft_n,
        vision=vision,
        vllm=fmt == "vllm",
        devices=",".join(str(d) for d in devices),
        status="converting" if fmt == "hf" else "starting",
    )

    if fmt == "gguf":
        _warn_cuda_capacity(instance, source.resolve())

    with _lock:
        _servers[server_id] = instance

    threading.Thread(
        target=_boot_server, args=(instance, source, fmt), daemon=True
    ).start()

    return instance


def stop_server(server_id: str) -> bool:
    with _lock:
        server = _servers.get(server_id)
        if not server:
            return False
        server.status = "stopped"
        proc = server.process
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    proc.kill()
        return True


def stop_all_servers() -> list[str]:
    ids = [s.id for s in list_servers() if s.status not in ("stopped",)]
    stopped: list[str] = []
    for sid in ids:
        if stop_server(sid):
            stopped.append(sid)
    return stopped


def find_server_by_model(model: str) -> ServerInstance | None:
    """Match OpenAI model id to a running/starting server."""
    raw = (model or "").replace("openai:", "").strip()
    want = model_alias(raw)
    want_l = want.lower()
    want_hub = want_l.replace("/", "--")  # org/repo → org--repo
    candidates = [
        s
        for s in list_servers()
        if s.status in ("running", "starting")
    ]

    def _matches(s: ServerInstance) -> bool:
        ids = {
            model_alias(s.alias).lower(),
            model_alias(s.model_name).lower(),
            Path(s.model_name).stem.lower(),
        }
        if want_l in ids:
            return True
        path_l = (s.model_path or "").lower().replace("\\", "/")
        if want_l and want_l in path_l:
            return True
        # HF hub folder: models--org--repo
        if want_hub and f"models--{want_hub}" in path_l.replace("/", "--"):
            return True
        if "--" in want_hub:
            token = f"models--{want_hub}"
            if token in path_l.replace("\\", "/"):
                return True
        return False

    for s in candidates:
        if _matches(s):
            return s
    for s in candidates:
        if s.model_name == model or s.model_path == model:
            return s
    if len(candidates) == 1 and (not model or model in ("default", "local")):
        return candidates[0]
    return None


def running_aliases() -> list[str]:
    return [
        s.alias or model_alias(s.model_name)
        for s in list_servers()
        if s.status in ("running", "starting")
    ]

def get_logs(server_id: str, tail: int = 100) -> list[str]:
    server = get_server(server_id)
    if not server:
        return []
    return list(server.logs)[-tail:]
