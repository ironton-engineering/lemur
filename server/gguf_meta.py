"""Lightweight GGUF metadata reader for VRAM estimates."""
from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path
from typing import Any


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _skip_value(f, typ: int) -> None:
    if typ in (0, 1, 7):  # u8/i8/bool
        f.read(1)
    elif typ in (2, 3):  # u16/i16
        f.read(2)
    elif typ in (4, 5, 6):  # u32/i32/f32
        f.read(4)
    elif typ == 8:  # string
        _read_string(f)
    elif typ in (10, 11, 12):  # u64/i64/f64
        f.read(8)
    elif typ == 9:  # array
        (at,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _skip_value(f, at)
    else:
        raise ValueError(f"unsupported GGUF type {typ}")


def _read_value(f, typ: int) -> Any:
    if typ == 4:
        return struct.unpack("<I", f.read(4))[0]
    if typ == 5:
        return struct.unpack("<i", f.read(4))[0]
    if typ == 6:
        return struct.unpack("<f", f.read(4))[0]
    if typ == 7:
        return bool(f.read(1)[0])
    if typ == 8:
        return _read_string(f)
    if typ == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if typ == 11:
        return struct.unpack("<q", f.read(8))[0]
    if typ == 12:
        return struct.unpack("<d", f.read(8))[0]
    _skip_value(f, typ)
    return None


def _find(meta: dict[str, Any], suffix: str) -> Any:
    if suffix in meta:
        return meta[suffix]
    for k, v in meta.items():
        if k == suffix or k.endswith("." + suffix):
            return v
    return None


def _as_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def is_qwen35_arch(arch: dict[str, Any] | None) -> bool:
    """Qwen 3.5 and later GGUFs share the qwen35 runtime profile."""
    return str((arch or {}).get("architecture") or "").lower().startswith("qwen35")


def is_qwen38_nvfp4_mtp(
    path: str | Path, arch: dict[str, Any] | None = None
) -> bool:
    """True for the Qwen3.8 NVFP4 GGUF profile with an embedded MTP head."""
    name = Path(path).name.lower()
    return (
        is_qwen35_arch(arch)
        and "qwen3.8" in name
        and "nvfp4" in name
        and model_looks_mtp(str(path), arch)
    )


def recommended_mtp_draft_n(
    path: str | Path, arch: dict[str, Any] | None = None
) -> int:
    """Return the publisher-tested MTP draft count for a GGUF."""
    return 3 if is_qwen38_nvfp4_mtp(path, arch) else 2


def uses_qwen35_q8_kv(
    arch: dict[str, Any] | None, ctx: int, weights_bytes: int
) -> bool:
    # Qwen3.8 NVFP4 is only 18.34 GiB, but it is still a 27B model with 64
    # base layers. Use layer count as well as file size so its 128K+ cache uses
    # the publisher-tested Q8 profile. Small Qwen models keep their default KV.
    n_layer = int((arch or {}).get("n_layer") or 0)
    nextn = int((arch or {}).get("nextn_predict_layers") or 0)
    base_layers = max(0, n_layer - nextn)
    return (
        is_qwen35_arch(arch)
        and ctx >= 131072
        and (weights_bytes >= 20 * 1024**3 or base_layers >= 60)
    )


def _attention_layer_count(architecture: str, base_layers: int) -> int:
    if (
        architecture.startswith("qwen35") or architecture.startswith("qwen3next")
    ) and base_layers >= 4:
        return max(1, base_layers // 4)
    return base_layers


@lru_cache(maxsize=256)
def read_gguf_arch(path: str) -> dict[str, Any] | None:
    """Return architecture fields needed for KV estimates, or None."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            if f.read(4) != b"GGUF":
                return None
            struct.unpack("<I", f.read(4))  # version
            _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            interesting = (
                "architecture",
                "block_count",
                "embedding_length",
                "attention.head_count",
                "attention.head_count_kv",
                "attention.key_length",
                "attention.value_length",
                "context_length",
                "expert_count",
                "expert_used_count",
                "nextn_predict_layers",
            )
            meta: dict[str, Any] = {}
            for _ in range(n_kv):
                key = _read_string(f)
                (typ,) = struct.unpack("<I", f.read(4))
                if any(s in key for s in interesting):
                    meta[key] = _read_value(f, typ)
                else:
                    _skip_value(f, typ)
    except (OSError, struct.error, ValueError, UnicodeError):
        return None

    n_layer = _as_int(_find(meta, "block_count"))
    n_embd = _as_int(_find(meta, "embedding_length"))
    n_head = _as_int(_find(meta, "attention.head_count"))
    if not n_layer or not n_embd or not n_head or n_head <= 0:
        return None
    n_head_kv = _as_int(_find(meta, "attention.head_count_kv")) or n_head
    key_len = _as_int(_find(meta, "attention.key_length"))
    val_len = _as_int(_find(meta, "attention.value_length"))
    # Prefer explicit K/V dims (Qwen3.5/3.6 hybrid: embd % n_head != 0).
    head_dim = key_len or val_len
    if not head_dim:
        if n_embd % n_head != 0:
            return None
        head_dim = n_embd // n_head

    architecture = str(_find(meta, "architecture") or "unknown")
    nextn = _as_int(_find(meta, "nextn_predict_layers")) or 0
    # Hybrid qwen35/qwen3next: groups of 4 layers, last is full attention.
    # Remaining layers are linear-attn/SSM with tiny state vs full KV.
    base_layers = max(1, n_layer - max(0, nextn))
    n_attn_layer = _attention_layer_count(architecture, base_layers)

    return {
        "architecture": architecture,
        "n_layer": n_layer,
        "n_attn_layer": n_attn_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "head_dim": head_dim,
        "n_ctx_train": _as_int(_find(meta, "context_length")) or None,
        "n_expert": _as_int(_find(meta, "expert_count")) or None,
        "nextn_predict_layers": nextn,
        "mtp_capable": nextn > 0,
    }


def model_looks_mtp(path: str, arch: dict[str, Any] | None = None) -> bool:
    if arch and arch.get("mtp_capable"):
        return True
    hay = path.lower()
    return "mtp" in Path(hay).name or "/mtp" in hay or "-mtp" in hay


def model_looks_diffusion(path: str, arch: dict[str, Any] | None = None) -> bool:
    """Block-diffusion GGUFs (e.g. DiffusionGemma) need a dedicated runner."""
    if arch:
        a = str(arch.get("architecture") or "").lower()
        if a.startswith("diffusion") or "diffusion" in a:
            return True
    hay = path.lower().replace("\\", "/")
    name = Path(hay).name
    return (
        "diffusiongemma" in hay
        or "diffusion-gemma" in hay
        or name.startswith("diffusion")
    )


def _read_hf_config(path: str | Path) -> dict[str, Any] | None:
    import json

    p = Path(path)
    cfg_path = p / "config.json" if p.is_dir() else None
    if cfg_path is None or not cfg_path.is_file():
        return None
    try:
        with cfg_path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def model_looks_dream(path: str, arch: dict[str, Any] | None = None) -> bool:
    """Dream / Dream-Coder discrete diffusion models (HF or GGUF)."""
    if arch:
        a = str(arch.get("architecture") or "").lower()
        if a == "dream" or a.startswith("dream"):
            return True
    p = Path(path)
    cfg = _read_hf_config(p)
    if cfg:
        mt = str(cfg.get("model_type") or "").lower()
        if mt == "dream":
            return True
        archs = " ".join(str(x) for x in (cfg.get("architectures") or []))
        if "DreamModel" in archs or "dreammodel" in archs.lower():
            return True
        if (p / "modeling_dream.py").is_file():
            return True
    hay = path.lower().replace("\\", "/")
    name = Path(hay).name
    return (
        "dream-coder" in hay
        or "dream_coder" in hay
        or "/dream-org/" in hay
        or "models--dream-org--" in hay
        or name.startswith("dream-")
        or name.startswith("dream_")
    )


def estimate_dream_vram_mib(
    *,
    weights_bytes: int,
    max_new_tokens: int = 768,
) -> dict[str, float | bool | int]:
    """Estimate VRAM for Dream Transformers BF16 inference.

    Weights dominate (~BF16 safetensors size). Activations grow with the
    diffusion canvas (max_new_tokens); Hub spill/MTP/ngl do not apply.
    """
    weights_mib = weights_bytes / (1024 * 1024)
    # Runtime / optimizer-free activations / SDPA workspace pad.
    max_new = max(1, min(int(max_new_tokens or 768), 4096))
    # Rough: ~2–4 bytes * hidden * seq for a few scratch tensors; pad generously.
    act_mib = 1024.0 + (max_new / 768.0) * 512.0
    total = weights_mib + act_mib
    return {
        "weights_mib": weights_mib,
        "kv_mib": 0.0,
        "scratch_mib": act_mib,
        "mtp_mib": 0.0,
        "total_mib": total,
        "confidence": "medium",
        "mtp": False,
        "mtp_draft_n": 0,
        "diffusion": True,
        "dream": True,
        "maxtok": max_new,
        "max_new_tokens": max_new,
    }


def estimate_diffusion_vram_mib(
    *,
    weights_bytes: int,
    maxtok: int = 0,
    n_head: int = 16,
    canvas: int = 256,
) -> dict[str, float | bool | int]:
    """Estimate memory for DiffusionGemma's visual decoder (not llama-server).

    The runner loads weights with full GPU offload (NGL=99), then auto-sizes a
    non-causal MAXTOK context to remaining VRAM (else system RAM). Compute has a
    quadratic fp32 scores term ~ n_head * N * N * 4 bytes. Hub spill modes do not
    apply — the binary decides VRAM vs RAM itself.
    """
    weights_mib = weights_bytes / (1024 * 1024)
    n_head = max(1, int(n_head or 16))
    canvas = max(1, int(canvas or 256))
    # Hub maps ctx→maxtok only when 1..8192; else auto (0).
    explicit = int(maxtok) if 0 < int(maxtok) <= 8192 else 0
    # Runtime / logits / SC pad after weights are resident.
    runtime_mib = 1536.0
    if explicit:
        # Match visual-server probe: fp32 [n_head, N, N] scores buffer.
        scores_mib = (n_head * explicit * explicit * 4.0) / (1024 * 1024)
        compute_mib = scores_mib + 512.0
        resolved = explicit
    else:
        # Auto-size: budget is "whatever free VRAM remains"; report a typical
        # working set (weights + modest pad). Actual MAXTOK grows into free VRAM.
        compute_mib = runtime_mib
        resolved = 0
    total = weights_mib + compute_mib
    return {
        "weights_mib": weights_mib,
        "kv_mib": 0.0,  # no AR KV cache of launch ctx
        "scratch_mib": compute_mib,
        "mtp_mib": 0.0,
        "total_mib": total,
        "confidence": "medium",
        "mtp": False,
        "mtp_draft_n": 0,
        "diffusion": True,
        "maxtok": resolved,
        "canvas": canvas,
        "n_head": n_head,
    }


def estimate_vram_mib(
    *,
    weights_bytes: int,
    ctx: int,
    arch: dict[str, Any] | None,
    mtp: bool = False,
    mtp_draft_n: int = 2,
    kv_bytes_per_elem: int = 2,  # f16/bf16 KV
) -> dict[str, float | bool]:
    """Estimate VRAM for full GPU offload (ngl=all / auto fit)."""
    ctx = max(512, int(ctx))
    draft_n = max(1, min(6, int(mtp_draft_n or 2)))
    weights_mib = weights_bytes / (1024 * 1024)
    kv_mib = 0.0
    confidence = "low"
    if arch:
        n_attn = int(arch.get("n_attn_layer") or arch["n_layer"])
        kv_bytes = (
            2
            * n_attn
            * ctx
            * int(arch["n_head_kv"])
            * int(arch["head_dim"])
            * kv_bytes_per_elem
        )
        kv_mib = kv_bytes / (1024 * 1024)
        # Hybrid SSM/DeltaNet state is small vs attention KV; pad a little.
        if int(arch.get("n_layer") or 0) > n_attn:
            kv_mib += (ctx / 1024.0) * 8.0
        confidence = "high"
    else:
        # Fallback: ~48MiB per 1k ctx @7B-ish GQA, scaled by weight size
        scale = max(0.5, min(3.0, (weights_bytes / (1024**3)) / 14.0))
        kv_mib = (ctx / 1024.0) * 48.0 * scale

    # Compute / graph / runtime scratch — grows with ctx
    scratch_mib = 768.0 + (ctx / 8192.0) * 128.0

    # MTP: Unsloth ~+1GB plus draft context / speculative buffers.
    # Failed launches showed ~0.7–0.9GB cudaMalloc on the small GPU alone.
    mtp_mib = 0.0
    if mtp:
        mtp_mib = 1024.0 + (kv_mib * 0.08 * draft_n) + (draft_n * 64.0)

    total = weights_mib + kv_mib + scratch_mib + mtp_mib
    return {
        "weights_mib": weights_mib,
        "kv_mib": kv_mib,
        "scratch_mib": scratch_mib,
        "mtp_mib": mtp_mib,
        "total_mib": total,
        "confidence": confidence,
        "mtp": bool(mtp),
        "mtp_draft_n": draft_n if mtp else 0,
    }
