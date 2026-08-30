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
