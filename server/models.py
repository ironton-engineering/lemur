import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from server.config import get_settings

# GGUF multi-part: name-00001-of-00003.gguf
SHARD_RE = re.compile(
    r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)

TEXT_ARCH_HINTS = (
    "ForCausalLM",
    "LMHeadModel",
)

SKIP_ARCH_HINTS = (
    "Diffusion",
    "UNet",
    "VAE",
    "CLIP",
    "Vision",
    "Image",
    "Audio",
    "Speech",
    "Whisper",
    "Flux",
    "ConditionalGeneration",  # usually multimodal
)

SKIP_CONFIG_KEYS = (
    "vision_config",
    "image_token_index",
    "image_token_id",
    "video_token_id",
    "mm_tokens_per_image",
    "video_preprocessor_config",
)


@dataclass
class ModelInfo:
    name: str
    path: str
    size_bytes: int
    folder: str
    format: str = "gguf"  # gguf | hf | vllm
    shards: int = 1
    shard_paths: list[str] = field(default_factory=list)
    # Preferred OpenAI / Codex model id (e.g. satgeze/Ornith-1.0-9B-1M-GGUF)
    alias: str | None = None
    has_mmproj: bool = False
    mmproj_name: str | None = None
    mtp_capable: bool = False
    recommended_mtp_draft_n: int = 2

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / (1024**3), 2),
            "folder": self.folder,
            "format": self.format,
            "shards": self.shards,
            "alias": self.alias,
            "has_mmproj": bool(self.has_mmproj),
            "mmproj_name": self.mmproj_name,
            "mtp_capable": bool(self.mtp_capable),
            "recommended_mtp_draft_n": int(self.recommended_mtp_draft_n),
        }


_scan_lock = threading.Lock()
_cache: list[ModelInfo] = []
_cache_time = 0.0
_scanning = False


def _hf_hub_repo_from_path(path: Path) -> str | None:
    """Map .../hub/models--org--repo/... → org/repo."""
    for part in path.parts:
        if not part.startswith("models--"):
            continue
        rest = part[len("models--") :]
        org, sep, repo = rest.partition("--")
        if sep and org and repo:
            return f"{org}/{repo}"
    return None


def _is_hf_hub_path(dirname: str) -> bool:
    parts = Path(dirname).parts
    try:
        i = parts.index("huggingface")
        return i + 1 < len(parts) and parts[i + 1] == "hub"
    except ValueError:
        return False


def _should_skip_dir(dirname: str, exclude_dirs: list[str]) -> bool:
    parts = Path(dirname).parts
    # GGUFs from `huggingface-cli download` live under ~/.cache/huggingface/hub
    if _is_hf_hub_path(dirname):
        leaf = parts[-1] if parts else ""
        return leaf in (".locks", "blobs", ".git")
    # Allow descending into the HF hub parent chain only
    if parts and parts[-1] == ".cache":
        return False
    if len(parts) >= 2 and parts[-2] == ".cache" and parts[-1] == "huggingface":
        return False
    for part in parts:
        if part in (".cache", ".git", "node_modules", "__pycache__", ".venv", "venv"):
            return True
    norm = dirname.replace("\\", "/").lower()
    for excl in exclude_dirs:
        el = excl.lower().strip("/")
        if not el:
            continue
        # Don't let a bare ".cache" exclude rule kill the HF hub path we allow above
        if el == ".cache":
            continue
        if el in norm:
            return True
    return False


def _preferred_alias(name: str, path: Path) -> str | None:
    return _hf_hub_repo_from_path(path)


def _recommended_mtp_draft_n(name: str) -> int:
    lower = name.lower()
    if "qwen3.8" in lower and "nvfp4" in lower and "mtp" in lower:
        return 3
    return 2

def _is_text_gguf(filename: str, size_bytes: int, min_size: int, is_shard: bool) -> bool:
    lower = filename.lower()
    if lower.startswith("mmproj") or "mmproj" in lower:
        return False
    if "vocab" in lower and size_bytes < min_size * 4:
        return False
    # First shard of a split model is often tiny (index/metadata only)
    if is_shard:
        return True
    if size_bytes < min_size:
        return False
    return True


def find_sibling_mmproj(gguf: Path) -> Path | None:
    """Prefer an mmproj GGUF next to the model (required for vision)."""
    folder = gguf.parent if gguf.is_file() else gguf
    candidates = sorted(folder.glob("mmproj*.gguf")) + sorted(
        folder.glob("*mmproj*.gguf")
    )
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    if not uniq:
        return None

    def rank(p: Path) -> tuple[int, int, str]:
        name = p.name.lower()
        quality = 0
        if "f16" in name or "bf16" in name:
            quality = 3
        elif "q8" in name:
            quality = 2
        elif "q4" in name:
            quality = 1
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        return (-quality, -size, p.name)

    return sorted(uniq, key=rank)[0]


def _mmproj_fields(gguf: Path) -> tuple[bool, str | None]:
    mmproj = find_sibling_mmproj(gguf)
    if mmproj is None:
        return False, None
    return True, mmproj.name


def _parse_shard(filename: str) -> tuple[str, int, int] | None:
    m = SHARD_RE.match(filename)
    if not m:
        return None
    return m.group("base"), int(m.group("index")), int(m.group("total"))


def _path_str(path: Path) -> str:
    """Absolute path without following HF hub snapshot→blob symlinks."""
    return str(path if path.is_absolute() else path.absolute())


def _group_ggufs(raw: list[tuple[str, Path, int]]) -> list[ModelInfo]:
    """Group sharded GGUFs; keep standalone files as-is."""
    groups: dict[tuple[str, str], list[tuple[int, int, Path, int]]] = {}
    singles: list[ModelInfo] = []

    for fname, fpath, size in raw:
        parsed = _parse_shard(fname)
        if parsed:
            base, index, total = parsed
            key = (str(fpath.parent), base.lower())
            groups.setdefault(key, []).append((index, total, fpath, size))
        else:
            has_mm, mm_name = _mmproj_fields(fpath)
            singles.append(
                ModelInfo(
                    name=fname,
                    path=_path_str(fpath),
                    size_bytes=size,
                    folder=str(fpath.parent),
                    format="gguf",
                    shards=1,
                    alias=_preferred_alias(fname, fpath),
                    has_mmproj=has_mm,
                    mmproj_name=mm_name,
                    recommended_mtp_draft_n=_recommended_mtp_draft_n(fname),
                )
            )

    for (_folder, _base), parts in groups.items():
        parts.sort(key=lambda x: x[0])
        totals = {t for _, t, _, _ in parts}
        total = max(totals) if totals else len(parts)
        # Prefer shard 00001 as load path (llama.cpp reads the set from there)
        first = next((p for p in parts if p[0] == 1), parts[0])
        _, _, load_path, _ = first
        total_size = sum(s for _, _, _, s in parts)
        parsed = _parse_shard(load_path.name)
        display_name = f"{parsed[0]}.gguf" if parsed else load_path.name
        has_mm, mm_name = _mmproj_fields(load_path)

        singles.append(
            ModelInfo(
                name=display_name,
                path=_path_str(load_path),
                size_bytes=total_size,
                folder=str(load_path.parent),
                format="gguf",
                shards=len(parts),
                shard_paths=[_path_str(p) for _, _, p, _ in parts],
                alias=_preferred_alias(display_name, load_path),
                has_mmproj=has_mm,
                mmproj_name=mm_name,
                recommended_mtp_draft_n=_recommended_mtp_draft_n(display_name),
            )
        )

    return singles


def _is_text_hf_dir(dirpath: Path) -> bool:
    config_path = dirpath / "config.json"
    if not config_path.is_file():
        return False
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    archs = cfg.get("architectures") or []
    if _is_native_vllm_config(cfg):
        return _has_hf_weights(dirpath)
    if any(k in cfg for k in SKIP_CONFIG_KEYS):
        return False
    if (dirpath / "video_preprocessor_config.json").exists():
        return False
    if (dirpath / "preprocessor_config.json").exists() and (
        "vision_config" in cfg or "image_token_id" in cfg
    ):
        return False

    if not archs:
        # Fallback: model_type present + weight files
        if not cfg.get("model_type"):
            return False
    else:
        arch_str = " ".join(archs)
        if any(s in arch_str for s in SKIP_ARCH_HINTS):
            return False
        if not any(h in arch_str for h in TEXT_ARCH_HINTS):
            # Allow common text model_type without CausalLM suffix
            mt = (cfg.get("model_type") or "").lower()
            # DreamModel is a diffusion LM (trust_remote_code), not CausalLM.
            if "DreamModel" in arch_str or mt == "dream":
                pass
            elif mt not in (
                "llama",
                "qwen2",
                "qwen3",
                "mistral",
                "gemma",
                "gemma2",
                "gemma3",
                "phi",
                "phi3",
                "gpt2",
                "gpt_neox",
                "minimax",
                "minimax_m2",
                "deepseek",
                "chatglm",
                "internlm",
                "dream",
            ):
                return False
            # model_type alone is weak — require no multimodal markers (already checked)
            # and prefer dirs that look like pure text (no processor_config)
            if (dirpath / "processor_config.json").exists():
                return False

    # Must have weight files
    return _has_hf_weights(dirpath)


def _has_hf_weights(dirpath: Path) -> bool:
    has_weights = False
    for pattern in ("*.safetensors", "pytorch_model*.bin", "model*.safetensors"):
        if list(dirpath.glob(pattern)):
            has_weights = True
            break
    return has_weights


def _is_native_vllm_config(cfg: dict) -> bool:
    """True for compressed NVFP4 checkpoints that must stay in HF format."""
    quant = cfg.get("quantization_config") or {}
    groups = quant.get("config_groups") or {}
    formats = {
        str(group.get("format") or "").lower()
        for group in groups.values()
        if isinstance(group, dict)
    }
    has_nvfp4 = (
        str(quant.get("format") or "").lower() == "nvfp4-pack-quantized"
        or "nvfp4-pack-quantized" in formats
    )
    return quant.get("quant_method") == "compressed-tensors" and has_nvfp4


def _hf_format(dirpath: Path) -> str:
    try:
        with open(dirpath / "config.json") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "hf"
    return "vllm" if _is_native_vllm_config(cfg) else "hf"


def _hf_native_features(dirpath: Path) -> tuple[bool, bool]:
    try:
        with open(dirpath / "config.json") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, False
    vision = bool(cfg.get("vision_config") or cfg.get("image_token_id"))
    text = cfg.get("text_config") or cfg
    mtp = bool(
        text.get("mtp_num_hidden_layers")
        or (dirpath / "model_mtp.safetensors").is_file()
    )
    return vision, mtp


def _hf_dir_size(dirpath: Path) -> int:
    total = 0
    for pattern in ("*.safetensors", "pytorch_model*.bin", "*.bin"):
        for f in dirpath.glob(pattern):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _run_scan() -> list[ModelInfo]:
    settings = get_settings()
    root = Path(settings["scan_root"]).expanduser()
    min_size = int(settings["min_model_size_mb"]) * 1024 * 1024
    exclude_dirs = settings.get("scan_exclude_dirs", [])

    raw_ggufs: list[tuple[str, Path, int]] = []
    hf_models: list[ModelInfo] = []
    seen_hf: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if _should_skip_dir(dirpath, exclude_dirs):
            dirnames.clear()
            continue
        dirnames[:] = [
            d
            for d in dirnames
            if (not d.startswith(".")) or d in (".lmstudio", ".cache")
        ]

        # HF model directory
        dpath = Path(dirpath)
        if _is_text_hf_dir(dpath):
            resolved = str(dpath.resolve())
            if resolved not in seen_hf:
                size = _hf_dir_size(dpath)
                if size >= min_size:
                    seen_hf.add(resolved)
                    repo = _hf_hub_repo_from_path(dpath)
                    fmt = _hf_format(dpath)
                    native_vision, mtp_capable = _hf_native_features(dpath)
                    hf_models.append(
                        ModelInfo(
                            name=repo or dpath.name,
                            path=resolved,
                            size_bytes=size,
                            folder=str(dpath.parent),
                            format=fmt,
                            shards=1,
                            alias=repo,
                            has_mmproj=native_vision,
                            mmproj_name="native vLLM processor" if native_vision else None,
                            mtp_capable=mtp_capable,
                            recommended_mtp_draft_n=_recommended_mtp_draft_n(
                                repo or dpath.name
                            ),
                        )
                    )
            # Still scan for GGUFs inside (some folders have both)

        for fname in filenames:
            if not fname.endswith(".gguf"):
                continue
            fpath = Path(dirpath) / fname
            try:
                stat = fpath.stat()
            except OSError:
                continue
            is_shard = _parse_shard(fname) is not None
            if not _is_text_gguf(fname, stat.st_size, min_size, is_shard):
                continue
            raw_ggufs.append((fname, fpath, stat.st_size))

    models = _group_ggufs(raw_ggufs) + hf_models
    # Drop incomplete shard groups that are missing shard 1 when total > 1
    cleaned: list[ModelInfo] = []
    for m in models:
        if m.format == "gguf" and m.shards > 1:
            # path should be shard 00001
            if not SHARD_RE.search(Path(m.path).name):
                cleaned.append(m)
                continue
            parsed = _parse_shard(Path(m.path).name)
            if parsed and parsed[1] != 1:
                # Try to find shard 1 among shard_paths
                for sp in m.shard_paths:
                    p = _parse_shard(Path(sp).name)
                    if p and p[1] == 1:
                        m.path = sp
                        break
        cleaned.append(m)

    cleaned.sort(key=lambda m: m.name.lower())
    return cleaned


def _background_scan(force: bool = False) -> None:
    global _cache, _cache_time, _scanning
    try:
        found = _run_scan()
        with _scan_lock:
            _cache = found
            _cache_time = time.time()
    finally:
        with _scan_lock:
            _scanning = False


def scan_models(force: bool = False) -> tuple[list[ModelInfo], bool]:
    global _scanning

    ttl = 60.0
    now = time.time()

    with _scan_lock:
        if not force and _cache and (now - _cache_time) < ttl:
            return list(_cache), False
        if _scanning and not force:
            return list(_cache), True
        _scanning = True

    if force:
        _background_scan(force=True)
        with _scan_lock:
            return list(_cache), False

    threading.Thread(target=_background_scan, daemon=True).start()
    with _scan_lock:
        return list(_cache), True


def is_scanning() -> bool:
    with _scan_lock:
        return _scanning


def hf_hub_repo_from_path(path: str | Path) -> str | None:
    return _hf_hub_repo_from_path(Path(path))


def find_model(path: str) -> ModelInfo | None:
    items, _ = scan_models()
    for m in items:
        if m.path == path:
            return m
    return None
