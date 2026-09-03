import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".config" / "lemur"
STATE_FILE = CONFIG_DIR / "state.json"


def _detect_llama_server() -> str:
    found = shutil.which("llama-server")
    if found:
        return found
    home = Path.home()
    install_root = Path(
        os.environ.get("LEMUR_INSTALL_ROOT", home / ".local/share/llm-hub")
    ).expanduser()
    installed = sorted(
        install_root.glob("backends/llama.cpp-*/bin/llama-server"), reverse=True
    )
    if installed:
        return str(installed[0])
    for rel in (
        "llama.cpp/build/bin/llama-server",
        "llama.cpp/build-cuda/bin/llama-server",
        "llama.cpp/build-cuda128-safe/bin/llama-server",
        ".local/bin/llama-server",
    ):
        path = home / rel
        if path.is_file():
            return str(path)
    return ""


DEFAULT_LLAMA_SERVER = _detect_llama_server()
_INSTALL_ROOT = Path(
    os.environ.get("LEMUR_INSTALL_ROOT", Path.home() / ".local/share/llm-hub")
).expanduser()
_VLLM_BINS = sorted(_INSTALL_ROOT.glob("backends/vllm-*/bin/vllm"), reverse=True)
DEFAULT_VLLM_BIN = str(_VLLM_BINS[0]) if _VLLM_BINS else ""

DEFAULT_SETTINGS: dict[str, Any] = {
    "llama_server_path": DEFAULT_LLAMA_SERVER,
    "vllm_bin": DEFAULT_VLLM_BIN,
    "scan_root": str(Path.home()),
    "min_model_size_mb": 50,
    "default_ctx": 8192,
    "default_ngl": 999,
    "default_host": "127.0.0.1",
    "port_start": 8080,
    # convert_hf_to_gguf.py can emit Q8 directly; K-quants need a separate
    # llama-quantize pass that the hub does not implement.
    "hf_outtype": "q8_0",
    # Python with torch for convert_hf_to_gguf.py. Empty uses the current Python.
    "hf_convert_python": "",
    "ui_font_size": 15,
    "show_splash_on_startup": True,
    "scan_exclude_dirs": [
        ".cache",
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "comfyui/models",
        ".ollama",
        "site-packages",
    ],
}

CONVERTED_DIR = CONFIG_DIR / "converted"


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict[str, Any]:
    ensure_config_dir()
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"settings": dict(DEFAULT_SETTINGS)}


def save_state(state: dict[str, Any]) -> None:
    ensure_config_dir()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_settings() -> dict[str, Any]:
    state = load_state()
    settings = dict(DEFAULT_SETTINGS)
    settings.update(state.get("settings", {}))
    return settings


def update_settings(updates: dict[str, Any]) -> dict[str, Any]:
    state = load_state()
    settings = get_settings()
    settings.update(updates)
    if "ui_font_size" in settings:
        try:
            settings["ui_font_size"] = max(11, min(22, int(settings["ui_font_size"])))
        except (TypeError, ValueError):
            settings["ui_font_size"] = DEFAULT_SETTINGS["ui_font_size"]
    state["settings"] = settings
    save_state(state)
    return settings


FAVORITE_FIELDS = (
    "model_path",
    "model_name",
    "alias",
    "format",
    "gpu",
    "ctx",
    "ngl",
    "spill",
    "mtp",
    "mtp_draft_n",
    "vision",
)


def get_favorites() -> list[dict[str, Any]]:
    state = load_state()
    items = state.get("favorites") or []
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict) and item.get("id")]


def add_favorite(values: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Save a launch preset. Return the preset and whether it was created."""
    state = load_state()
    items = [
        dict(item)
        for item in (state.get("favorites") or [])
        if isinstance(item, dict) and item.get("id")
    ]
    preset = {key: values.get(key) for key in FAVORITE_FIELDS}
    for item in items:
        if all(item.get(key) == preset.get(key) for key in FAVORITE_FIELDS):
            return item, False

    preset.update(
        {
            "id": uuid.uuid4().hex[:10],
            "created_at": time.time(),
        }
    )
    items.insert(0, preset)
    state["favorites"] = items
    save_state(state)
    return preset, True


def get_favorite(favorite_id: str) -> dict[str, Any] | None:
    return next(
        (item for item in get_favorites() if item.get("id") == favorite_id),
        None,
    )


def delete_favorite(favorite_id: str) -> bool:
    state = load_state()
    items = state.get("favorites") or []
    kept = [
        item
        for item in items
        if not isinstance(item, dict) or item.get("id") != favorite_id
    ]
    if len(kept) == len(items):
        return False
    state["favorites"] = kept
    save_state(state)
    return True
