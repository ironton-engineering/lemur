"""Sync Lemur into a Codex CLI profile (does not touch main config.toml)."""
from __future__ import annotations

import json
from pathlib import Path

from server.aliases import model_alias

CODEX_HOME = Path.home() / ".codex"
PROFILE_NAME = "lemur"
PROFILE_PATH = CODEX_HOME / f"{PROFILE_NAME}.config.toml"
CATALOG_DIR = CODEX_HOME / "model-catalogs"
CATALOG_PATH = CATALOG_DIR / "lemur.json"
PROVIDER_ID = "lemur"
HUB_BASE_URL = "http://127.0.0.1:9000/v1"
DEFAULT_CTX = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 32768

# Codex rejects catalog entries missing these fields (strict serde).
_BASE_INSTRUCTIONS = (
    "You are Codex, a coding agent. Collaborate with the user until their goal "
    "is handled. Prefer rg for search, parallel tool calls when possible, and "
    "apply_patch for edits. Be concise and stay within the task."
)


def codex_command(alias: str | None = None) -> str:
    base = f"codex --profile {PROFILE_NAME}"
    if alias:
        return f"{base} -m {model_alias(alias)}"
    return base


def _compact_limit(ctx: int) -> int:
    ctx = max(1024, int(ctx))
    output_reserve = min(DEFAULT_MAX_OUTPUT_TOKENS, ctx // 4)
    safety_margin = min(1024, ctx // 32)
    return max(512, ctx - output_reserve - safety_margin)


def _catalog_entry(slug: str, ctx: int) -> dict:
    ctx = max(1024, min(1048576, int(ctx)))
    compact = _compact_limit(ctx)
    trunc = max(1024, min(ctx // 2, 16000))
    return {
        "slug": slug,
        "display_name": slug,
        "description": f"Local model via Lemur (ctx={ctx}).",
        "context_window": ctx,
        "max_context_window": ctx,
        "auto_compact_token_limit": compact,
        "base_instructions": _BASE_INSTRUCTIONS,
        "model_messages": {
            "instructions_template": _BASE_INSTRUCTIONS + "\n\n{{ personality }}\n",
            "instructions_variables": {
                "personality_default": "",
                "personality_friendly": "",
                "personality_pragmatic": "",
            },
        },
        "input_modalities": ["text"],
        "supported_in_api": True,
        "visibility": "list",
        "shell_type": "shell_command",
        "supports_parallel_tool_calls": True,
        "supports_reasoning_summaries": False,
        "default_reasoning_summary": "none",
        "reasoning_summary_format": None,
        "default_reasoning_level": None,
        "supported_reasoning_levels": [],
        "prefer_websockets": False,
        "support_verbosity": True,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text",
        "supports_image_detail_original": False,
        "truncation_policy": {"mode": "tokens", "limit": trunc},
        "minimal_client_version": "0.124.0",
        "priority": 0,
        "include_skills_usage_instructions": True,
        "experimental_supported_tools": [],
        "supports_search_tool": False,
        "service_tiers": [],
        "additional_speed_tiers": [],
        "default_service_tier": None,
        "availability_nux": None,
        "upgrade": None,
        "available_in_plans": [
            "free",
            "plus",
            "pro",
            "team",
            "business",
            "enterprise",
        ],
    }


def write_catalog(models: list[tuple[str, int]]) -> Path:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    by_slug: dict[str, dict] = {}
    for slug, ctx in models:
        s = model_alias(slug)
        if not s:
            continue
        by_slug[s] = _catalog_entry(s, ctx)
    CATALOG_PATH.write_text(json.dumps({"models": list(by_slug.values())}, indent=2) + "\n")
    # Guard against incomplete writes (Codex hard-fails on missing fields)
    loaded = json.loads(CATALOG_PATH.read_text())
    for m in loaded.get("models", []):
        if "base_instructions" not in m or "support_verbosity" not in m:
            raise RuntimeError(
                f"Refusing incomplete Codex catalog entry for {m.get('slug')}"
            )
    return CATALOG_PATH


def sync_profile(
    *,
    model: str | None = None,
    context_window: int | None = None,
    catalog_models: list[tuple[str, int]] | None = None,
    base_url: str = HUB_BASE_URL,
) -> dict:
    CODEX_HOME.mkdir(parents=True, exist_ok=True)
    alias = model_alias(model) if model else None
    if not alias:
        if PROFILE_PATH.exists():
            for line in PROFILE_PATH.read_text().splitlines():
                if line.startswith("model = "):
                    alias = line.split("=", 1)[1].strip().strip('"')
                    break
        alias = alias or "local"

    ctx = int(context_window) if context_window else 0
    if not ctx and PROFILE_PATH.exists():
        for line in PROFILE_PATH.read_text().splitlines():
            if line.startswith("model_context_window = "):
                try:
                    ctx = int(line.split("=", 1)[1].strip())
                except ValueError:
                    ctx = 0
                break
    if not ctx:
        ctx = DEFAULT_CTX
    ctx = max(1024, min(1048576, ctx))
    compact_at = _compact_limit(ctx)

    entries = list(catalog_models or [])
    entries.append((alias, ctx))
    catalog = write_catalog(entries)

    text = (
        f'model = "{alias}"\n'
        f'model_provider = "{PROVIDER_ID}"\n'
        'model_reasoning_effort = "none"\n'
        f"model_context_window = {ctx}\n"
        f"model_auto_compact_token_limit = {compact_at}\n"
        f'model_catalog_json = "{catalog}"\n'
        "\n"
        f"[model_providers.{PROVIDER_ID}]\n"
        'name = "Lemur"\n'
        f'base_url = "{base_url}"\n'
        'wire_api = "responses"\n'
        "stream_idle_timeout_ms = 300000\n"
        "requires_openai_auth = false\n"
    )
    PROFILE_PATH.write_text(text)
    return {
        "path": str(PROFILE_PATH),
        "profile": PROFILE_NAME,
        "base_url": base_url,
        "model": alias,
        "model_context_window": ctx,
        "model_auto_compact_token_limit": compact_at,
        "model_catalog_json": str(catalog),
        "command": codex_command(alias),
    }


def sync_from_hub(
    running: list[dict],
    default: str | None = None,
) -> dict:
    catalog_models: list[tuple[str, int]] = []
    for s in running:
        alias = model_alias(s.get("alias") or s.get("model_name") or "")
        if not alias:
            continue
        c = int(s.get("ctx") or 0) or DEFAULT_CTX
        catalog_models.append((alias, c))

    model = default
    ctx: int | None = None
    if not model and catalog_models:
        model = catalog_models[0][0]
    if model:
        want = model_alias(model)
        for alias, c in catalog_models:
            if alias == want:
                ctx = c
                break
    if ctx is None and catalog_models:
        ctx = catalog_models[0][1]
    return sync_profile(
        model=model,
        context_window=ctx,
        catalog_models=catalog_models,
    )
