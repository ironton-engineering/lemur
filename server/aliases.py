"""Stable model ids for llama-server --alias and client configs."""
from __future__ import annotations

import re


def model_alias(name: str) -> str:
    n = name.strip()
    if n.lower().endswith(".gguf"):
        n = n[:-5]
    n = re.sub(r"-\d{5}-of-\d{5}$", "", n, flags=re.IGNORECASE)
    return n or "model"
