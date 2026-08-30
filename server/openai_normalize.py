"""Normalize OpenAI-shaped bodies for llama.cpp chat templates.

Codex Responses requests send top-level `instructions` plus `developer`
messages (including mid-history after context compaction). llama.cpp turns
`instructions` into a system message and maps `developer` → `system`, which
Qwen-style Jinja templates reject unless the only system message is first.
"""
from __future__ import annotations

from typing import Any


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for c in content:
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, dict):
            t = c.get("text")
            if t is None:
                t = c.get("content")
            if t is not None:
                parts.append(str(t))
    return "\n".join(parts)


def _fold_responses(body: dict[str, Any]) -> dict[str, Any]:
    inp = body.get("input")
    if not isinstance(inp, list):
        return body

    parts: list[str] = []
    instr = body.get("instructions")
    if isinstance(instr, str) and instr.strip():
        parts.append(instr.strip())

    rest: list[Any] = []
    folded = False
    for item in inp:
        if isinstance(item, dict) and item.get("role") in ("developer", "system"):
            text = _content_text(item.get("content")).strip()
            if text:
                parts.append(text)
            folded = True
        else:
            rest.append(item)

    if not folded:
        return body

    out = dict(body)
    if parts:
        out["instructions"] = "\n\n".join(parts)
    elif "instructions" in out:
        del out["instructions"]
    out["input"] = rest
    return out


def _merge_chat_systems(body: dict[str, Any]) -> dict[str, Any]:
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return body

    texts: list[str] = []
    rest: list[Any] = []
    found = 0
    for m in msgs:
        if isinstance(m, dict) and m.get("role") in ("system", "developer"):
            text = _content_text(m.get("content")).strip()
            if text:
                texts.append(text)
            found += 1
        else:
            rest.append(m)

    if found < 2 and not any(
        isinstance(m, dict) and m.get("role") == "developer" for m in msgs
    ):
        return body
    if not texts:
        return body

    out = dict(body)
    out["messages"] = [{"role": "system", "content": "\n\n".join(texts)}] + rest
    return out


def normalize_openai_body(path: str, body: dict[str, Any]) -> dict[str, Any]:
    if "responses" in path:
        return _fold_responses(body)
    if "chat/completions" in path:
        return _merge_chat_systems(body)
    return body
