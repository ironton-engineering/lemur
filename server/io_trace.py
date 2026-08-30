"""Capture live model input/output from hub-proxied traffic."""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict, deque
_lock = threading.Lock()
_traces: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
_by_id: dict[str, dict] = {}
_active: dict[str, str] = {}  # server_id -> trace_id


def _format_chat_input(body: dict) -> str:
    msgs = body.get("messages")
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or "?"
            content = m.get("content")
            if isinstance(content, list):
                bits = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        bits.append(str(c.get("text") or ""))
                    elif isinstance(c, str):
                        bits.append(c)
                content = "\n".join(bits)
            parts.append(f"[{role}]\n{content if content is not None else ''}")
        return "\n\n".join(parts).strip()
    return json.dumps(body, indent=2, ensure_ascii=False)[:20000]


def _format_responses_input(body: dict) -> str:
    inp = body.get("input")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        parts = []
        for item in inp:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                role = item.get("role") or item.get("type") or "item"
                content = item.get("content")
                if isinstance(content, list):
                    texts = []
                    for c in content:
                        if isinstance(c, dict):
                            texts.append(str(c.get("text") or c.get("content") or ""))
                        else:
                            texts.append(str(c))
                    content = "\n".join(texts)
                parts.append(f"[{role}]\n{content if content is not None else json.dumps(item)}")
        return "\n\n".join(parts).strip()
    if inp is not None:
        return json.dumps(inp, indent=2, ensure_ascii=False)[:20000]
    return json.dumps(body, indent=2, ensure_ascii=False)[:20000]


def format_request_input(path: str, body: dict) -> str:
    if "chat/completions" in path or path.endswith("/chat"):
        return _format_chat_input(body)
    if "responses" in path:
        return _format_responses_input(body)
    return _format_chat_input(body) or _format_responses_input(body)


def begin(
    server_id: str,
    *,
    kind: str,
    input_text: str,
    model: str | None = None,
) -> str:
    tid = f"io-{uuid.uuid4().hex[:12]}"
    now = time.time()
    trace = {
        "id": tid,
        "server_id": server_id,
        "kind": kind,
        "model": model,
        "ts": now,
        "updated_at": now,
        "done": False,
        "input": input_text or "",
        "thinking": "",
        "output": "",
    }
    with _lock:
        _by_id[tid] = trace
        _traces[server_id].append(trace)
        _active[server_id] = tid
    return tid


def append(
    server_id: str,
    trace_id: str,
    *,
    thinking: str | None = None,
    output: str | None = None,
) -> None:
    with _lock:
        t = _by_id.get(trace_id)
        if not t or t.get("server_id") != server_id:
            return
        if thinking:
            t["thinking"] = (t.get("thinking") or "") + thinking
        if output:
            t["output"] = (t.get("output") or "") + output
        t["updated_at"] = time.time()


def end(server_id: str, trace_id: str, *, timings: dict | None = None) -> None:
    with _lock:
        t = _by_id.get(trace_id)
        if not t or t.get("server_id") != server_id:
            return
        t["done"] = True
        t["updated_at"] = time.time()
        if timings:
            t["timings"] = timings
        if _active.get(server_id) == trace_id:
            _active.pop(server_id, None)


def get_trace(server_id: str, trace_id: str | None = None) -> dict | None:
    with _lock:
        if trace_id:
            t = _by_id.get(trace_id)
            if t and t.get("server_id") == server_id:
                return dict(t)
            return None
        active = _active.get(server_id)
        if active and active in _by_id:
            return dict(_by_id[active])
        traces = _traces.get(server_id)
        if not traces:
            return None
        # Prefer in-flight even if _active was cleared early by a partial event.
        for t in reversed(traces):
            if not t.get("done"):
                return dict(t)
        return dict(traces[-1])


def list_traces(server_id: str, limit: int = 24) -> list[dict]:
    with _lock:
        traces = list(_traces.get(server_id, ()))
    out = []
    for t in reversed(traces[-limit:]):
        out.append(
            {
                "id": t["id"],
                "ts": t.get("ts"),
                "updated_at": t.get("updated_at"),
                "done": bool(t.get("done")),
                "kind": t.get("kind"),
                "model": t.get("model"),
                "input_chars": len(t.get("input") or ""),
                "thinking_chars": len(t.get("thinking") or ""),
                "output_chars": len(t.get("output") or ""),
                "preview": ((t.get("input") or "")[:120]).replace("\n", " "),
            }
        )
    return out


def nearest_trace(server_id: str, ts: float | None, window_s: float = 120.0) -> dict | None:
    if ts is None:
        return get_trace(server_id)
    with _lock:
        traces = list(_traces.get(server_id, ()))
    best = None
    best_d = None
    for t in traces:
        d = abs(float(t.get("ts") or 0) - ts)
        if d <= window_s and (best_d is None or d < best_d):
            best = t
            best_d = d
    return dict(best) if best else None


def _delta_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)
    if isinstance(value, dict):
        text = value.get("text")
        if text is None:
            text = value.get("content")
        if isinstance(text, str):
            return text
    return ""


def ingest_chat_chunk(server_id: str, trace_id: str, obj: dict) -> None:
    """Parse OpenAI chat.completion.chunk JSON."""
    choices = obj.get("choices") or []
    if not choices:
        return
    delta = choices[0].get("delta") or {}
    thinking_raw = (
        delta.get("reasoning_content")
        or delta.get("reasoning")
        or delta.get("thinking")
    )
    thinking = _delta_text(thinking_raw)
    content = _delta_text(delta.get("content"))
    if thinking:
        append(server_id, trace_id, thinking=thinking)
    if content:
        append(server_id, trace_id, output=content)


def ingest_chat_final(server_id: str, trace_id: str, obj: dict) -> None:
    choices = obj.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        th = _delta_text(msg.get("reasoning_content") or msg.get("reasoning"))
        co = _delta_text(msg.get("content"))
        if th:
            with _lock:
                t = _by_id.get(trace_id)
                if t and not t.get("thinking"):
                    t["thinking"] = th
                    t["updated_at"] = time.time()
        if co:
            with _lock:
                t = _by_id.get(trace_id)
                if t and not t.get("output"):
                    t["output"] = co
                    t["updated_at"] = time.time()
    end(server_id, trace_id, timings=obj.get("timings"))


def ingest_responses_event(server_id: str, trace_id: str, obj: dict) -> None:
    """Parse OpenAI Responses API SSE event object."""
    typ = obj.get("type") or ""
    if typ in (
        "response.reasoning_text.delta",
        "response.reasoning.delta",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_part.delta",
    ):
        delta = obj.get("delta")
        text = _delta_text(delta)
        if text:
            append(server_id, trace_id, thinking=text)
        return
    if typ in (
        "response.output_text.delta",
        "response.content_part.delta",
        "response.function_call_arguments.delta",
    ):
        delta = obj.get("delta")
        text = _delta_text(delta)
        if text:
            # content_part.delta may carry reasoning parts
            part = obj.get("part") if isinstance(obj.get("part"), dict) else {}
            if typ == "response.content_part.delta" and part.get("type") in (
                "reasoning",
                "reasoning_text",
            ):
                append(server_id, trace_id, thinking=text)
            else:
                append(server_id, trace_id, output=text)
        return
    if typ == "response.completed":
        resp = obj.get("response") or {}
        # Backfill from final output items if stream deltas were sparse
        for item in resp.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "reasoning":
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("text"):
                        with _lock:
                            t = _by_id.get(trace_id)
                            if t and c["text"] not in (t.get("thinking") or ""):
                                if not t.get("thinking"):
                                    t["thinking"] = c["text"]
                                    t["updated_at"] = time.time()
            if item.get("type") == "message":
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("text"):
                        with _lock:
                            t = _by_id.get(trace_id)
                            if t and not t.get("output"):
                                t["output"] = c["text"]
                                t["updated_at"] = time.time()
        end(server_id, trace_id, timings=resp.get("usage"))
        return
    # Only whole-response failure ends the trace. Events like
    # response.output_item.done / content_part.done are mid-stream and must
    # not clear the active live pointer.
    if typ == "response.failed":
        end(server_id, trace_id)


def parse_sse_payloads(chunk: bytes) -> list[dict]:
    """Extract JSON objects from an SSE byte chunk."""
    out: list[dict] = []
    if not chunk:
        return out
    for raw_line in chunk.split(b"\n"):
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out
