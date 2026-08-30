"""Live process analyzer for running llama-server instances."""
from __future__ import annotations

import math
import re
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import httpx

from server import processes

_PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*"
    r"([\d.]+)\s*tokens per second",
    re.I,
)
_EVAL_RE = re.compile(
    r"(?<!prompt )(?<!prompt  )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*"
    r"([\d.]+)\s*tokens per second",
    re.I,
)
_TOTAL_RE = re.compile(
    r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",
    re.I,
)
_LIVE_TG_RE = re.compile(
    r"n_decoded\s*=\s*(\d+).*?\btg\s*=\s*([\d.]+)\s*t/s(?:.*?tg_3s\s*=\s*([\d.]+)\s*t/s)?",
    re.I,
)
_TASK_RE = re.compile(r"task\s+(\d+)", re.I)
_SLOT_RE = re.compile(r"id\s+(\d+)\s*\|", re.I)
_RELEASE_RE = re.compile(
    r"stop processing:\s*n_tokens\s*=\s*(\d+).*truncated\s*=\s*(\d+)",
    re.I,
)
# Per-server rolling query history (also used for orphans with empty logs)
_queries: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
_slot_state: dict[str, dict[int, dict]] = defaultdict(dict)
_load_hints: dict[str, list[str]] = defaultdict(list)
_live_tps: dict[str, dict[str, float]] = defaultdict(dict)
# Analyzer poll time-series (UI shows last 30 minutes)
_HISTORY_WINDOW_S = 30 * 60
_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
_HISTORY_MIN_INTERVAL_S = 1.0
_MAX_DISPLAY_TPS = 1000.0


def _sane_tps(value: Any) -> float | None:
    """Return a display-safe throughput sample or reject an invalid spike."""
    if value is None:
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate) or rate <= 0.0 or rate > _MAX_DISPLAY_TPS:
        return None
    return rate


def note_log_line(server_id: str, line: str) -> None:
    """Call from process log reader to capture timings + load hints."""
    lower = line.lower()
    if any(
        k in lower
        for k in (
            "offload",
            "buffer type",
            "model size",
            "gpu_layers",
            "kv cache",
            "device",
            "mmap",
            "tensor",
            "split",
            "cuda",
            "layer",
        )
    ):
        hints = _load_hints[server_id]
        if line not in hints:
            hints.append(line)
            if len(hints) > 30:
                del hints[:-30]

    if m := _LIVE_TG_RE.search(line):
        live = _live_tps[server_id]
        rate = _sane_tps(m.group(3) or m.group(2))
        if rate is not None:
            live["gen_tps"] = rate
            live["ts"] = time.time()

    task_m = _TASK_RE.search(line)
    if not task_m:
        return
    task = int(task_m.group(1))
    slot_m = _SLOT_RE.search(line)
    slot = int(slot_m.group(1)) if slot_m else None
    q = _find_or_create_query(server_id, task, slot)

    if m := _PROMPT_RE.search(line):
        q["prompt_ms"] = float(m.group(1))
        q["prompt_tokens"] = int(m.group(2))
        q["prompt_tps"] = _sane_tps(m.group(3))
        q["ts"] = time.time()
        q["source"] = "log"
    elif m := _EVAL_RE.search(line):
        q["gen_ms"] = float(m.group(1))
        q["gen_tokens"] = int(m.group(2))
        q["gen_tps"] = _sane_tps(m.group(3))
        q["ts"] = time.time()
        q["source"] = "log"
        q["done"] = True
    elif m := _TOTAL_RE.search(line):
        q["total_ms"] = float(m.group(1))
        q["total_tokens"] = int(m.group(2))
        q["ts"] = time.time()
    elif m := _RELEASE_RE.search(line):
        q["n_tokens"] = int(m.group(1))
        q["truncated"] = int(m.group(2)) != 0
        q["ts"] = time.time()
        q["done"] = True


def note_api_timings(
    server_id: str,
    timings: dict | None,
    *,
    slot: int | None = None,
) -> None:
    """Record prompt/gen tok/s from llama-server response `timings` object."""
    if not timings or not isinstance(timings, dict):
        return
    prompt_tps = _sane_tps(timings.get("prompt_per_second"))
    gen_tps = _sane_tps(timings.get("predicted_per_second"))
    if prompt_tps is None and gen_tps is None:
        return
    q = {
        "task": f"api-{int(time.time() * 1000)}",
        "slot": slot,
        "ts": time.time(),
        "done": True,
        "source": "api",
        "prompt_tokens": timings.get("prompt_n"),
        "gen_tokens": timings.get("predicted_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "gen_ms": timings.get("predicted_ms"),
        "cache_tokens": timings.get("cache_n"),
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "diffusion": bool(timings.get("diffusion")),
        "diffusion_parallel_tok_s": timings.get("diffusion_parallel_tok_s"),
    }
    _queries[server_id].append(q)
    live = _live_tps[server_id]
    if gen_tps is not None:
        live["gen_tps"] = gen_tps
        live["ts"] = time.time()
    if prompt_tps is not None:
        live["prompt_tps"] = prompt_tps
        live["ts"] = time.time()
    if timings.get("diffusion"):
        live["diffusion"] = 1.0
    if timings.get("draft_n"):
        live["draft_n"] = float(timings["draft_n"])
        if timings.get("draft_n_accepted") is not None:
            live["draft_n_accepted"] = float(timings["draft_n_accepted"])


def _find_or_create_query(server_id: str, task: int, slot: int | None) -> dict:
    for q in reversed(_queries[server_id]):
        if q.get("task") == task:
            if slot is not None:
                q["slot"] = slot
            return q
    q = {
        "task": task,
        "slot": slot,
        "ts": time.time(),
        "done": False,
        "source": "log",
    }
    _queries[server_id].append(q)
    return q


def _query_richness(q: dict) -> int:
    score = len(q)
    if q.get("prompt_tps") is not None:
        score += 10
    if q.get("gen_tps") is not None:
        score += 10
    return score


def _tps_from_state(st: dict, now: float) -> tuple[float | None, float | None]:
    """Estimate prompt/gen tok/s from wall-clock markers on a slot sample."""
    prompt_tps = st.get("prompt_tps")
    gen_tps = st.get("gen_tps")
    pt = int(st.get("prompt_proc") or st.get("prompt_tok") or 0)
    gt = int(st.get("decoded") or 0)
    t_start = st.get("t_start")
    t_gen = st.get("t_gen_start")
    t_end = st.get("t_end") or now

    if prompt_tps is None and t_start and pt > 0:
        t_prompt_end = t_gen if t_gen and t_gen > t_start else t_end
        dt = t_prompt_end - t_start
        if dt > 0.02:
            prompt_tps = pt / dt

    if gen_tps is None and gt > 0:
        if t_gen and t_end > t_gen:
            dt = t_end - t_gen
            if dt > 0.02:
                gen_tps = gt / dt
        elif t_start and t_end > t_start and not t_gen:
            dt = t_end - t_start
            if dt > 0.02:
                gen_tps = gt / dt
    return _sane_tps(prompt_tps), _sane_tps(gen_tps)


def _proc_rss_mib(pid: int | None) -> float | None:
    if not pid:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _nvidia_snapshot(pid: int | None) -> tuple[list[dict], list[dict]]:
    """Return process VRAM and all-GPU state using two nvidia-smi calls."""
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if gpu_result.returncode != 0:
            return [], []

        process_used: dict[str, float] = {}
        if pid:
            apps = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in apps.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3 and int(float(parts[0])) == pid:
                    process_used[parts[1]] = process_used.get(parts[1], 0.0) + float(
                        parts[2]
                    )

        devices = []
        all_gpus = []
        for line in gpu_result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",", 6)]
            if len(parts) < 7:
                continue
            idx = int(parts[0])
            uuid = parts[1]
            used = process_used.get(uuid, 0.0)
            row = {
                "gpu": idx,
                "name": parts[2],
                "used_mib": used,
                "total_mib": float(parts[3]),
                "free_mib": float(parts[4]),
                "util_gpu": float(parts[5]),
                "util_mem": float(parts[6]),
            }
            all_gpus.append(row)
            if uuid in process_used:
                devices.append(dict(row))
        return devices, all_gpus
    except Exception:
        return [], []


def _fetch_json(url: str, timeout: float = 2.0) -> Any | None:
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            if r.status_code == 200:
                return r.json()
    except Exception:
        return None
    return None


def _append_slot_query(server_id: str, sid: int, st: dict, now: float) -> None:
    task = st.get("task")
    if task in (None, -1):
        return
    prompt_tps, gen_tps = _tps_from_state(st, now)
    pt = int(st.get("prompt_proc") or st.get("prompt_tok") or 0)
    gt = int(st.get("decoded") or 0)
    prompt_tok = int(st.get("prompt_tok") or 0)
    prompt_cache = int(st.get("prompt_cache") or 0)
    entry = {
        "task": task,
        "slot": sid,
        "ts": now,
        "done": True,
        "source": "slot",
        "prompt_tokens": pt,
        "cache_tokens": prompt_cache,
        "gen_tokens": gt,
        "n_tokens": pt + gt,
        "prompt_tps": round(prompt_tps, 1) if prompt_tps is not None else None,
        "gen_tps": round(gen_tps, 1) if gen_tps is not None else None,
    }
    if prompt_tok:
        entry["cache_hit_pct"] = round(100.0 * prompt_cache / prompt_tok, 1)
    _queries[server_id].append(entry)
    live = _live_tps[server_id]
    if gen_tps is not None:
        live["gen_tps"] = gen_tps
        live["ts"] = now
    if prompt_tps is not None:
        live["prompt_tps"] = prompt_tps
        live["ts"] = now


def _update_from_slots(server_id: str, slots: list[dict]) -> None:
    prev = _slot_state[server_id]
    now = time.time()
    live_gen: list[float] = []
    live_prompt: list[float] = []

    for s in slots:
        sid = int(s.get("id", -1))
        task = s.get("id_task")
        processing = bool(s.get("is_processing"))
        prompt_proc = int(s.get("n_prompt_tokens_processed") or 0)
        prompt_cache = int(s.get("n_prompt_tokens_cache") or 0)
        prompt_tok = int(s.get("n_prompt_tokens") or 0)
        decoded = 0
        nt = s.get("next_token") or []
        if nt and isinstance(nt, list):
            decoded = int(nt[0].get("n_decoded") or 0)

        old = prev.get(sid) or {}

        if old and old.get("task") != task and old.get("task") not in (None, -1):
            fin = dict(old)
            fin["t_end"] = now
            _append_slot_query(server_id, sid, fin, now)
        elif old and old.get("processing") and not processing:
            fin = {
                **old,
                "task": task,
                "prompt_proc": prompt_proc,
                "prompt_cache": prompt_cache,
                "prompt_tok": prompt_tok,
                "decoded": decoded,
                "t_end": now,
            }
            if fin.get("t_gen_start") is None and decoded > 0:
                fin["t_gen_start"] = old.get("t_start")
            _append_slot_query(server_id, sid, fin, now)

        st: dict[str, Any] = {
            "task": task,
            "processing": processing,
            "prompt_proc": prompt_proc,
            "prompt_cache": prompt_cache,
            "prompt_tok": prompt_tok,
            "decoded": decoded,
            "t_start": old.get("t_start"),
            "t_gen_start": old.get("t_gen_start"),
            "last_decoded": old.get("last_decoded"),
            "last_ts": old.get("last_ts"),
        }

        if processing:
            same_task = old.get("task") == task and old.get("processing")
            if not same_task:
                st["t_start"] = now
                st["t_gen_start"] = now if decoded > 0 else None
            else:
                st["t_start"] = old.get("t_start") or now
                st["t_gen_start"] = old.get("t_gen_start")
                if st["t_gen_start"] is None and decoded > 0:
                    st["t_gen_start"] = now
                prev_dec = int(old.get("last_decoded") or old.get("decoded") or 0)
                prev_ts = float(old.get("last_ts") or 0)
                if decoded > prev_dec and prev_ts and now > prev_ts:
                    inst = (decoded - prev_dec) / (now - prev_ts)
                    inst = _sane_tps(inst)
                    if inst is not None:
                        live_gen.append(inst)
                        st["gen_tps"] = inst
                if decoded == 0 and st.get("t_start") and prompt_proc > 0:
                    dt = now - float(st["t_start"])
                    if dt > 0.05:
                        prompt_rate = _sane_tps(prompt_proc / dt)
                        if prompt_rate is not None:
                            live_prompt.append(prompt_rate)
                            st["prompt_tps"] = prompt_rate
            st["last_decoded"] = decoded
            st["last_ts"] = now
        else:
            st["t_start"] = None
            st["t_gen_start"] = None
            st["last_decoded"] = decoded
            st["last_ts"] = now

        prev[sid] = st

    live = _live_tps[server_id]
    if live_gen:
        live["gen_tps"] = sum(live_gen) / len(live_gen)
        live["ts"] = now
    if live_prompt:
        live["prompt_tps"] = sum(live_prompt) / len(live_prompt)
        live["ts"] = now


def analyze_server(server: processes.ServerInstance) -> dict:
    connect_host = "127.0.0.1" if server.host == "0.0.0.0" else server.host
    base = f"http://{connect_host}:{server.port}"
    props = _fetch_json(f"{base}/props") or {}
    slots_raw = _fetch_json(f"{base}/slots")
    slots = slots_raw if isinstance(slots_raw, list) else []

    if slots:
        _update_from_slots(server.id, slots)

    n_ctx = int(
        (props.get("default_generation_settings") or {}).get("n_ctx")
        or server.ctx
        or 0
    )
    slot_views = []
    ctx_used = 0
    cache_hits = 0
    busy = 0
    for s in slots:
        prompt_tok = int(s.get("n_prompt_tokens") or 0)
        prompt_proc = int(s.get("n_prompt_tokens_processed") or 0)
        prompt_cache = int(s.get("n_prompt_tokens_cache") or 0)
        decoded = 0
        nt = s.get("next_token") or []
        if nt and isinstance(nt, list):
            decoded = int(nt[0].get("n_decoded") or 0)
        # Prefer filled context estimate
        fill = max(prompt_proc, prompt_tok, decoded)
        if s.get("is_processing"):
            fill = max(prompt_tok, prompt_proc) + decoded
            busy += 1
        ctx_used = max(ctx_used, fill)
        cache_hits += prompt_cache
        slot_views.append(
            {
                "id": s.get("id"),
                "busy": bool(s.get("is_processing")),
                "task": s.get("id_task"),
                "n_ctx": int(s.get("n_ctx") or n_ctx),
                "prompt_tokens": prompt_tok,
                "prompt_processed": prompt_proc,
                "cache_tokens": prompt_cache,
                "decoded": decoded,
                "fill": fill,
                "fill_pct": round(100.0 * fill / max(n_ctx, 1), 1),
                "cache_hit_pct": (
                    round(100.0 * prompt_cache / prompt_tok, 1) if prompt_tok else 0.0
                ),
            }
        )

    # Recent queries (newest first), dedupe by task keeping richest (prefer timed)
    by_task: dict[Any, dict] = {}
    for q in _queries[server.id]:
        t = q.get("task")
        prev = by_task.get(t)
        if not prev or _query_richness(q) >= _query_richness(prev):
            by_task[t] = dict(q)
    recent = sorted(by_task.values(), key=lambda x: x.get("ts") or 0, reverse=True)[:12]
    for q in recent:
        q["prompt_tps"] = _sane_tps(q.get("prompt_tps"))
        q["gen_tps"] = _sane_tps(q.get("gen_tps"))
        pt = q.get("prompt_tokens") or 0
        gt = q.get("gen_tokens") or 0
        q["context_tokens"] = q.get("n_tokens") or (pt + gt)
        q["ctx_pct"] = round(100.0 * (q["context_tokens"] or 0) / max(n_ctx, 1), 1)
        if q.get("ts"):
            q["age_s"] = round(time.time() - q["ts"], 1)

    # Speed summary: recent timed queries, fall back to live slot estimate
    prompt_tps = [q["prompt_tps"] for q in recent if q.get("prompt_tps")]
    gen_tps = [q["gen_tps"] for q in recent if q.get("gen_tps")]
    live = _live_tps.get(server.id) or {}
    live["prompt_tps"] = _sane_tps(live.get("prompt_tps"))
    live["gen_tps"] = _sane_tps(live.get("gen_tps"))
    live_age = time.time() - float(live.get("ts") or 0)
    live_fresh = live_age < 10.0
    prompt_tps_avg = (
        round(sum(prompt_tps) / len(prompt_tps), 1)
        if prompt_tps
        else (round(live["prompt_tps"], 1) if live_fresh and live.get("prompt_tps") else None)
    )
    gen_tps_avg = (
        round(sum(gen_tps) / len(gen_tps), 1)
        if gen_tps
        else (round(live["gen_tps"], 1) if live_fresh and live.get("gen_tps") else None)
    )

    # Prefer authoritative API timings over slot wall-clock when both exist.
    api_gen = [
        q["gen_tps"]
        for q in recent
        if q.get("source") == "api" and q.get("gen_tps") is not None
    ]
    api_prompt = [
        q["prompt_tps"]
        for q in recent
        if q.get("source") == "api" and q.get("prompt_tps") is not None
    ]
    if api_gen:
        gen_tps_avg = round(sum(api_gen) / len(api_gen), 1)
    if api_prompt:
        prompt_tps_avg = round(sum(api_prompt) / len(api_prompt), 1)

    draft_n = None
    draft_acc = None
    for q in recent:
        if q.get("draft_n"):
            draft_n = int(q["draft_n"])
            if q.get("draft_n_accepted") is not None:
                draft_acc = int(q["draft_n_accepted"])
            break
    is_diffusion = bool(live.get("diffusion")) or any(
        q.get("diffusion") for q in recent
    )
    diffusion_parallel = None
    for q in recent:
        if q.get("diffusion_parallel_tok_s") is not None:
            diffusion_parallel = round(float(q["diffusion_parallel_tok_s"]), 1)
            break

    devices, all_gpus = _nvidia_snapshot(server.pid)
    claimed_idxs = {
        int(i)
        for i in (server.devices or str(server.gpu)).split(",")
        if i.strip().isdigit()
    }
    for row in all_gpus:
        row["claimed"] = row["gpu"] in claimed_idxs

    rss = _proc_rss_mib(server.pid)
    model_size_mib = None
    try:
        p = Path(server.model_path)
        if p.is_file():
            model_size_mib = p.stat().st_size / (1024 * 1024)
    except OSError:
        pass

    vram_total = sum(d["used_mib"] for d in devices)
    now = time.time()
    ctx_pct = round(100.0 * ctx_used / max(n_ctx, 1), 1)
    claimed = [
        g
        for g in all_gpus
        if g.get("claimed")
    ]
    util_vals = [g["util_gpu"] for g in claimed if g.get("util_gpu") is not None]
    gpu_util = round(sum(util_vals) / len(util_vals), 1) if util_vals else None

    # Prefer live speeds while busy; otherwise keep last completed averages.
    sample_gen = None
    sample_prompt = None
    if busy and live_fresh and live.get("gen_tps") is not None:
        sample_gen = round(float(live["gen_tps"]), 2)
    elif gen_tps_avg is not None and busy:
        sample_gen = gen_tps_avg
    if busy and live_fresh and live.get("prompt_tps") is not None:
        sample_prompt = round(float(live["prompt_tps"]), 2)
    elif prompt_tps_avg is not None and busy:
        sample_prompt = prompt_tps_avg

    hist = _history[server.id]
    for old_sample in hist:
        old_sample["gen_tps"] = _sane_tps(old_sample.get("gen_tps"))
        old_sample["prompt_tps"] = _sane_tps(old_sample.get("prompt_tps"))
    if not hist or (now - float(hist[-1].get("ts") or 0)) >= _HISTORY_MIN_INTERVAL_S:
        hist.append(
            {
                "ts": now,
                "ctx_pct": ctx_pct,
                "ctx_used": ctx_used,
                "n_ctx": n_ctx,
                "gen_tps": sample_gen,
                "prompt_tps": sample_prompt,
                "vram_mib": round(vram_total, 0),
                "rss_mib": round(rss, 0) if rss is not None else None,
                "gpu_util": gpu_util,
                "busy": bool(busy),
            }
        )
    cutoff = now - _HISTORY_WINDOW_S
    while hist and float(hist[0].get("ts") or 0) < cutoff:
        hist.popleft()

    last_q = recent[0] if recent else None
    return {
        "server": server.to_dict(),
        "props": {
            "model_alias": props.get("model_alias") or server.alias,
            "model_path": props.get("model_path") or server.model_path,
            "model_ftype": props.get("model_ftype"),
            "n_ctx": n_ctx,
            "total_slots": props.get("total_slots") or len(slots),
            "build_info": props.get("build_info"),
            "is_sleeping": props.get("is_sleeping"),
        },
        "kpis": {
            "n_ctx": n_ctx,
            "ctx_used": ctx_used,
            "ctx_pct": ctx_pct,
            "slots_busy": busy,
            "slots_total": len(slots) or int(props.get("total_slots") or 0),
            "prompt_tps_avg": prompt_tps_avg,
            "gen_tps_avg": gen_tps_avg,
            "vram_mib": round(vram_total, 0),
            "rss_mib": round(rss, 0) if rss is not None else None,
            "model_size_mib": round(model_size_mib, 0) if model_size_mib else None,
            "cache_tokens": cache_hits,
            "gpu_util": gpu_util,
            "busy": bool(busy),
            "diffusion": is_diffusion,
            "diffusion_parallel_tok_s": diffusion_parallel,
            "draft_n": draft_n,
            "draft_n_accepted": draft_acc,
            "mtp": bool(server.mtp) or (draft_n is not None and draft_n > 0),
        },
        "gpus": all_gpus,
        "devices_claimed": server.devices or str(server.gpu),
        "spill": server.spill,
        "slots": slot_views,
        "queries": recent,
        "last_query": last_q,
        "history": list(hist),
        "load_hints": list(_load_hints.get(server.id, []))[-12:],
        "ts": now,
    }


def analyze(server_id: str | None = None) -> dict:
    servers = [
        s
        for s in processes.list_servers()
        if s.status in ("running", "starting")
    ]
    if not servers:
        return {"servers": [], "active": None, "error": "no running servers"}
    chosen = None
    if server_id:
        chosen = next((s for s in servers if s.id == server_id), None)
    if not chosen:
        chosen = servers[0]
    snap = analyze_server(chosen)
    return {
        "servers": [s.to_dict() for s in servers],
        "active": snap,
    }
