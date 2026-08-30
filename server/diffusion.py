"""DiffusionGemma runner — Unsloth-style visual decoder + OpenAI HTTP shim.

Standard llama-server cannot load architecture `diffusion-gemma`. Unsloth's
`llama-diffusion-gemma-visual-server` does the entropy-bound decode; this module
drives it and exposes `/v1/chat/completions` so the Hub proxy/chat keep working.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

CANVAS = 256

DEFAULT_VISUAL_CANDIDATES = (
    Path.home() / ".unsloth/llama.cpp/build/bin/llama-diffusion-gemma-visual-server",
)


def resolve_visual_bin(configured: str | None = None) -> Path | None:
    cands: list[Path] = []
    if configured:
        cands.append(Path(configured).expanduser())
    env = os.environ.get("DG_VISUAL_BIN")
    if env:
        cands.append(Path(env).expanduser())
    cands.extend(DEFAULT_VISUAL_CANDIDATES)
    for p in cands:
        if p.is_file():
            return p.resolve()
    return None


def _set_pdeathsig() -> None:
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:
        pass


class VisualServer:
    """Persistent stdin/stdout driver for llama-diffusion-gemma-visual-server."""

    def __init__(
        self,
        gguf: str,
        *,
        visual_bin: Path,
        gpu: str = "0",
        maxtok: int = 0,
    ):
        self.gguf = gguf
        self.visual_bin = visual_bin
        self.gpu = str(gpu)
        self.maxtok_req = int(maxtok) if 0 < int(maxtok) <= 8192 else 0
        req_dir = "/dev/shm" if Path("/dev/shm").is_dir() else tempfile.gettempdir()
        self.req = str(Path(req_dir) / f"lemur_dg_{os.getpid()}.req")
        self.p: subprocess.Popen | None = None
        self.n_vocab = 0
        self.maxtok = self.maxtok_req
        self._spawn()

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        env["NGL"] = "99"
        env["MAXTOK"] = str(self.maxtok_req)
        bin_dir = str(self.visual_bin.parent)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            bin_dir if not existing else f"{bin_dir}{os.pathsep}{existing}"
        )
        return env

    def _spawn(self) -> None:
        # Keep stdout protocol-clean (stderr carries ggml load logs → Hub).
        self.p = subprocess.Popen(
            [str(self.visual_bin), self.gguf],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=self._env(),
            preexec_fn=_set_pdeathsig if os.name == "posix" else None,
        )
        assert self.p.stdout is not None
        line = self.p.stdout.readline().strip()
        if not line.startswith("READY"):
            raise RuntimeError(f"visual server failed to start: {line!r}")
        parts = line.split()
        self.n_vocab = int(parts[1]) if len(parts) > 1 else 0
        self.maxtok = int(parts[2]) if len(parts) > 2 else self.maxtok_req
        print(line, flush=True)

    def close(self) -> None:
        if not self.p:
            return
        try:
            assert self.p.stdin is not None
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()

    def generate(
        self,
        messages: list[dict],
        *,
        seed: int = 3407,
        max_blocks: int = 8,
        on_commit=None,
        on_stats=None,
    ) -> tuple[str, dict]:
        assert self.p and self.p.stdin and self.p.stdout
        req = {"seed": int(seed), "n_blocks": int(max_blocks), "messages": messages}
        with open(self.req, "w", encoding="utf-8") as f:
            json.dump(req, f, ensure_ascii=False)
        self.p.stdin.write(self.req + "\n")
        self.p.stdin.flush()

        full = ""
        stats: dict[str, Any] = {}
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("visual server closed the stream")
            line = line.rstrip("\n")
            if line == "DONE":
                break
            if line.startswith("ERR"):
                raise RuntimeError(f"visual server error: {line}")
            if line.startswith("STATS"):
                stats = _parse_stats(line)
                if on_stats:
                    on_stats(stats)
                continue
            if line.startswith("F "):
                continue
            if line.startswith("C "):
                parts = line.split(" ", 2)
                if len(parts) >= 3:
                    full = json.loads(parts[2])
                    if on_commit:
                        on_commit(full)
        return full, stats


def _parse_stats(line: str) -> dict[str, Any]:
    """Parse `STATS k=v ...` from the visual server."""
    stats: dict[str, Any] = {}
    for tok in line.split()[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            stats[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            stats[k] = v
    return stats


def timings_from_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    """Build llama-server-shaped timings from diffusion STATS.

    `predicted_per_second` is committed answer tokens / wall — comparable to
    autoregressive gen tok/s (Qwen MTP included). Parallel canvas throughput is
    kept under diffusion_* for the CLI-style internal rate.
    """
    stats = stats or {}

    def rate(n: float, ms: float) -> float:
        return (n / ms * 1000.0) if ms > 0 else 0.0

    prompt_n = int(stats.get("prompt_n", 0) or 0)
    predicted_n = int(stats.get("predicted_n", 0) or 0)
    prep_ms = float(stats.get("prompt_prepare_ms", stats.get("prompt_ms", 0.0)) or 0.0)
    wall_ms = float(stats.get("wall_ms", stats.get("predicted_ms", 0.0)) or 0.0)
    decode_ms = float(stats.get("decode_ms", 0.0) or 0.0)
    steps = int(stats.get("steps", 0) or 0)
    blocks = int(stats.get("blocks", 0) or 0)
    canvas = int(stats.get("canvas", 0) or CANVAS)
    out_tps = rate(predicted_n, wall_ms)
    par_tps = rate(canvas * steps, wall_ms)
    eff_tps = rate(canvas * blocks, wall_ms)
    return {
        "prompt_n": prompt_n,
        "prompt_ms": prep_ms,
        "prompt_per_token_ms": (prep_ms / prompt_n) if prompt_n else 0.0,
        # No AR prefill — omit prompt_per_second so the analyzer doesn't show
        # a meaningless tokenize rate as "prompt t/s".
        "predicted_n": predicted_n,
        "predicted_ms": wall_ms,
        "predicted_per_token_ms": (wall_ms / predicted_n) if predicted_n else 0.0,
        # Output throughput — same meaning as llama-server for AR/MTP models.
        "predicted_per_second": out_tps,
        "cache_n": 0,
        "diffusion": True,
        "diffusion_blocks": blocks,
        "diffusion_steps": steps,
        "diffusion_canvas": canvas,
        "diffusion_decode_ms": decode_ms,
        "diffusion_wall_ms": wall_ms,
        "diffusion_effective_tok_s": eff_tps,
        "diffusion_parallel_tok_s": par_tps,
        "diffusion_output_tok_s": out_tps,
    }


# ── OpenAI-compatible HTTP shim ──────────────────────────────────────────────

_STATE: dict[str, Any] = {}
_LOCK = threading.Lock()
app = FastAPI()


def _slot_busy(busy: bool, *, decoded: int = 0, prompt_n: int = 0) -> None:
    slot = _STATE.setdefault(
        "slot",
        {
            "busy": False,
            "task": 0,
            "decoded": 0,
            "prompt_n": 0,
            "prompt_proc": 0,
            "t_start": None,
        },
    )
    if busy and not slot["busy"]:
        slot["task"] = int(slot.get("task") or 0) + 1
        slot["t_start"] = time.time()
        slot["decoded"] = 0
        slot["prompt_n"] = prompt_n
        slot["prompt_proc"] = prompt_n
    slot["busy"] = busy
    if busy:
        slot["decoded"] = max(int(decoded), int(slot.get("decoded") or 0))
        if prompt_n:
            slot["prompt_n"] = prompt_n
            slot["prompt_proc"] = prompt_n
    else:
        slot["decoded"] = int(decoded) if decoded else int(slot.get("decoded") or 0)


def _chunk(cid: str, created: int, delta: dict, finish=None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": _STATE.get("model_id", "diffusiongemma"),
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _max_blocks(body: dict) -> int:
    mt = body.get("max_tokens") or body.get("max_completion_tokens") or 2048
    try:
        mt = int(mt)
    except (TypeError, ValueError):
        mt = 2048
    return max(1, math.ceil(mt / CANVAS))


@app.get("/health")
def health():
    return {"status": "ok", "model": _STATE.get("model_id")}


@app.get("/props")
def props():
    """Minimal llama-server /props so the Hub analyzer can resolve n_ctx."""
    srv: VisualServer | None = _STATE.get("server")
    n_ctx = int(getattr(srv, "maxtok", 0) or _STATE.get("maxtok") or 8192)
    return {
        "model_alias": _STATE.get("model_id", "diffusiongemma"),
        "model_path": getattr(srv, "gguf", ""),
        "model_ftype": "diffusion-gemma",
        "total_slots": 1,
        "default_generation_settings": {"n_ctx": n_ctx},
    }


@app.get("/slots")
def slots():
    """Minimal llama-server /slots so live gen t/s works during diffusion."""
    slot = _STATE.get("slot") or {}
    busy = bool(slot.get("busy"))
    n_ctx = int(getattr(_STATE.get("server"), "maxtok", 0) or 8192)
    decoded = int(slot.get("decoded") or 0)
    prompt_n = int(slot.get("prompt_n") or 0)
    return [
        {
            "id": 0,
            "id_task": int(slot.get("task") or 0) if busy else -1,
            "is_processing": busy,
            "n_ctx": n_ctx,
            "n_prompt_tokens": prompt_n,
            "n_prompt_tokens_processed": int(slot.get("prompt_proc") or prompt_n),
            "n_prompt_tokens_cache": 0,
            "next_token": [
                {
                    "has_next_token": busy,
                    "n_decoded": decoded,
                }
            ],
        }
    ]


@app.get("/v1/models")
def models():
    mid = _STATE.get("model_id", "diffusiongemma")
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": 1700000000, "owned_by": "lemur"}
        ],
    }


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages") or []
    stream = bool(body.get("stream", False))
    max_blocks = _max_blocks(body)
    seed = int(body.get("seed", 3407))
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    mid = _STATE.get("model_id", "diffusiongemma")
    srv: VisualServer = _STATE["server"]
    loop = asyncio.get_event_loop()

    if not stream:

        def work():
            with _LOCK:
                _slot_busy(True)
                try:
                    text, stats = srv.generate(
                        messages, seed=seed, max_blocks=max_blocks
                    )
                    t = timings_from_stats(stats)
                    _slot_busy(False, decoded=int(t.get("predicted_n") or 0))
                    return text, t
                except Exception:
                    _slot_busy(False)
                    raise

        text, timings = await loop.run_in_executor(None, work)
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": mid,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": int(timings.get("prompt_n") or 0),
                    "completion_tokens": int(timings.get("predicted_n") or 0),
                    "total_tokens": int(timings.get("prompt_n") or 0)
                    + int(timings.get("predicted_n") or 0),
                },
                "timings": timings,
            }
        )

    async def gen():
        q: asyncio.Queue = asyncio.Queue()
        stats_box: dict[str, Any] = {}

        def on_commit(cumulative: str):
            # Rough live token progress so /slots gen t/s moves during the turn.
            approx = max(1, len(cumulative) // 4)
            _slot_busy(True, decoded=approx)
            loop.call_soon_threadsafe(q.put_nowait, ("delta", cumulative))

        def work():
            try:
                with _LOCK:
                    _slot_busy(True)
                    full, stats = srv.generate(
                        messages,
                        seed=seed,
                        max_blocks=max_blocks,
                        on_commit=on_commit,
                        on_stats=stats_box.update,
                    )
                    timings = timings_from_stats(stats or stats_box)
                    _slot_busy(False, decoded=int(timings.get("predicted_n") or 0))
                loop.call_soon_threadsafe(q.put_nowait, ("done", (full, timings)))
            except Exception as exc:
                _slot_busy(False)
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        yield _sse(_chunk(cid, created, {"role": "assistant"}))
        sent = ""
        while True:
            kind, payload = await q.get()
            if kind in ("delta", "done"):
                if kind == "delta":
                    new = payload[len(sent) :]
                    if new:
                        yield _sse(_chunk(cid, created, {"content": new}))
                        sent = payload
                    continue
                full, timings = payload
                new = full[len(sent) :]
                if new:
                    yield _sse(_chunk(cid, created, {"content": new}))
                # llama-server-style final timings chunk (empty choices).
                yield _sse(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": mid,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": int(timings.get("prompt_n") or 0),
                            "completion_tokens": int(timings.get("predicted_n") or 0),
                            "total_tokens": int(timings.get("prompt_n") or 0)
                            + int(timings.get("predicted_n") or 0),
                        },
                        "timings": timings,
                    }
                )
                yield _sse(_chunk(cid, created, {}, finish="stop"))
                yield "data: [DONE]\n\n"
                return
            else:
                yield _sse(
                    _chunk(
                        cid,
                        created,
                        {"content": f"\n[engine error: {payload}]"},
                        finish="stop",
                    )
                )
                yield "data: [DONE]\n\n"
                return

    return StreamingResponse(gen(), media_type="text/event-stream")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DiffusionGemma OpenAI shim for Lemur")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--maxtok", type=int, default=0)
    ap.add_argument("--alias", default="diffusiongemma")
    ap.add_argument("--visual-bin", default="")
    args = ap.parse_args(argv)

    visual = resolve_visual_bin(args.visual_bin or None)
    if visual is None:
        print(
            "error: llama-diffusion-gemma-visual-server not found "
            "(set diffusion_visual_bin / DG_VISUAL_BIN)",
            flush=True,
        )
        sys.exit(1)

    print(
        f"loading diffusion model {args.gguf} via {visual.name} on GPU {args.gpu} ...",
        flush=True,
    )
    _STATE["model_id"] = args.alias
    _STATE["maxtok"] = args.maxtok
    _STATE["server"] = VisualServer(
        args.gguf, visual_bin=visual, gpu=args.gpu, maxtok=args.maxtok
    )
    _slot_busy(False)
    # Match llama-server readiness phrase so Hub marks the instance running.
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
