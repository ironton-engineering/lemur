"""Dream / Dream-Coder runner — Transformers diffusion_generate + OpenAI shim.

Dream is a discrete diffusion LLM. Official inference uses Hugging Face
Transformers (`diffusion_generate`), not llama-server. This module keeps the
model resident and exposes the same OpenAI-shaped endpoints the Hub already
proxies for llama-server / DiffusionGemma.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Official Dream-Coder defaults (README).
DEFAULT_MAX_NEW = 768
DEFAULT_STEPS = 768
DEFAULT_TEMP = 0.1
DEFAULT_TOP_P = 0.95
DEFAULT_ALG = "entropy"

_STATE: dict[str, Any] = {}
_LOCK = threading.Lock()

app = FastAPI()


def _slot_busy(
    busy: bool,
    *,
    decoded: int = 0,
    prompt_n: int = 0,
    prompt_proc: int | None = None,
    task: int = 1,
) -> None:
    _STATE["slot"] = {
        "busy": busy,
        "decoded": int(decoded),
        "prompt_n": int(prompt_n),
        "prompt_proc": int(prompt_proc if prompt_proc is not None else prompt_n),
        "task": int(task),
    }


class DreamEngine:
    """Persistent Transformers Dream model on one CUDA device."""

    def __init__(self, model_path: str, *, gpu: str = "0"):
        # Must set before importing/initializing CUDA in this process.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_path = str(model_path)
        self.gpu = str(gpu)
        print(f"loading Dream weights from {self.model_path} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = self.model.to("cuda").eval()
        cfg = self.model.config
        self.n_ctx = int(getattr(cfg, "max_position_embeddings", 2048) or 2048)
        self.mask_id = int(getattr(cfg, "mask_token_id", -1) or -1)
        print(
            f"Dream ready n_ctx={self.n_ctx} dtype=bfloat16 device=cuda:0 "
            f"(CVD={gpu})",
            flush=True,
        )

    def _decode_new(self, prompt_ids: list[int], seq_ids: list[int]) -> str:
        tok = self.tokenizer
        new_ids = seq_ids[len(prompt_ids) :]
        if self.mask_id >= 0:
            new_ids = [t for t in new_ids if t != self.mask_id]
        text = tok.decode(new_ids, skip_special_tokens=False)
        eos = tok.eos_token or ""
        if eos and eos in text:
            text = text.split(eos)[0]
        # Common Dream stop / pad leftovers.
        for stop in ("<|endoftext|>", "<|im_end|>"):
            if stop in text:
                text = text.split(stop)[0]
        return text.strip()

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW,
        steps: int | None = None,
        temperature: float = DEFAULT_TEMP,
        top_p: float = DEFAULT_TOP_P,
        seed: int | None = None,
        on_partial: Callable[[str, int], None] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        import torch

        max_new_tokens = max(1, min(int(max_new_tokens), self.n_ctx - 1))
        steps = int(steps) if steps is not None else max_new_tokens
        steps = max(1, min(steps, max_new_tokens))

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        prompt_ids = input_ids[0].tolist()
        prompt_n = len(prompt_ids)

        t0 = time.perf_counter()

        def hook(step, x, logits):
            if on_partial is None or x is None:
                return x
            try:
                # Intermediate canvas may still contain mask tokens.
                text = self._decode_new(prompt_ids, x[0].tolist())
                # Rough committed count: non-mask tokens past the prompt.
                if self.mask_id >= 0:
                    decoded_n = int((x[0, prompt_n:] != self.mask_id).sum().item())
                else:
                    decoded_n = max(1, len(text) // 4)
                on_partial(text, decoded_n)
            except Exception:
                pass
            return x

        with torch.inference_mode():
            out = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=steps,
                temperature=float(temperature),
                top_p=float(top_p),
                alg=DEFAULT_ALG,
                alg_temp=0.0,
                generation_tokens_hook_func=hook,
            )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        seq = out.sequences[0].tolist()
        text = self._decode_new(prompt_ids, seq)
        predicted_n = max(0, len(seq) - prompt_n)
        # Prefer non-mask count when available.
        if self.mask_id >= 0:
            predicted_n = sum(1 for t in seq[prompt_n:] if t != self.mask_id)
        stats = {
            "prompt_n": prompt_n,
            "predicted_n": predicted_n,
            "prompt_ms": 0.0,
            "predicted_ms": wall_ms,
            "wall_ms": wall_ms,
            "steps": steps,
            "max_new_tokens": max_new_tokens,
        }
        return text, stats


def timings_from_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = stats or {}
    prompt_n = int(stats.get("prompt_n") or 0)
    predicted_n = int(stats.get("predicted_n") or 0)
    wall_ms = float(stats.get("wall_ms", stats.get("predicted_ms", 0.0)) or 0.0)
    steps = int(stats.get("steps") or 0)
    out_tps = (predicted_n / wall_ms * 1000.0) if wall_ms > 0 else 0.0
    # Dream unmasks in parallel across the canvas; report wall output rate as
    # primary (matches analyzer gen t/s) and step throughput as parallel.
    par_tps = (steps / wall_ms * 1000.0) if wall_ms > 0 and steps else out_tps
    return {
        "prompt_n": prompt_n,
        "prompt_ms": float(stats.get("prompt_ms") or 0.0),
        "prompt_per_second": 0.0,
        "predicted_n": predicted_n,
        "predicted_ms": wall_ms,
        "predicted_per_second": out_tps,
        "diffusion": True,
        "dream": True,
        "diffusion_steps": steps,
        "diffusion_wall_ms": wall_ms,
        "diffusion_output_tok_s": out_tps,
        "diffusion_parallel_tok_s": par_tps,
        "diffusion_effective_tok_s": out_tps,
    }


def _max_new(body: dict) -> int:
    for key in ("max_tokens", "max_completion_tokens", "max_new_tokens"):
        if key in body and body[key] is not None:
            try:
                return max(1, int(body[key]))
            except (TypeError, ValueError):
                pass
    # Hub ctx is mapped by the spawn path into a default; fall back to README.
    return int(_STATE.get("max_new_tokens") or DEFAULT_MAX_NEW)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _chunk(cid: str, created: int, delta: dict, finish: str | None = None) -> dict:
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish is not None:
        choice["finish_reason"] = finish
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": _STATE.get("model_id", "dream"),
        "choices": [choice],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": _STATE.get("model_id")}


@app.get("/props")
def props():
    eng: DreamEngine | None = _STATE.get("engine")
    n_ctx = int(getattr(eng, "n_ctx", 0) or _STATE.get("n_ctx") or 2048)
    return {
        "model_alias": _STATE.get("model_id", "dream"),
        "model_path": getattr(eng, "model_path", ""),
        "model_ftype": "dream",
        "total_slots": 1,
        "default_generation_settings": {"n_ctx": n_ctx},
    }


@app.get("/slots")
def slots():
    slot = _STATE.get("slot") or {}
    busy = bool(slot.get("busy"))
    eng: DreamEngine | None = _STATE.get("engine")
    n_ctx = int(getattr(eng, "n_ctx", 0) or 2048)
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
    mid = _STATE.get("model_id", "dream")
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
    max_new = _max_new(body)
    steps = body.get("steps")
    try:
        steps_i = int(steps) if steps is not None else None
    except (TypeError, ValueError):
        steps_i = None
    temperature = float(body.get("temperature", DEFAULT_TEMP))
    top_p = float(body.get("top_p", DEFAULT_TOP_P))
    seed = body.get("seed")
    try:
        seed_i = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed_i = None

    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    mid = _STATE.get("model_id", "dream")
    eng: DreamEngine = _STATE["engine"]
    loop = asyncio.get_event_loop()

    if not stream:

        def work():
            with _LOCK:
                _slot_busy(True)
                try:
                    text, stats = eng.generate(
                        messages,
                        max_new_tokens=max_new,
                        steps=steps_i,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed_i,
                    )
                    t = timings_from_stats(stats)
                    _slot_busy(
                        False,
                        decoded=int(t.get("predicted_n") or 0),
                        prompt_n=int(t.get("prompt_n") or 0),
                    )
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

        def on_partial(cumulative: str, decoded_n: int):
            _slot_busy(True, decoded=max(1, decoded_n))
            loop.call_soon_threadsafe(q.put_nowait, ("delta", cumulative))

        def work():
            try:
                with _LOCK:
                    _slot_busy(True)
                    full, stats = eng.generate(
                        messages,
                        max_new_tokens=max_new,
                        steps=steps_i,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed_i,
                        on_partial=on_partial,
                    )
                    timings = timings_from_stats(stats)
                    _slot_busy(
                        False,
                        decoded=int(timings.get("predicted_n") or 0),
                        prompt_n=int(timings.get("prompt_n") or 0),
                    )
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
    ap = argparse.ArgumentParser(description="Dream OpenAI shim for Lemur")
    ap.add_argument("--model", required=True, help="HF model directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument("--alias", default="dream")
    args = ap.parse_args(argv)

    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        print(f"error: Dream model directory not found: {model_path}", flush=True)
        sys.exit(1)

    # Cap default generation length; full n_ctx is for prompt+answer.
    max_new = int(args.max_new_tokens) if args.max_new_tokens > 0 else DEFAULT_MAX_NEW
    max_new = max(1, min(max_new, 4096))

    _STATE["model_id"] = args.alias
    _STATE["max_new_tokens"] = max_new
    _STATE["engine"] = DreamEngine(str(model_path), gpu=str(args.gpu))
    _STATE["n_ctx"] = int(getattr(_STATE["engine"], "n_ctx", 2048))
    _slot_busy(False)
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
