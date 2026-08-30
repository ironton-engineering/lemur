#!/usr/bin/env python3
"""Benchmark DiffusionGemma at a target MAXTOK (default 131072)."""
from __future__ import annotations

import argparse
import json
import math
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_GGUF = (
    Path.home()
    / ".cache/huggingface/hub/models--unsloth--diffusiongemma-26B-A4B-it-GGUF"
    / "snapshots/f4183a2c7a354128d02545752303c4354d165bf0"
    / "diffusiongemma-26B-A4B-it-Q8_0.gguf"
)
DEFAULT_BIN = (
    Path.home()
    / ".unsloth/llama.cpp/build/bin/llama-diffusion-gemma-visual-server"
)


def gpu_used_mib() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return "?"


def rss_gib(pid: int) -> str:
    try:
        for ln in Path(f"/proc/{pid}/status").read_text().splitlines():
            if ln.startswith("VmRSS:"):
                return f"{int(ln.split()[1]) / 1024 / 1024:.1f}"
    except Exception:
        pass
    return "?"


def parse_stats(line: str) -> dict:
    out = {}
    for tok in line.split()[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--maxtok", type=int, default=131072)
    ap.add_argument("--blocks", type=int, default=2, help="diffusion blocks to generate")
    ap.add_argument("--fill-tokens", type=int, default=0, help="0 = maxtok-512")
    ap.add_argument("--fa", type=int, default=1)
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument(
        "--bypass-scores-check",
        action="store_true",
        help="Set DG_FREE_RAM_MB huge so visual-server skips N² scores gate",
    )
    args = ap.parse_args()

    if not args.gguf.is_file():
        print(f"missing gguf: {args.gguf}", file=sys.stderr)
        return 1
    if not args.bin.is_file():
        print(f"missing bin: {args.bin}", file=sys.stderr)
        return 1

    fill = args.fill_tokens or max(512, args.maxtok - 512)
    # ~1 token ≈ 1 word of filler for this rough fill (tokenizer may differ)
    filler = ("alpha bravo charlie delta echo foxtrot golf hotel " * 20000)
    # Build a long user message aiming for ~fill tokens (chars/4 heuristic, then trim)
    target_chars = fill * 4
    body = (filler * (target_chars // len(filler) + 1))[:target_chars]
    prompt = (
        "You are continuing a long document. Read all prior context carefully.\n\n"
        + body
        + "\n\nIn one short paragraph, summarize the document theme and say DONE."
    )

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(args.bin.parent) + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["NGL"] = "99"
    env["FA"] = str(args.fa)
    env["MAXTOK"] = str(args.maxtok)
    # Optional: bypass scores pre-check (can READY a MAXTOK that later OOMs on encode).
    if args.bypass_scores_check:
        env["DG_FREE_RAM_MB"] = "2000000"
    else:
        env.pop("DG_FREE_RAM_MB", None)

    log_path = Path("/tmp/dg_bench.log")
    log_f = log_path.open("w", buffering=1)
    print(
        f"launching {args.bin.name} maxtok={args.maxtok} fa={args.fa} "
        f"fill≈{fill} blocks={args.blocks}",
        flush=True,
    )
    p = subprocess.Popen(
        [str(args.bin), str(args.gguf)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log_f,
        text=True,
        bufsize=1,
        env=env,
    )
    assert p.stdin and p.stdout
    t0 = time.time()
    ready = None
    while time.time() - t0 < args.load_timeout:
        if p.poll() is not None:
            print(f"server exited {p.returncode} after {time.time()-t0:.1f}s", flush=True)
            print(log_path.read_text()[-4000:], flush=True)
            return 2
        r, _, _ = select.select([p.stdout], [], [], 2.0)
        if r:
            line = p.stdout.readline().strip()
            print(f"OUT {line}", flush=True)
            if line.startswith("READY"):
                ready = line
                break
        else:
            el = int(time.time() - t0)
            if el % 10 <= 2:
                print(
                    f"loading t={el}s gpu={gpu_used_mib()}MiB rss={rss_gib(p.pid)}G",
                    flush=True,
                )
                hits = [
                    ln
                    for ln in log_path.read_text().splitlines()
                    if any(
                        k in ln.lower()
                        for k in ("context:", "maxtok", "failed", "oom", "error")
                    )
                ]
                for ln in hits[-3:]:
                    print(f"  log: {ln}", flush=True)
    if not ready:
        print("timeout waiting for READY", flush=True)
        p.kill()
        return 3

    parts = ready.split()
    n_vocab = int(parts[1]) if len(parts) > 1 else 0
    resolved = int(parts[2]) if len(parts) > 2 else args.maxtok
    print(f"READY n_vocab={n_vocab} resolved_maxtok={resolved}", flush=True)
    for ln in log_path.read_text().splitlines():
        if "context: MAXTOK" in ln:
            print(ln, flush=True)

    if resolved < fill + 256:
        print(
            f"resolved MAXTOK {resolved} < fill+canvas {fill+256}; "
            f"clamping fill to {max(0, resolved - 512)}",
            flush=True,
        )
        fill = max(512, resolved - 512)
        target_chars = fill * 4
        body = (filler * (target_chars // len(filler) + 1))[:target_chars]
        prompt = (
            "You are continuing a long document. Read all prior context carefully.\n\n"
            + body
            + "\n\nIn one short paragraph, summarize the document theme and say DONE."
        )

    req = {
        "seed": 3407,
        "n_blocks": args.blocks,
        "messages": [{"role": "user", "content": prompt}],
    }
    req_path = Path(tempfile.gettempdir()) / f"dg_bench_{os.getpid()}.req"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    print(
        f"request fill_chars={len(prompt)} (~{len(prompt)//4} tok heuristic) "
        f"blocks={args.blocks}",
        flush=True,
    )
    t_req = time.time()
    p.stdin.write(str(req_path) + "\n")
    p.stdin.flush()

    stats = {}
    commits = 0
    full = ""
    while True:
        line = p.stdout.readline()
        if not line:
            print("server closed during generate", flush=True)
            break
        line = line.rstrip("\n")
        if line.startswith("C "):
            commits += 1
            parts = line.split(" ", 2)
            if len(parts) >= 3:
                full = json.loads(parts[2])
            print(f"commit#{commits} chars={len(full)}", flush=True)
        elif line.startswith("STATS"):
            stats = parse_stats(line)
            print(f"STATS {stats}", flush=True)
        elif line == "DONE":
            break
        elif line.startswith("ERR"):
            print(f"ERR {line}", flush=True)
            break
        elif line.startswith("F "):
            continue
        else:
            print(f"OUT {line}", flush=True)

    wall = time.time() - t_req
    print("--- result ---", flush=True)
    print(f"wall_s={wall:.2f}", flush=True)
    print(f"resolved_maxtok={resolved}", flush=True)
    print(f"reply_chars={len(full)} reply_preview={full[:240]!r}", flush=True)
    if stats:
        pn = int(stats.get("prompt_n") or 0)
        gn = int(stats.get("predicted_n") or 0)
        wall_ms = float(stats.get("wall_ms") or stats.get("predicted_ms") or 0)
        steps = int(stats.get("steps") or 0)
        blocks = int(stats.get("blocks") or 0)
        canvas = int(stats.get("canvas") or 256)
        out_tps = (gn / wall_ms * 1000.0) if wall_ms > 0 else 0.0
        par_tps = (canvas * steps / wall_ms * 1000.0) if wall_ms > 0 else 0.0
        print(f"prompt_n={pn} predicted_n={gn}", flush=True)
        print(f"output_tok_s={out_tps:.1f}", flush=True)
        print(f"parallel_tok_s={par_tps:.1f}", flush=True)
        print(f"steps={steps} blocks={blocks} canvas={canvas}", flush=True)
        print(f"wall_ms={wall_ms} decode_ms={stats.get('decode_ms')}", flush=True)

    try:
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=15)
    except Exception:
        p.kill()
    log_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
