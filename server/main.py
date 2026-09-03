import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from server import analyzer, codex, config, gguf_meta, gpu, huggingface, io_trace, models, processes
from server.aliases import model_alias
from server.openai_normalize import normalize_openai_body

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _sync_codex()
    except Exception:
        pass
    yield
    processes.stop_all_servers()


app = FastAPI(
    title="Lemur",
    version=VERSION_FILE.read_text().strip() if VERSION_FILE.is_file() else "dev",
    lifespan=lifespan,
)

NO_CACHE_SUFFIXES = {".html", ".css", ".js"}


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        suffix = Path(path).suffix.lower() if path else ".html"
        if suffix in NO_CACHE_SUFFIXES or path in ("", "index.html"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


class StartServerRequest(BaseModel):
    model: str
    gpu: int
    ctx: int | None = None
    port: int | None = None
    ngl: int | None = None
    host: str | None = None
    spill: str = "none"
    mtp: bool = False
    mtp_draft_n: int | None = None
    vision: bool = False
    sync_codex: bool = True
    set_codex_default: bool = False


class SettingsUpdate(BaseModel):
    llama_server_path: str | None = None
    vllm_bin: str | None = None
    scan_root: str | None = None
    min_model_size_mb: int | None = None
    default_ctx: int | None = None
    default_ngl: int | None = None
    default_host: str | None = None
    port_start: int | None = None
    ui_font_size: int | None = None
    show_splash_on_startup: bool | None = None


class NetworkAccessUpdate(BaseModel):
    lan_enabled: bool


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    server_id: str
    messages: list[ChatMessage]
    stream: bool = True


class CodexSyncRequest(BaseModel):
    default: str | None = None


class FavoriteCreate(BaseModel):
    server_id: str


class HuggingFaceDownloadRequest(BaseModel):
    repo_id: str
    files: list[str]


def _sync_codex(default: str | None = None, context_window: int | None = None) -> dict:
    running = [
        s.to_dict()
        for s in processes.list_servers()
        if s.status in ("running", "starting", "converting")
    ]
    if context_window and default:
        # Prefer explicit ctx from the launch that triggered sync
        for s in running:
            alias = s.get("alias") or s.get("model_name")
            if alias == default:
                s["ctx"] = int(context_window)
                break
        else:
            running = [{"alias": default, "ctx": int(context_window)}] + running
    return codex.sync_from_hub(running, default=default)


def _parse_upstream_error(raw: bytes, status: int) -> dict[str, Any]:
    text = raw.decode(errors="replace")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"error": {"message": text, "type": "upstream_error", "code": status}}


_OUTPUT_TOKEN_FIELDS = ("max_output_tokens", "max_completion_tokens", "max_tokens")


def _context_overflow_retry(
    body: dict[str, Any], raw: bytes, status: int
) -> dict[str, Any] | None:
    """Reduce the output reserve after an upstream context-overflow response."""
    if status != 400:
        return None
    parsed = _parse_upstream_error(raw, status)
    error = parsed.get("error") if isinstance(parsed.get("error"), dict) else parsed
    message = str(error.get("message") or "")
    context_match = re.search(r"maximum context length is (\d+)", message)
    input_match = re.search(r"prompt contains at least (\d+) input tokens", message)
    if not context_match or not input_match:
        return None

    field = next((key for key in _OUTPUT_TOKEN_FIELDS if key in body), None)
    if field is None:
        return None
    try:
        current = int(body[field])
    except (TypeError, ValueError):
        return None
    if current <= 1:
        return None

    context = int(context_match.group(1))
    input_floor = int(input_match.group(1))
    available = max(1, context - input_floor - 1024)
    reduced = max(1, min(current // 2, context // 8, available))
    if reduced >= current:
        return None

    retry = dict(body)
    retry[field] = reduced
    return retry


async def _resolve_running_server(model: str):
    server = processes.find_server_by_model(model)
    if not server:
        running = processes.running_aliases()
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model}' is not running in Lemur. "
                f"Running: {running or 'none'}. Start it in Lemur first."
            ),
        )
    if server.status == "starting":
        for _ in range(60):
            if server.status == "running":
                break
            if server.status in ("failed", "exited", "stopped"):
                raise HTTPException(status_code=503, detail=f"Server {server.status}")
            await asyncio.sleep(0.5)
        else:
            raise HTTPException(status_code=503, detail="Server still starting")
    return server


def _note_timings_from_payload(server_id: str, payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    analyzer.note_api_timings(server_id, payload.get("timings"))


def _ingest_stream_chunk(server_id: str, trace_id: str, path: str, chunk: bytes) -> None:
    for obj in io_trace.parse_sse_payloads(chunk):
        _note_timings_from_payload(server_id, obj)
        if "chat/completions" in path:
            io_trace.ingest_chat_chunk(server_id, trace_id, obj)
        else:
            io_trace.ingest_responses_event(server_id, trace_id, obj)


def _ingest_complete_sse_lines(
    server_id: str,
    trace_id: str,
    path: str,
    pending: bytes,
    chunk: bytes,
) -> bytes:
    """Ingest complete SSE lines and retain an arbitrary partial HTTP chunk."""
    data = pending + chunk
    end = data.rfind(b"\n")
    if end < 0:
        return data
    _ingest_stream_chunk(server_id, trace_id, path, data[: end + 1])
    return data[end + 1 :]


async def _proxy_openai(path: str, body: dict[str, Any]):
    model = body.get("model") or ""
    server = await _resolve_running_server(model)
    connect_host = "127.0.0.1" if server.host == "0.0.0.0" else server.host
    url = f"http://{connect_host}:{server.port}{path}"
    body = normalize_openai_body(path, dict(body))
    body["model"] = server.alias or model_alias(server.model_name)
    stream = bool(body.get("stream"))
    kind = "responses" if "responses" in path else "chat"
    trace_id = io_trace.begin(
        server.id,
        kind=kind,
        input_text=io_trace.format_request_input(path, body),
        model=body.get("model"),
    )

    if stream:
        client = httpx.AsyncClient(timeout=600.0)
        resp = None
        for _ in range(3):
            req = client.build_request("POST", url, json=body)
            resp = await client.send(req, stream=True)
            if resp.status_code == 200:
                break
            err_body = await resp.aread()
            retry = _context_overflow_retry(body, err_body, resp.status_code)
            await resp.aclose()
            if retry is None:
                break
            changed = next(
                key for key in _OUTPUT_TOKEN_FIELDS if retry.get(key) != body.get(key)
            )
            server.logs.append(
                f"context overflow: retrying {changed}={retry[changed]}"
            )
            body = retry
        if resp is None or resp.status_code != 200:
            await client.aclose()
            io_trace.end(server.id, trace_id)
            return JSONResponse(
                content=_parse_upstream_error(err_body, resp.status_code),
                status_code=resp.status_code,
            )

        async def stream_proxy():
            pending = b""
            try:
                async for chunk in resp.aiter_bytes():
                    pending = _ingest_complete_sse_lines(
                        server.id, trace_id, path, pending, chunk
                    )
                    yield chunk
            finally:
                if pending:
                    _ingest_stream_chunk(server.id, trace_id, path, pending)
                io_trace.end(server.id, trace_id)
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(stream_proxy(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=600.0) as client:
        for _ in range(3):
            resp = await client.post(url, json=body)
            if resp.status_code == 200:
                break
            retry = _context_overflow_retry(body, resp.content, resp.status_code)
            if retry is None:
                break
            changed = next(
                key for key in _OUTPUT_TOKEN_FIELDS if retry.get(key) != body.get(key)
            )
            server.logs.append(
                f"context overflow: retrying {changed}={retry[changed]}"
            )
            body = retry
        if resp.status_code != 200:
            io_trace.end(server.id, trace_id)
            return JSONResponse(
                content=_parse_upstream_error(resp.content, resp.status_code),
                status_code=resp.status_code,
            )
        data = resp.json()
        if kind == "chat":
            io_trace.ingest_chat_final(server.id, trace_id, data)
        else:
            # Non-stream responses API: synthesize from final body
            for item in data.get("output") or []:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "reasoning":
                    for c in item.get("content") or []:
                        if isinstance(c, dict) and c.get("text"):
                            io_trace.append(server.id, trace_id, thinking=c["text"])
                if item.get("type") == "message":
                    for c in item.get("content") or []:
                        if isinstance(c, dict) and c.get("text"):
                            io_trace.append(server.id, trace_id, output=c["text"])
            io_trace.end(server.id, trace_id, timings=data.get("usage"))
        _note_timings_from_payload(server.id, data)
        return JSONResponse(data)


@app.get("/api/models")
def api_models():
    items, scanning = models.scan_models()
    out = []
    for m in items:
        d = m.to_dict()
        d["alias"] = m.alias or model_alias(m.name)
        d["codex_cmd"] = codex.codex_command(d["alias"])
        out.append(d)
    return {
        "models": out,
        "scanning": scanning or models.is_scanning(),
        "count": len(out),
    }


@app.post("/api/models/refresh")
def api_models_refresh():
    items, _ = models.scan_models(force=True)
    out = []
    for m in items:
        d = m.to_dict()
        d["alias"] = m.alias or model_alias(m.name)
        d["codex_cmd"] = codex.codex_command(d["alias"])
        out.append(d)
    try:
        sync = _sync_codex()
    except Exception as e:
        sync = {"error": str(e)}
    return {"models": out, "count": len(out), "codex": sync}


@app.get("/api/huggingface/models")
def api_huggingface_models(
    q: str,
    limit: int = 20,
    sort: str = "downloads",
    direction: int = -1,
):
    try:
        limit = max(1, min(99, limit))
        models_found = huggingface.search_models(q, limit + 1, sort, direction)
        return {
            "models": models_found[:limit],
            "has_more": len(models_found) > limit,
            "limit": limit,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/huggingface/models/{author}/{name}")
def api_huggingface_model(author: str, name: str):
    try:
        return huggingface.model_files(f"{author}/{name}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/huggingface/downloads")
def api_huggingface_download(req: HuggingFaceDownloadRequest):
    try:
        return huggingface.start_download(req.repo_id, req.files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/huggingface/downloads/{job_id}")
def api_huggingface_download_status(job_id: str):
    try:
        return huggingface.download_status(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/gpus")
def api_gpus():
    return {"gpus": [g.to_dict() for g in gpu.list_gpus()]}


@app.get("/api/servers")
def api_servers():
    return {"servers": [s.to_dict() for s in processes.list_servers()]}


@app.get("/api/favorites")
def api_favorites():
    items = config.get_favorites()
    return {"favorites": items, "count": len(items)}


@app.post("/api/favorites")
def api_create_favorite(req: FavoriteCreate):
    server = processes.get_server(req.server_id)
    if server is None or server.status not in ("running", "starting", "converting"):
        raise HTTPException(status_code=404, detail="Running server not found")
    preset, created = config.add_favorite(
        {
            "model_path": server.model_path,
            "model_name": server.model_name,
            "alias": server.alias,
            "format": server.format,
            "gpu": server.gpu,
            "ctx": server.ctx,
            "ngl": server.ngl,
            "spill": server.spill,
            "mtp": bool(server.mtp),
            "mtp_draft_n": server.mtp_draft_n,
            "vision": bool(server.vision),
        }
    )
    return {"favorite": preset, "created": created}


@app.delete("/api/favorites/{favorite_id}")
def api_delete_favorite(favorite_id: str):
    if not config.delete_favorite(favorite_id):
        raise HTTPException(status_code=404, detail="Favorite not found")
    return {"ok": True}


@app.post("/api/favorites/{favorite_id}/start")
def api_start_favorite(favorite_id: str):
    favorite = config.get_favorite(favorite_id)
    if favorite is None:
        raise HTTPException(status_code=404, detail="Favorite not found")
    try:
        instance = processes.start_server(
            model_path=str(favorite["model_path"]),
            gpu=int(favorite["gpu"]),
            ctx=int(favorite["ctx"]),
            ngl=int(favorite["ngl"]),
            spill=str(favorite.get("spill") or "none"),
            mtp=bool(favorite.get("mtp")),
            mtp_draft_n=int(favorite.get("mtp_draft_n") or 2),
            vision=bool(favorite.get("vision")),
            format_hint=str(favorite.get("format") or ""),
        )
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except processes.PlacementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = instance.to_dict()
    try:
        result["codex"] = _sync_codex(
            default=instance.alias,
            context_window=instance.ctx,
        )
    except Exception as exc:
        result["codex"] = {"error": str(exc)}
    return result


@app.put("/api/network-access")
def api_update_network_access(req: NetworkAccessUpdate):
    active = [
        s
        for s in processes.list_servers()
        if s.status in ("running", "starting", "converting")
    ]
    busy = [s.model_name for s in active if s.status != "running"]
    if busy:
        raise HTTPException(
            status_code=409,
            detail="Wait for models to finish starting before changing LAN access",
        )

    host = "0.0.0.0" if req.lan_enabled else "127.0.0.1"
    if active and all(s.host == host for s in active):
        config.update_settings({"default_host": host})
        return {"lan_enabled": req.lan_enabled, "restarted": []}

    launches = [
        {
            "model_path": s.model_path,
            "gpu": s.gpu,
            "ctx": s.ctx,
            "port": s.port,
            "ngl": s.ngl,
            "host": host,
            "spill": s.spill,
            "mtp": s.mtp,
            "mtp_draft_n": s.mtp_draft_n,
            "vision": s.vision,
        }
        for s in active
    ]
    processes.stop_all_servers()
    config.update_settings({"default_host": host})

    restarted = []
    try:
        for launch in launches:
            restarted.append(processes.start_server(**launch).to_dict())
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"LAN setting saved, but a model failed to restart: {e}",
        ) from e

    try:
        _sync_codex()
    except Exception:
        pass
    return {"lan_enabled": req.lan_enabled, "restarted": restarted}


@app.post("/api/servers")
def api_start_server(req: StartServerRequest):
    try:
        instance = processes.start_server(
            model_path=req.model,
            gpu=req.gpu,
            ctx=req.ctx,
            port=req.port,
            ngl=req.ngl,
            host=req.host,
            spill=req.spill,
            mtp=req.mtp,
            mtp_draft_n=req.mtp_draft_n,
            vision=req.vision,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except processes.PlacementError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    result = instance.to_dict()
    if req.sync_codex:
        try:
            default = instance.alias if req.set_codex_default else None
            result["codex"] = _sync_codex(
                default=default or instance.alias,
                context_window=instance.ctx,
            )
        except Exception as e:
            result["codex"] = {"error": str(e)}
    return result


@app.delete("/api/servers")
def api_stop_all_servers():
    stopped = processes.stop_all_servers()
    try:
        sync = _sync_codex()
    except Exception as e:
        sync = {"error": str(e)}
    return {"ok": True, "stopped": stopped, "codex": sync}


@app.delete("/api/servers/{server_id}")
def api_stop_server(server_id: str):
    if not processes.stop_server(server_id):
        raise HTTPException(status_code=404, detail="Server not found")
    try:
        sync = _sync_codex()
    except Exception as e:
        sync = {"error": str(e)}
    return {"ok": True, "codex": sync}


@app.get("/api/servers/{server_id}/logs")
def api_server_logs(server_id: str, tail: int = 100):
    if not processes.get_server(server_id):
        raise HTTPException(status_code=404, detail="Server not found")
    return {"logs": processes.get_logs(server_id, tail=tail)}


class VramEstimateRequest(BaseModel):
    model: str
    ctx: int = 8192
    spill: str = "none"
    gpu: int = 0
    mtp: bool = False
    mtp_draft_n: int = 2


@app.post("/api/vram-estimate")
def api_vram_estimate(req: VramEstimateRequest):
    info = models.find_model(req.model)
    if not info:
        # allow raw path
        p = Path(req.model)
        if not p.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        size = p.stat().st_size if p.is_file() else 0
        path = str(p.resolve())
        fmt = "gguf" if p.is_file() else "hf"
    else:
        size = info.size_bytes
        path = info.path
        fmt = info.format

    arch = None
    if fmt == "gguf":
        arch = gguf_meta.read_gguf_arch(path)

    mtp = bool(req.mtp)
    # Auto-hint: GGUF has nextn layers even if UI forgot to check MTP
    mtp_capable = bool(arch and arch.get("mtp_capable")) or gguf_meta.model_looks_mtp(
        path, arch
    )
    draft_n = max(1, min(6, int(req.mtp_draft_n or 2)))
    # The launch path uses Q8 KV for large Qwen 3.5/3.6 models at long
    # context. Estimate the configuration that will actually run.
    qwen35_q8_kv = gguf_meta.uses_qwen35_q8_kv(arch, req.ctx, size)
    est = gguf_meta.estimate_vram_mib(
        weights_bytes=size,
        ctx=req.ctx,
        arch=arch,
        mtp=mtp,
        mtp_draft_n=draft_n,
        kv_bytes_per_elem=1 if qwen35_q8_kv else 2,
    )

    gpus = gpu.list_gpus()
    primary = next((g for g in gpus if g.index == req.gpu), None)
    free_primary = float(primary.memory_free_mib if primary else 0)
    total_primary = float(primary.memory_total_mib if primary else 0)
    spill = (req.spill or "none").lower()

    # Devices llama-server will use (mirrors processes._device_list).
    devices = [req.gpu]
    if spill in ("gpu", "both"):
        others = sorted(g.index for g in gpus if g.index != req.gpu)
        devices = [req.gpu] + others

    by_idx = {g.index: g for g in gpus}
    totals = [max(1.0, float(by_idx[i].memory_total_mib)) for i in devices if i in by_idx]
    frees = [float(by_idx[i].memory_free_mib) for i in devices if i in by_idx]
    if not totals:
        totals = [max(1.0, total_primary or 1.0)]
        frees = [free_primary]
        devices = [req.gpu]

    sum_total = sum(totals) or 1.0
    pool_free = sum(frees)
    vram_safety_mib = 1024.0
    pool_usable = sum(max(0.0, free - vram_safety_mib) for free in frees)
    if spill in ("ram", "both"):
        ram_credit_mib = gpu.available_ram_mib()
    else:
        ram_credit_mib = 0.0

    need = float(est["total_mib"])
    # Per-device share follows llama.cpp -ts by VRAM capacity
    per_gpu = []
    bottleneck = None
    for idx, tot, free in zip(devices, totals, frees):
        share = tot / sum_total
        device_need = need * share
        # MTP + large KV often fails on the smallest card first
        headroom = free - device_need
        entry = {
            "gpu": idx,
            "share_pct": round(100.0 * share, 1),
            "total_gb": round(tot / 1024.0, 2),
            "free_gb": round(free / 1024.0, 2),
            "need_gb": round(device_need / 1024.0, 2),
            "headroom_gb": round(headroom / 1024.0, 2),
            "fits": headroom >= vram_safety_mib,
        }
        per_gpu.append(entry)
        if bottleneck is None or headroom < bottleneck["_mib"]:
            bottleneck = {
                "gpu": idx,
                "headroom_gb": round(headroom / 1024.0, 2),
                "need_gb": entry["need_gb"],
                "free_gb": entry["free_gb"],
                "_mib": headroom,
            }

    fits_vram = all(g["fits"] for g in per_gpu)
    fits = need <= (pool_usable + ram_credit_mib)
    gpu_shortfall_mib = sum(
        max(0.0, vram_safety_mib - float(g["headroom_gb"]) * 1024.0)
        for g in per_gpu
    )
    estimated_ram_mib = (
        max(0.0, need - pool_usable, gpu_shortfall_mib)
        if spill in ("ram", "both") and not fits_vram
        else 0.0
    )
    uses_ram = estimated_ram_mib > 0 and fits

    # Suggested max ctx if over budget (binary-search-ish via proportion)
    tip = None
    if len(devices) > 1 and need + 1024.0 <= free_primary:
        tip = (
            f"fits GPU {req.gpu} alone; adding another GPU is a capacity option "
            "and can reduce generation speed on mismatched cards"
        )
    elif uses_ram:
        if spill == "ram" and req.ctx >= 262144 and len(gpus) >= 2:
            tip = (
                f"about {estimated_ram_mib / 1024.0:.1f} GB may spill to system RAM "
                "for large contexts; another GPU can reduce RAM use"
            )
        else:
            tip = (
                f"about {estimated_ram_mib / 1024.0:.1f} GB may spill to system RAM; "
                "this should load but will be slower"
            )
    elif not fits_vram:
        # Scale ctx down by worst headroom ratio
        worst = min(per_gpu, key=lambda g: g["headroom_gb"])
        if need > 0 and worst["need_gb"] > 0:
            # leave ~1.5GB slack on bottleneck GPU
            target = max(0.5, worst["free_gb"] - 1.5)
            scale = min(1.0, target / worst["need_gb"]) if worst["need_gb"] else 1.0
            # KV+MTP scale with ctx; weights don't — approximate ctx scale more aggressively
            kv_frac = float(est["kv_mib"] + est["mtp_mib"]) / need
            if kv_frac > 0.05 and scale < 1.0:
                # need' ≈ weights + scale_ctx*(kv+mtp+scratch_ctx)
                fixed = float(est["weights_mib"]) / 1024.0
                variable = need / 1024.0 - fixed
                # solve fixed + s*variable ≈ pool share for bottleneck
                share = worst["share_pct"] / 100.0
                budget = (worst["free_gb"] - 1.0) / max(share, 0.05)
                if variable > 0 and budget > fixed:
                    s = (budget - fixed) / variable
                    s = max(0.05, min(1.0, s))
                    suggest = int(max(512, (req.ctx * s) // 512 * 512))
                    if suggest < req.ctx:
                        tip = f"bottleneck GPU {worst['gpu']}: try ctx ≤ {suggest} or disable MTP / add RAM spill"
                else:
                    tip = f"bottleneck GPU {worst['gpu']}: lower ctx, disable MTP, or spill to RAM"
            else:
                tip = f"bottleneck GPU {worst['gpu']}: lower ctx or use + ram spill"
        if mtp and not tip:
            tip = "MTP needs ~1GB+ extra VRAM; lower ctx or turn MTP off"
    elif mtp_capable and not mtp:
        tip = "GGUF looks MTP-capable — enable mtp for ~1.4–2× gen speed (+~1GB VRAM)"
    elif mtp and fits_vram and bottleneck and bottleneck["headroom_gb"] < 2.0:
        tip = f"tight on GPU {bottleneck['gpu']} ({bottleneck['headroom_gb']} GB headroom) — MTP may still OOM"

    return {
        "model": path,
        "ctx": req.ctx,
        "spill": spill,
        "gpu": req.gpu,
        "devices": devices,
        "arch": arch,
        "mtp": mtp,
        "mtp_draft_n": draft_n if mtp else 0,
        "mtp_capable": mtp_capable,
        "confidence": est["confidence"],
        "weights_gb": round(float(est["weights_mib"]) / 1024.0, 2),
        "kv_gb": round(float(est["kv_mib"]) / 1024.0, 2),
        "scratch_gb": round(float(est["scratch_mib"]) / 1024.0, 2),
        "mtp_gb": round(float(est["mtp_mib"]) / 1024.0, 2),
        "need_gb": round(need / 1024.0, 2),
        "free_gb": round(free_primary / 1024.0, 2),
        "pool_gb": round(pool_free / 1024.0, 2),
        "ram_credit_gb": round(ram_credit_mib / 1024.0, 2),
        "estimated_ram_gb": round(estimated_ram_mib / 1024.0, 2),
        "uses_ram": uses_ram,
        "fits": fits,
        "fits_vram": fits_vram,
        "per_gpu": per_gpu,
        "bottleneck": (
            {k: v for k, v in bottleneck.items() if k != "_mib"} if bottleneck else None
        ),
        "tip": tip,
        "verdict": (
            "ok"
            if fits_vram and (bottleneck is None or bottleneck["headroom_gb"] >= 2.0)
            else "tight"
            if fits
            else "oom"
        ),
    }


@app.get("/api/analyzer")
def api_analyzer(server_id: str | None = None):
    return analyzer.analyze(server_id)


@app.get("/api/analyzer/io")
def api_analyzer_io(
    server_id: str | None = None,
    trace_id: str | None = None,
    ts: float | None = None,
):
    servers = [
        s
        for s in processes.list_servers()
        if s.status in ("running", "starting")
    ]
    if not servers:
        raise HTTPException(status_code=404, detail="no running servers")
    server = None
    if server_id:
        server = next((s for s in servers if s.id == server_id), None)
    if not server:
        server = servers[0]
    if trace_id:
        trace = io_trace.get_trace(server.id, trace_id)
    elif ts is not None:
        trace = io_trace.nearest_trace(server.id, ts)
    else:
        trace = io_trace.get_trace(server.id)
    return {
        "server_id": server.id,
        "live": bool(trace and not trace.get("done")),
        "trace": trace,
        "traces": io_trace.list_traces(server.id),
    }


@app.get("/api/analyzer/{server_id}")
def api_analyzer_server(server_id: str):
    data = analyzer.analyze(server_id)
    if data.get("error"):
        raise HTTPException(status_code=404, detail=data["error"])
    active = data.get("active") or {}
    if (active.get("server") or {}).get("id") != server_id and server_id:
        ids = [s.get("id") for s in data.get("servers") or []]
        if server_id not in ids:
            raise HTTPException(status_code=404, detail="Server not found")
    return data


@app.get("/api/settings")
def api_get_settings():
    settings = config.get_settings()
    binary = settings["llama_server_path"]
    return {
        **settings,
        "binary_exists": Path(binary).is_file(),
        "codex_base_url": codex.HUB_BASE_URL,
        "codex_profile": codex.PROFILE_NAME,
    }


@app.put("/api/settings")
def api_update_settings(req: SettingsUpdate):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    settings = config.update_settings(updates)
    binary = settings["llama_server_path"]
    return {
        **settings,
        "binary_exists": Path(binary).is_file(),
        "codex_base_url": codex.HUB_BASE_URL,
        "codex_profile": codex.PROFILE_NAME,
    }


@app.get("/api/codex")
def api_codex_status():
    running = processes.running_aliases()
    cmds = [codex.codex_command(a) for a in running]
    profile = {}
    if codex.PROFILE_PATH.exists():
        for line in codex.PROFILE_PATH.read_text().splitlines():
            if "=" in line and not line.strip().startswith("["):
                k, _, v = line.partition("=")
                profile[k.strip()] = v.strip().strip('"')
    return {
        "base_url": codex.HUB_BASE_URL,
        "profile": codex.PROFILE_NAME,
        "config": str(codex.PROFILE_PATH),
        "running": running,
        "usage": cmds
        or [f"Start a model in Lemur, then: {codex.codex_command('<alias>')}"],
        "command": cmds[0] if cmds else codex.codex_command(),
        "model_context_window": profile.get("model_context_window"),
        "model_auto_compact_token_limit": profile.get(
            "model_auto_compact_token_limit"
        ),
    }


@app.post("/api/codex/sync")
def api_codex_sync(req: CodexSyncRequest | None = None):
    req = req or CodexSyncRequest()
    try:
        return _sync_codex(default=req.default)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    server = processes.get_server(req.server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    if server.status not in ("running", "starting"):
        raise HTTPException(status_code=400, detail=f"Server is {server.status}")

    connect_host = "127.0.0.1" if server.host == "0.0.0.0" else server.host
    url = f"http://{connect_host}:{server.port}/v1/chat/completions"
    payload = {
        "model": server.alias or server.model_name,
        "messages": [m.model_dump() for m in req.messages],
        "stream": req.stream,
    }
    path = "/v1/chat/completions"
    trace_id = io_trace.begin(
        server.id,
        kind="chat",
        input_text=io_trace.format_request_input(path, payload),
        model=payload["model"],
    )

    if req.stream:
        async def stream_proxy():
            pending = b""
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream("POST", url, json=payload) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            yield f"data: {json.dumps({'error': body.decode()})}\n\n"
                            return
                        async for chunk in resp.aiter_bytes():
                            pending = _ingest_complete_sse_lines(
                                server.id, trace_id, path, pending, chunk
                            )
                            yield chunk
            finally:
                if pending:
                    _ingest_stream_chunk(server.id, trace_id, path, pending)
                io_trace.end(server.id, trace_id)

        return StreamingResponse(stream_proxy(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            io_trace.end(server.id, trace_id)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        io_trace.ingest_chat_final(server.id, trace_id, data)
        _note_timings_from_payload(server.id, data)
        return data


# --- OpenAI-compatible API for Codex / other clients ---

@app.get("/v1/models")
def v1_models():
    data = []
    for s in processes.list_servers():
        if s.status not in ("running", "starting"):
            continue
        alias = s.alias or model_alias(s.model_name)
        data.append(
            {
                "id": alias,
                "object": "model",
                "created": int(s.started_at),
                "owned_by": "lemur",
            }
        )
    items, _ = models.scan_models()
    seen = {d["id"] for d in data}
    for m in items:
        alias = m.alias or model_alias(m.name)
        if alias not in seen:
            data.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "lemur-available",
                }
            )
            seen.add(alias)
        # Also list the GGUF basename so either id works
        file_alias = model_alias(m.name)
        if file_alias and file_alias not in seen:
            data.append(
                {
                    "id": file_alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "lemur-available",
                }
            )
            seen.add(file_alias)
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    body: dict[str, Any] = await request.json()
    return await _proxy_openai("/v1/chat/completions", body)


@app.post("/v1/responses")
async def v1_responses(request: Request):
    body: dict[str, Any] = await request.json()
    return await _proxy_openai("/v1/responses", body)


@app.get("/v1/health")
def v1_health():
    return {"status": "ok", "running": processes.running_aliases()}


if ASSET_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(ASSET_DIR)), name="assets")

if STATIC_DIR.is_dir():
    app.mount("/", NoCacheStaticFiles(directory=str(STATIC_DIR), html=True), name="static")
