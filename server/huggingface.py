"""Small Hugging Face model browser and GGUF downloader."""

from __future__ import annotations

import os
import re
import shutil
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx

HUB_URL = "https://huggingface.co"
DOWNLOAD_ROOT = Path.home() / "models" / "HuggingFace"
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _repo_id(value: str) -> str:
    value = value.strip()
    if not _REPO_RE.fullmatch(value):
        raise ValueError("Invalid Hugging Face repository ID")
    return value


def _filename(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".gguf":
        raise ValueError("Only GGUF files can be downloaded")
    return str(path)


def _request_json(path: str, *, params: dict[str, Any] | None = None) -> Any:
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(f"{HUB_URL}{path}", params=params, headers=_headers())
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            raise ValueError("This model needs a valid HF_TOKEN") from exc
        if exc.response.status_code == 404:
            raise ValueError("Hugging Face model not found") from exc
        raise ValueError(f"Hugging Face returned HTTP {exc.response.status_code}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        if isinstance(exc, ValueError) and not isinstance(exc, httpx.HTTPError):
            raise
        raise ValueError(f"Cannot reach Hugging Face: {exc}") from exc


def search_models(
    query: str,
    limit: int = 30,
    sort: str = "downloads",
    direction: int = -1,
) -> list[dict[str, Any]]:
    query = query.strip()
    if len(query) < 2:
        raise ValueError("Enter at least two search characters")
    allowed_sorts = {"downloads", "likes", "lastModified"}
    if sort not in allowed_sorts:
        sort = "downloads"
    direction = 1 if int(direction) == 1 else -1
    data = _request_json(
        "/api/models",
        params={
            "search": query,
            "filter": "gguf",
            "sort": sort,
            "direction": direction,
            "limit": max(1, min(100, int(limit))),
            "full": "true",
        },
    )
    if not isinstance(data, list):
        return []
    results = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        author = str(item.get("author") or item["id"].split("/", 1)[0])
        tags = item.get("tags") or []
        license_tag = next(
            (tag.split(":", 1)[1] for tag in tags if str(tag).startswith("license:")),
            None,
        )
        gguf_count = sum(
            1
            for file in item.get("siblings") or []
            if str(file.get("rfilename", "")).lower().endswith(".gguf")
        )
        results.append(
            {
                "id": item["id"],
                "author": author,
                "avatar_url": f"{HUB_URL}/avatars/{quote(author, safe='')}.svg",
                "author_url": f"{HUB_URL}/{quote(author, safe='')}",
                "downloads": int(item.get("downloads") or 0),
                "likes": int(item.get("likes") or 0),
                "license": license_tag,
                "gated": bool(item.get("gated")),
                "updated": item.get("lastModified"),
                "gguf_count": gguf_count,
                "url": f"{HUB_URL}/{item['id']}",
            }
        )
    return results


def model_files(repo_id: str) -> dict[str, Any]:
    repo_id = _repo_id(repo_id)
    data = _request_json(f"/api/models/{quote(repo_id, safe='/')}", params={"blobs": "true"})
    files = []
    for item in data.get("siblings") or []:
        name = str(item.get("rfilename") or "")
        if not name.lower().endswith(".gguf"):
            continue
        size = item.get("size") or (item.get("lfs") or {}).get("size") or 0
        files.append({"name": name, "size": int(size)})
    files.sort(key=lambda item: item["name"].lower())
    return {
        "id": repo_id,
        "gated": bool(data.get("gated")),
        "files": files,
        "url": f"{HUB_URL}/{repo_id}",
    }


def _set_job(job_id: str, **updates: Any) -> None:
    with _lock:
        _jobs[job_id].update(updates)


def _download(job_id: str, repo_id: str, files: list[str]) -> None:
    target_root = DOWNLOAD_ROOT.joinpath(*repo_id.split("/"))
    downloaded = 0
    try:
        target_root.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(files, start=1):
            target = target_root.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                downloaded += target.stat().st_size
                _set_job(
                    job_id,
                    file=name,
                    file_index=index,
                    downloaded_bytes=downloaded,
                )
                continue
            partial = target.with_name(f".{target.name}.part")
            _set_job(job_id, file=name, file_index=index, downloaded_bytes=downloaded)
            url = f"{HUB_URL}/{quote(repo_id, safe='/')}/resolve/main/{quote(name, safe='/')}"
            with httpx.stream(
                "GET", url, headers=_headers(), follow_redirects=True, timeout=None
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        _set_job(job_id, downloaded_bytes=downloaded)
            os.replace(partial, target)
        _set_job(job_id, status="complete", path=str(target_root), file=None)
    except Exception as exc:
        try:
            partial.unlink(missing_ok=True)
        except UnboundLocalError:
            pass
        _set_job(job_id, status="failed", error=str(exc), file=None)


def start_download(repo_id: str, files: list[str]) -> dict[str, Any]:
    repo_id = _repo_id(repo_id)
    clean_files = list(dict.fromkeys(_filename(item) for item in files))
    if not clean_files:
        raise ValueError("Select at least one GGUF file")
    if len(clean_files) > 64:
        raise ValueError("Too many files selected")
    info = model_files(repo_id)
    remote = {item["name"]: item["size"] for item in info["files"]}
    if any(name not in remote for name in clean_files):
        raise ValueError("A selected GGUF file is not in this repository")
    total = sum(remote[name] for name in clean_files)
    free = shutil.disk_usage(Path.home()).free
    if total and total > free:
        raise ValueError(
            f"The download needs {total / 1024**3:.1f} GB, but only "
            f"{free / 1024**3:.1f} GB is free"
        )
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "repo_id": repo_id,
        "files": clean_files,
        "file": None,
        "file_index": 0,
        "file_count": len(clean_files),
        "downloaded_bytes": 0,
        "total_bytes": total,
        "status": "queued",
        "error": None,
        "path": None,
    }
    with _lock:
        _jobs[job_id] = job
    _set_job(job_id, status="downloading")
    thread = threading.Thread(
        target=_download, args=(job_id, repo_id, clean_files), daemon=True
    )
    thread.start()
    return download_status(job_id)


def download_status(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise ValueError("Download not found")
        return dict(job)
