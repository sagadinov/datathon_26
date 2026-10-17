"""FastAPI service: X-API-Key auth, POST /api/v1/analyze, GET /api/v1/jobs/{job_id}, static UI.

Run with ONE uvicorn process (scripts/serve.sh): the job queue lives in-process.
Configuration comes from environment variables, with .env at the repo root as defaults.
"""
from __future__ import annotations

import hmac
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

from src.api.jobs import JobStore
from src.api.worker import PipelineWorker
from src.common import LOGS_DIR, ROOT

UPLOAD_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["file"],
            "properties": {"file": {"type": "string", "format": "binary",
                                    "description": "Видео .mov или .mp4"}},
        }}},
    }
}


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=value lines, '#' comments, optional quotes; env wins."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    port: int = 8000
    jobs_dir: Path = ROOT / "jobs"
    workers: int = 2
    torch_threads: int = 32
    time_budget_sec: float = 170.0
    max_upload_mb: int = 500
    pipeline_device: str = "cpu"
    pipeline_vlm: bool = False
    static_dir: Path = ROOT / "src" / "ui" / "static"


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    return Settings(
        api_key=env("API_KEY", "").strip(),
        port=int(env("PORT", "8000")),
        jobs_dir=Path(env("JOBS_DIR", str(ROOT / "jobs"))),
        workers=max(1, int(env("WORKERS", "2"))),
        torch_threads=int(env("TORCH_THREADS", "32")),
        time_budget_sec=float(env("TIME_BUDGET_SEC", "170")),
        max_upload_mb=int(env("MAX_UPLOAD_MB", "500")),
        pipeline_device=env("PIPELINE_DEVICE", "cpu"),
        pipeline_vlm=env("PIPELINE_VLM", "0").lower() in ("1", "true", "yes"),
    )


def setup_api_logging() -> logging.Logger:
    """INFO logger 'api' -> logs/api.log (+ stderr when API_LOG_STDERR=1, e.g. in Docker)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("api")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers: list[logging.Handler] = [logging.FileHandler(LOGS_DIR / "api.log", encoding="utf-8")]
    if os.environ.get("API_LOG_STDERR") == "1":
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _save_upload(src: BinaryIO, dest: Path, limit: int) -> int:
    """Copy the spooled upload to `dest`; returns size, raises 413 when over `limit`."""
    size = 0
    src.seek(0)
    with open(dest, "wb") as fh:
        while chunk := src.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail=f"file too large (max {limit >> 20} MB)")
            fh.write(chunk)
    return size


def create_app(settings: Settings) -> FastAPI:
    log = setup_api_logging()
    store = JobStore(settings.jobs_dir)
    worker = PipelineWorker(
        store, n_workers=settings.workers, time_budget_sec=settings.time_budget_sec,
        torch_threads=settings.torch_threads, repo_root=ROOT, device=settings.pipeline_device,
        use_vlm=settings.pipeline_vlm, log=log,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not settings.api_key:
            raise RuntimeError("API_KEY is not set: put it in .env (see .env.example)")
        requeued = worker.requeue_unfinished()
        worker.start()
        log.info("api started: workers=%d threads=%d budget=%.0fs jobs_dir=%s requeued=%d",
                 settings.workers, settings.torch_threads, settings.time_budget_sec,
                 settings.jobs_dir, requeued)
        yield
        log.info("api stopping: running=%d queue=%d (jobs in flight stay `processing`)",
                 worker.running, worker.queue_size)
        worker.stop()
        log.info("api stopped")

    app = FastAPI(title="Контроль кассовой дисциплины — API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.settings, app.state.store, app.state.worker = settings, store, worker

    async def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        ok = bool(settings.api_key and x_api_key) and hmac.compare_digest(
            x_api_key.encode("utf-8"), settings.api_key.encode("utf-8"))
        if not ok:
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/health")
    async def health() -> dict:
        alive = worker.ensure_threads()  # live worker threads (dead ones are respawned here)
        return {"status": "ok" if alive >= worker.n_workers else "degraded",
                "queue": worker.queue_size, "workers": alive, "running": worker.running}

    @app.post("/api/v1/analyze", dependencies=[Depends(require_key)], openapi_extra=UPLOAD_SCHEMA)
    async def analyze(request: Request) -> dict:
        limit = settings.max_upload_mb << 20
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise HTTPException(status_code=413, detail=f"file too large (max {settings.max_upload_mb} MB)")
        try:
            form = await request.form()
        except Exception:  # malformed multipart body
            raise HTTPException(status_code=400, detail="cannot parse multipart body")
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            await form.close()
            raise HTTPException(status_code=400, detail="multipart field 'file' is required")

        job_id = store.new_job()
        tmp, dest = store.job_dir(job_id) / "input.part", store.input_path(job_id)
        try:
            size = await run_in_threadpool(_save_upload, upload.file, tmp, limit)
            if size == 0:
                raise HTTPException(status_code=400, detail="empty file")
        except Exception:
            shutil.rmtree(store.job_dir(job_id), ignore_errors=True)
            raise
        finally:
            await form.close()
        os.replace(tmp, dest)
        store.write_status(job_id, "processing", created_at=time.time(), filename=upload.filename,
                           content_type=upload.content_type, size_bytes=size)
        worker.enqueue(job_id)
        log.info("job %s queued: %s (%.1f MB) queue=%d", job_id, upload.filename, size / 2**20,
                 worker.queue_size)
        return {"job_id": job_id}

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    async def get_job(job_id: str) -> dict:
        status = store.read_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="job not found")
        return store.api_response(job_id, status)

    # Static UI last so that API routes always win; fallback index when it is not built yet.
    if settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(settings.static_dir), html=True), name="ui")
    else:
        @app.get("/", include_in_schema=False)
        async def index() -> dict:
            return {"service": "Контроль кассовой дисциплины — API", "health": "/health",
                    "docs": "/docs", "analyze": "POST /api/v1/analyze", "jobs": "GET /api/v1/jobs/{job_id}"}
    return app


app = create_app(load_settings())
