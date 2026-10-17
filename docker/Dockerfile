# CPU-only image: FastAPI service + pipeline. UNTESTED (no Docker on the cluster) - docs/HOWTO.md.
# Build from the repo root:   docker build -f docker/Dockerfile -t datathon-cashier-api .
# Weights are NOT baked in: mount ./models (filled by scripts/download_models.sh) at /app/models.
# Put a .dockerignore at the repo root before building (see docs/HOWTO.md, "Docker").
# docker stop sends SIGTERM: uvicorn shuts down gracefully, jobs in flight stay `processing`
# in the jobs volume and are re-queued when the container starts again.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODELS_DIR=/app/models \
    JOBS_DIR=/app/jobs \
    LOGS_DIR=/app/logs \
    CACHE_DIR=/app/cache \
    API_LOG_STDERR=1

# Runtime libs for opencv-python-headless / torch wheels on a slim image. ffmpeg itself comes
# from the imageio-ffmpeg wheel (static binary), the same fallback the pipeline uses on the cluster.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch wheels first so nothing pulls the CUDA build (same pins as scripts/setup_env.sh).
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" "torchvision==0.20.1" \
 && pip install -r requirements.txt

COPY src/ ./src/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
RUN mkdir -p /app/models /app/jobs /app/logs /app/cache

VOLUME ["/app/models", "/app/jobs"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# Exactly one process: the job queue is in-process (do not add --workers).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
