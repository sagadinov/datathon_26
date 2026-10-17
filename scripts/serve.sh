#!/usr/bin/env bash
# Run the public API as ONE uvicorn process (the job queue is in-process: never --workers N).
# Usage: bash scripts/serve.sh              # foreground, all output -> $LOGS_DIR/api.log
#        nohup bash scripts/serve.sh &      # background
# Config: .env at the repo root (API_KEY, PORT, WORKERS, TORCH_THREADS, TIME_BUDGET_SEC, ...).
# HOST defaults to 127.0.0.1: the tunnel (scripts/tunnel.sh) connects locally.
# Stop with SIGTERM (kill <pid>, Ctrl-C, pkill -f "uvicorn src.api.main:app"): the running
# pipelines are killed and their jobs stay `processing`, the next start re-queues them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f .env ]; then
  # .env provides defaults; variables already set in the shell (PORT=18800 bash scripts/serve.sh) win.
  _saved="$(export -p)"
  set -a; . ./.env; set +a
  eval "$_saved"
fi
: "${API_KEY:?API_KEY is not set: copy .env.example to .env and fill it in}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
# Same defaults as src/common.py (LOGS_DIR) and src/api/main.py (JOBS_DIR); exported so the app,
# the uvicorn access log and this script agree on the directories.
export LOGS_DIR="${LOGS_DIR:-$ROOT/logs}" JOBS_DIR="${JOBS_DIR:-$ROOT/jobs}"
mkdir -p "$LOGS_DIR" "$JOBS_DIR"
echo "API: http://$HOST:$PORT  (log: $LOGS_DIR/api.log, jobs: $JOBS_DIR)"
exec .venv/bin/uvicorn src.api.main:app --host "$HOST" --port "$PORT" --log-level info >> "$LOGS_DIR/api.log" 2>&1
