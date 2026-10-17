#!/usr/bin/env bash
# Expose the local API through ngrok. Token lives in ~/.config/ngrok/ngrok.yml (never in the repo).
# Usage: bash scripts/tunnel.sh [port]   (reads NGROK_DOMAIN and PORT from .env if present)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
PORT="${1:-${PORT:-8000}}"
BIN="$ROOT/bin/ngrok"
if [ -n "${NGROK_DOMAIN:-}" ]; then
  exec "$BIN" http "127.0.0.1:$PORT" --url "https://$NGROK_DOMAIN" --log stdout --log-format json
fi
exec "$BIN" http "127.0.0.1:$PORT" --log stdout --log-format json
