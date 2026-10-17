#!/usr/bin/env bash
# Expose the local API through ngrok (token lives in ~/.config/ngrok/ngrok.yml, never in the repo).
# Usage: bash scripts/tunnel.sh [port] ; set NGROK_DOMAIN for a reserved static domain.
set -euo pipefail
PORT="${1:-8000}"
BIN="$(dirname "$0")/../bin/ngrok"
if [ -n "${NGROK_DOMAIN:-}" ]; then exec "$BIN" http "127.0.0.1:$PORT" --url "https://$NGROK_DOMAIN"; fi
exec "$BIN" http "127.0.0.1:$PORT"
