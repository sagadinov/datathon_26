#!/usr/bin/env bash
# Start (or restart) the public service on the login node inside a tmux session:
#   window "api"    -> scripts/serve.sh   (uvicorn, restarts automatically if it dies)
#   window "tunnel" -> scripts/tunnel.sh  (ngrok to NGROK_DOMAIN, restarts automatically)
#   window "backup" -> localhost.run ssh tunnel (reserve URL, changes on reconnect; see logs/backup_tunnel.log)
# Usage: bash scripts/start_service.sh        # idempotent: kills the previous session first
#        tmux attach -t fortebank             # watch;  bash scripts/stop_service.sh to stop
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f .env ] || { echo ".env missing (copy .env.example)"; exit 1; }
SESSION="${SESSION:-fortebank}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f "[u]vicorn src.api.main:app" 2>/dev/null || true
pkill -f "[b]in/ngrok http" 2>/dev/null || true
sleep 1
tmux new-session -d -s "$SESSION" -n api -c "$ROOT" \
  "while true; do bash scripts/serve.sh; echo 'api exited, restarting in 2s'; sleep 2; done"
tmux new-window -t "$SESSION" -n tunnel -c "$ROOT" \
  "while true; do bash scripts/tunnel.sh >> logs/tunnel.log 2>&1; echo 'tunnel exited, restarting in 3s'; sleep 3; done"
tmux new-window -t "$SESSION" -n backup -c "$ROOT" \
  "while true; do ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -R 80:localhost:${PORT:-8000} nokey@localhost.run 2>&1 | tee -a logs/backup_tunnel.log; sleep 5; done"
set -a; . ./.env; set +a
for i in $(seq 1 40); do
  curl -sf "http://127.0.0.1:${PORT:-8000}/health" >/dev/null 2>&1 && break
  sleep 1
done
echo "local:  http://127.0.0.1:${PORT:-8000}/health -> $(curl -s "http://127.0.0.1:${PORT:-8000}/health")"
[ -n "${NGROK_DOMAIN:-}" ] && echo "public: https://$NGROK_DOMAIN"
echo "tmux session '$SESSION' (windows: api, tunnel, backup); logs: logs/api.log logs/tunnel.log logs/backup_tunnel.log"
echo "backup URL: $(grep -oE 'https://[a-z0-9]+\.lhr\.life' logs/backup_tunnel.log 2>/dev/null | tail -1)"
