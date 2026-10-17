#!/usr/bin/env bash
# Stop the tmux service session, the API and the tunnel.
SESSION="${SESSION:-fortebank}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f "[u]vicorn src.api.main:app" 2>/dev/null || true
pkill -f "[b]in/ngrok http" 2>/dev/null || true
echo "stopped"
