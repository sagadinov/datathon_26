#!/usr/bin/env bash
# Full evaluation: original dataset + robustness variants (hflip, small, rot_p10, rot_m10) with cached detections.
# Usage: bash scripts/eval_all.sh [tag] [jobs]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
TAG="${1:-}"; JOBS="${2:-8}"
run() { .venv/bin/python -m eval.evaluate --no-vlm --jobs "$JOBS" --tag "$TAG $1" "${@:2}" 2>&1 | grep -vE "INFO|Warning|warn" | sed -n '/SUMMARY/,$p' | grep -E "recall|fp_on_clean|hit_3s|hit_5s|type_acc|accuracy|fn_videos|fp_videos|wrote"; }
echo "=== original ==="; run original
for v in hflip small rot_p10 rot_m10; do
  [ -d "data/aug/$v/videos" ] || continue
  echo "=== $v ==="; run "$v" --videos-dir "data/aug/$v/videos"
done
