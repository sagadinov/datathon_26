#!/usr/bin/env bash
# Download all model weights into models/hf (the ONLY place where network access for
# weights is allowed). Processing code runs with HF_HUB_OFFLINE=1 and reads from here.
# Usage: bash scripts/download_models.sh [all|detectors|vlm]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HOME="$ROOT/models/hf"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME"
PY="$ROOT/.venv/bin/python"
WHAT="${1:-detectors}"

dl() {  # dl <repo_id> [allow_patterns...]
  "$PY" - "$@" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, patterns = sys.argv[1], sys.argv[2:] or None
path = snapshot_download(repo, allow_patterns=patterns)
print(f"ok {repo} -> {path}")
PYEOF
}

if [[ "$WHAT" == "all" || "$WHAT" == "detectors" ]]; then
  # Open-vocabulary detectors, Apache-2.0. safetensors only (skip .bin duplicates).
  dl IDEA-Research/grounding-dino-tiny "*.json" "*.safetensors" "*.txt"
  dl google/owlv2-base-patch16-ensemble "*.json" "*.safetensors" "*.txt"
fi
if [[ "$WHAT" == "all" || "$WHAT" == "vlm" ]]; then
  dl Qwen/Qwen2.5-VL-7B-Instruct "*.json" "*.safetensors" "*.txt" "*.py"
fi
du -sh "$HF_HOME"
