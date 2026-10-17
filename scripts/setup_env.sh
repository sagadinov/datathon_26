#!/usr/bin/env bash
# Create the project virtualenv with uv (Python 3.11) and install CPU deps.
# Usage: bash scripts/setup_env.sh   (idempotent; re-run to sync requirements)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -d .venv ] || uv venv --python 3.11 .venv
# CPU-only torch wheels first so that nothing pulls the CUDA build.
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.5.1" "torchvision==0.20.1"
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import torch, transformers, cv2; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cv2', cv2.__version__)"
