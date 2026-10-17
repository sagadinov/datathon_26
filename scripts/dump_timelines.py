"""Write the text timeline of every cached video to logs/timelines/X-N.txt (skips uncached)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.labels import load_labels  # noqa: E402
from src.common import LOGS_DIR, ROOT  # noqa: E402
from src.pipeline.detect_cache import DET_KW, _cfg_without_model, cache_path  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402

out_dir = LOGS_DIR / "timelines"
out_dir.mkdir(parents=True, exist_ok=True)
cfg = _cfg_without_model(DET_KW)
missing = []
for l in load_labels():
    if not cache_path(extract_frames(l.video), cfg).exists():
        missing.append(l.n)
        continue
    txt = subprocess.run([sys.executable, str(ROOT / "scripts/debug_timeline.py"), str(l.n)],
                         capture_output=True, text=True, cwd=ROOT).stdout
    (out_dir / f"X-{l.n}.txt").write_text("\n".join(ln for ln in txt.splitlines() if "INFO" not in ln))
print("written:", len(list(out_dir.glob("X-*.txt"))), "missing cache:", missing)
