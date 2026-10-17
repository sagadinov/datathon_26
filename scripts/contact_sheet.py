"""Contact sheet per video: frames every `step` seconds with timestamps -> logs/sheets/X-N.jpg.

    .venv/bin/python scripts/contact_sheet.py 4 20 18      # specific videos
    .venv/bin/python scripts/contact_sheet.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from eval.labels import index_videos, load_labels  # noqa: E402
from src.common import LOGS_DIR  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402


def sheet(n: int, video: Path, label, out_dir: Path, cols: int = 6, tile_w: int = 232, max_tiles: int = 48) -> Path:
    fs = extract_frames(video)
    dur = fs.meta.duration_sec
    step = 1.0
    while dur / step > max_tiles:
        step *= 2
    picks = [fs.nearest(t) for t in [i * step for i in range(int(dur / step) + 1)] if t <= dur]
    tiles = []
    for f in picks:
        img = f.load()
        h, w = img.shape[:2]
        scale = tile_w / w
        img = cv2.resize(img, (tile_w, int(h * scale)))
        mark = label and label.t_sec is not None and abs(f.timestamp_sec - label.t_sec) < step / 2 + 1e-6
        color = (0, 0, 255) if mark else (0, 255, 255)
        cv2.rectangle(img, (0, 0), (tile_w - 1, int(h * scale) - 1), color, 3 if mark else 1)
        cv2.putText(img, f"{f.timestamp_sec:.0f}s", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, f"{f.timestamp_sec:.0f}s", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(tiles[-1] * 0)
    rows = [cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out = cv2.vconcat(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"X-{n}.jpg"
    cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", type=int)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    videos = index_videos()
    labels = {l.n: l for l in load_labels()}
    ns = sorted(videos) if args.all else args.videos
    for n in ns:
        p = sheet(n, videos[n], labels.get(n), LOGS_DIR / "sheets")
        print(p)


if __name__ == "__main__":
    main()
