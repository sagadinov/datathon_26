"""Run a detector on a few labelled windows and dump annotated frames to logs/debug/<backend>/.

    .venv/bin/python scripts/debug_detect.py --backend gdino --threads 16
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from eval.labels import index_videos  # noqa: E402
from src.common import LOGS_DIR, setup_logging  # noqa: E402
from src.pipeline.detect import Detector, draw  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402

# video number -> (t_from, t_to, what we expect to see)
WINDOWS = {4: (51, 57, "cash on desk, cashier away"), 20: (4, 9, "open safe door"),
           18: (19, 22, "second person in cashier zone"), 46: (11, 15, "cup on desk")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gdino", choices=["gdino", "owlv2"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--per-video", type=int, default=5)
    ap.add_argument("--box-threshold", type=float, default=0.25)
    ap.add_argument("--shortest-edge", type=int, default=None)
    ap.add_argument("--windows", default=None, help='override, e.g. "4:57-63,26:9-13"')
    args = ap.parse_args()
    setup_logging("debug_detect")

    windows = WINDOWS
    if args.windows:
        windows = {int(w.split(":")[0]): (float(w.split(":")[1].split("-")[0]), float(w.split(":")[1].split("-")[1]), "custom")
                   for w in args.windows.split(",")}
    det = Detector(args.backend, threads=args.threads, box_threshold=args.box_threshold,
                   shortest_edge=args.shortest_edge)
    out_dir = LOGS_DIR / "debug" / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = index_videos()
    tiles, total_s, total_n = [], 0.0, 0
    for n, (t0, t1, note) in windows.items():
        fs = extract_frames(videos[n])
        step = max(1, int((t1 - t0) * fs.sample_fps // args.per_video))
        picks = [f for f in fs if t0 <= f.timestamp_sec <= t1][::step][: args.per_video]
        print(f"\n=== X-{n} {t0}-{t1}s: expect {note} ===")
        for f in picks:
            img = f.load()
            tic = time.perf_counter()
            dets = det.detect(img)
            dt = time.perf_counter() - tic
            total_s, total_n = total_s + dt, total_n + 1
            found = Counter(d.label for d in dets)
            print(f"  t={f.timestamp_sec:5.1f}s {dt:5.2f}s  " + ", ".join(f"{k}x{v}" for k, v in found.most_common()))
            for d in sorted(dets, key=lambda d: -d.score)[:12]:
                print(f"       {d.label:15s} {d.score:.2f} '{d.phrase}' box={tuple(int(v) for v in d.box)}")
            vis = draw(img, dets)
            cv2.putText(vis, f"X-{n} t={f.timestamp_sec}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(str(out_dir / f"X-{n}_t{f.timestamp_sec:05.1f}.jpg"), vis)
            tiles.append(vis)
    # contact sheet: one row per video
    rows, k = [], args.per_video
    for i in range(0, len(tiles), k):
        row = tiles[i:i + k]
        h = min(t.shape[0] for t in row)
        rows.append(cv2.hconcat([t[:h] for t in row]))
    w = min(r.shape[1] for r in rows)
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), cv2.vconcat([r[:, :w] for r in rows]))
    print(f"\n{args.backend}: {total_n} frames, {total_s / total_n:.2f}s/frame with {args.threads} threads -> {out_dir}")


if __name__ == "__main__":
    main()
