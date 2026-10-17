"""CLI / library entry point for analysing one video.

    python -m src.pipeline.run video.mov --out out.json [--device cpu] [--no-vlm]

`analyze()` always returns a schema-valid result dict. Stages are added incrementally;
right now it only extracts frames and returns `no_violation` (evaluation baseline).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.common import setup_logging, timed
from src.pipeline.frames import extract_frames

log = setup_logging("pipeline")


def _limit_threads(device: str) -> None:
    n = os.environ.get("TORCH_THREADS")
    if device == "cpu" and n:
        import torch

        torch.set_num_threads(int(n))


def analyze(
    video_path: str | Path,
    device: str = "cpu",
    use_vlm: bool = False,
    time_budget_sec: float = 180.0,
) -> dict:
    """Run the full pipeline on `video_path` and return the API result body (without job_id)."""
    t_start = time.perf_counter()
    _limit_threads(device)
    timings: dict[str, float] = {}
    video_path = Path(video_path)
    log.info("analyze %s device=%s vlm=%s", video_path.name, device, use_vlm)

    with timed(log, "frames", timings):
        frames = extract_frames(video_path)

    violations: list[dict] = []  # filled by detect -> track -> zones -> rules -> vlm -> evidence

    timings["total"] = round(time.perf_counter() - t_start, 3)
    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
        "_debug": {
            "device": device,
            "use_vlm": use_vlm,
            "n_frames": len(frames),
            "duration_sec": frames.meta.duration_sec,
            "timings": timings,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse one cashier video.")
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="write result JSON here (default: stdout)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--keep-debug", action="store_true", help="keep the _debug block in the output")
    args = ap.parse_args()

    result = analyze(args.video, device=args.device, use_vlm=not args.no_vlm)
    if not args.keep_debug:
        result.pop("_debug", None)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        log.info("wrote %s", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
