"""CLI / library entry point for analysing one video.

    python -m src.pipeline.run video.mov --out out.json [--device cpu] [--no-vlm] [--time-budget 170]

Stages: frames -> detections (disk cache, budget-aware frame stride) -> tracks/zones/timeline
-> rules -> descriptions + evidence frames -> schema-valid JSON. Always returns a result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from src.common import CACHE_DIR, setup_logging, timed
from src.pipeline import vlm
from src.pipeline.describe import describe, rule_block
from src.pipeline.detect_cache import cached_detections
from src.pipeline.evidence import evidence_frames
from src.pipeline.frames import FrameSet, extract_frames
from src.pipeline import roi
from src.pipeline.rules import Event, Series, run_rules
from src.pipeline.state import build_timeline
from src.rules_catalog import RULES

log = setup_logging("pipeline")
_detector = None
MAX_VIOLATIONS = int(os.environ.get("MAX_VIOLATIONS", "4"))
SEC_PER_FRAME = float(os.environ.get("DET_SEC_PER_FRAME", "0.75"))


def get_detector():
    global _detector
    if _detector is None:
        from src.pipeline.detect import Detector
        from src.pipeline.detect_cache import DET_KW

        threads = int(os.environ.get("TORCH_THREADS", "0")) or None
        _detector = Detector(threads=threads, **DET_KW)
    return _detector


def _subsample(frames: FrameSet, budget_left: float) -> FrameSet:
    """Pick every k-th frame so that detection fits the remaining budget (k=1 when it already fits)."""
    est = len(frames) * SEC_PER_FRAME
    k = max(1, math.ceil(est / max(1.0, budget_left * 0.85)))
    if k == 1:
        return frames
    picked = frames.frames[::k]
    log.info("time budget: %d frames x %.2fs > %.0fs -> stride %d (%d frames)", len(frames), SEC_PER_FRAME,
             budget_left, k, len(picked))
    return FrameSet(frames.meta, frames.sample_fps / k, frames.long_side, picked)


def merge_events(events: list[Event]) -> list[Event]:
    """Merge same-type events that overlap or nearly touch; keep the strongest few, one per type first."""
    merged: list[Event] = []
    for ev in sorted(events, key=lambda e: (e.type_id, e.t_start)):
        last = merged[-1] if merged else None
        if last and last.type_id == ev.type_id and ev.t_start <= last.t_end + 2.0:
            last.t_end = max(last.t_end, ev.t_end)
            last.score = max(last.score, ev.score)
        else:
            merged.append(Event(ev.type_id, ev.t_start, ev.t_end, ev.score, ev.frame_idx, dict(ev.details)))
    by_type: dict[str, Event] = {}
    for ev in merged:
        if ev.type_id not in by_type or ev.score > by_type[ev.type_id].score:
            by_type[ev.type_id] = ev
    first = sorted(by_type.values(), key=lambda e: -e.score)[:MAX_VIOLATIONS]
    rest = [e for e in merged if e not in first]
    out = first + sorted(rest, key=lambda e: -e.score)[: max(0, MAX_VIOLATIONS - len(first))]
    return sorted(out, key=lambda e: e.t_start)


def _write_partial(path: Path | None, result: dict) -> None:
    if path:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False))
        tmp.replace(path)


def analyze(video_path: str | Path, device: str = "cpu", use_vlm: bool = False,
            time_budget_sec: float = 170.0, partial_path: Path | None = None, cache: bool = True) -> dict:
    """Run the full pipeline and return the API result body (without job_id)."""
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    video_path = Path(video_path)
    log.info("analyze %s device=%s vlm=%s budget=%.0fs", video_path.name, device, use_vlm, time_budget_sec)
    result = {"verdict": "no_violation", "violations": []}

    with timed(log, "frames", timings):
        frames = extract_frames(video_path)
    _write_partial(partial_path, {**result, "note": "partial: frames only"})

    with timed(log, "detector_load", timings):
        det = get_detector()
    left = time_budget_sec - (time.perf_counter() - t_start) - 12.0
    sub = _subsample(frames, left)
    with timed(log, "detect", timings):
        kw = {} if sub is frames else {"cache_dir": CACHE_DIR / "detections_stride"}
        if not cache:
            kw["cache_dir"] = Path("/nonexistent")
        deadline = t_start + time_budget_sec - 10.0
        dets = cached_detections(sub, det, deadline=deadline, **kw)
        if any(d is None for d in dets):                      # budget hit: drop skipped frames
            keep = [i for i, d in enumerate(dets) if d is not None]
            dets = [dets[i] for i in keep]
            sub = FrameSet(sub.meta, sub.sample_fps * len(keep) / max(1, len(sub)), sub.long_side,
                           [sub.frames[i] for i in keep])
            log.warning("budget: kept %d of %d frames", len(keep), len(dets) if False else len(keep))
    with timed(log, "timeline", timings):
        tl = build_timeline(sub, dets)
    with timed(log, "roi", timings):
        try:
            if os.environ.get("USE_ROI", "0") != "1":
                raise RuntimeError("disabled (USE_ROI=1 to enable)")
            cand = roi.candidate_frames(tl, Series(tl))
            left = time_budget_sec - (time.perf_counter() - t_start) - 8.0
            cand = cand[: max(0, int(left / 0.8))]
            tl.cash_roi = roi.desk_valuables(sub, tl.layout, det, cand, deadline=t_start + time_budget_sec - 6.0)
            log.info("roi: %d candidate frames, valuables in %d", len(cand), sum(tl.cash_roi))
        except Exception as e:  # never fail the job because of the second look
            log.info("roi stage skipped: %s", e)
    with timed(log, "blob", timings):
        try:
            if os.environ.get("USE_BLOB", "1") != "1":
                raise RuntimeError("disabled (USE_BLOB=0)")
            from src.pipeline.blob import desk_blobs
            boxes = desk_blobs(sub, tl)
            tl.blob_boxes = boxes
            series = [bool(b) for b in boxes]
            tl.cash_roi = series if tl.cash_roi is None else [a or b for a, b in zip(tl.cash_roi, series)]
            log.info("blob: new desk object in %d of %d frames", sum(series), len(series))
        except Exception as e:  # never fail the job because of the background-difference stage
            log.info("blob stage skipped: %s", e)
    with timed(log, "rules", timings):
        events = merge_events(run_rules(tl))
    log.info("events: %s", [(e.type_id, round(e.t_start, 1), round(e.t_end, 1), round(e.score, 2)) for e in events])

    violations = []
    with timed(log, "evidence", timings):
        for ev in events:
            frames_b64 = evidence_frames(sub, tl, ev)
            desc = describe(ev)
            if use_vlm and vlm.available():
                ans = vlm.confirm([f["image_base64"] for f in frames_b64],
                                  f"Опиши в двух предложениях по-русски только то, что видно на кадрах. "
                                  f"Подтверди или опровергни: {desc}")
                if ans and ans.get("description"):
                    desc = ans["description"]
            violations.append({"type": RULES.get(ev.type_id, RULES["T0"]).name, "timecode_sec": round(ev.t_start, 1),
                               "description": desc, "evidence_frames": frames_b64, "rule": rule_block(ev)})
    result = {"verdict": "violation" if violations else "no_violation", "violations": violations}
    timings["total"] = round(time.perf_counter() - t_start, 3)
    result["_debug"] = {"device": device, "use_vlm": use_vlm, "n_frames": len(frames), "n_detected": len(sub),
                        "duration_sec": frames.meta.duration_sec, "timings": timings,
                        "layout": {k: v for k, v in tl.layout.to_dict().items() if k != "presence"},
                        "events": [{"type": e.type_id, "t_start": round(e.t_start, 1), "t_end": round(e.t_end, 1),
                                    "score": round(e.score, 2), "details": e.details} for e in events]}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse one cashier video.")
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="write result JSON here (default: stdout)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--keep-debug", action="store_true", help="keep the _debug block in the output")
    ap.add_argument("--time-budget", type=float, default=170.0, help="seconds; partial result after it")
    args = ap.parse_args()

    partial = Path(args.out + ".partial.json") if args.out else None
    result = analyze(args.video, device=args.device, use_vlm=not args.no_vlm, time_budget_sec=args.time_budget,
                     partial_path=partial)
    if not args.keep_debug:
        result.pop("_debug", None)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        tmp = Path(args.out).with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(args.out)
        if partial and partial.exists():
            partial.unlink()
        log.info("wrote %s", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
