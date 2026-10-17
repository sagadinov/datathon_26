"""Second-look detection on a zoomed desk crop (spec D4): banknotes on the desk are 10-25 px in the
full frame and mostly invisible to the detector; a 2-3x crop of the desk makes them detectable.

Only candidate frames are re-detected (cashier away, around visitor enter/leave events), results are
cached per frame, and the whole stage respects the remaining time budget.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.common import CACHE_DIR
from src.pipeline.detect import Detection, Detector
from src.pipeline.frames import FrameSet
from src.pipeline.zones import SceneLayout, area, inter_area

log = logging.getLogger("pipeline.roi")
Box = tuple[float, float, float, float]
THR = {"cash": 0.40, "money_bag": 0.45}
_roi_det: Detector | None = None


def roi_detector(base: Detector) -> Detector:
    """A second Grounding DINO instance that upsamples the crop ~3x (shortest edge 576)."""
    global _roi_det
    if _roi_det is None:
        import torch

        _roi_det = Detector(box_threshold=0.25, shortest_edge=576, longest_edge=960, nms_iou=0.6,
                            threads=torch.get_num_threads())
    return _roi_det


def candidate_frames(tl, S, max_frames: int = 48) -> list[int]:
    """Frames worth a second look: cashier away from the seat, or around visitor enter/leave events."""
    n = len(tl.frames)
    want = set(i for i in range(n) if S.away[i])
    for t, _ in tl.entered:
        i = S._t_to_i(t)
        want.update(range(max(0, i - S.sec(5.0)), min(n, i + S.sec(2.0))))
    for t, _ in tl.left:
        i = S._t_to_i(t)
        want.update(range(max(0, i), min(n, i + S.sec(10.0))))
    for i in range(1, n):
        if S.visitors[i] != S.visitors[i - 1] or S.others_at_desk[i] != S.others_at_desk[i - 1]:
            want.update(range(max(0, i - S.sec(4.0)), min(n, i + S.sec(3.0))))
    idx = sorted(want)
    if len(idx) > max_frames:
        step = -(-len(idx) // max_frames)
        idx = idx[::step]
    return idx


def _crop_box(layout: SceneLayout) -> Box | None:
    d = layout.desk
    if d is None:
        return None
    w, h = d[2] - d[0], d[3] - d[1]
    cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
    return (max(0.0, cx - 0.65 * w), max(0.0, cy - 0.8 * h), min(float(layout.frame_w), cx + 0.65 * w),
            min(float(layout.frame_h), cy + 0.7 * h))


def desk_valuables(frames: FrameSet, layout: SceneLayout, detector: Detector, idx: list[int],
                   deadline: float | None = None, cache_dir: Path | None = None) -> list[bool]:
    """Per-frame flag (length = len(frames)): a banknote stack / money bag is visible on the desk crop."""
    n = len(frames)
    out = [False] * n
    crop = _crop_box(layout)
    if crop is None or not idx:
        return out
    masks = [b for b in (layout.counter_machine, layout.laptop) if b]
    cache_dir = cache_dir or CACHE_DIR / "roi"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{frames.frames[0].path.parent.name}_{'_'.join(str(int(v)) for v in crop)}_hr576.json"
    cpath = cache_dir / key
    cache: dict[str, list] = {}
    if cpath.exists():
        try:
            cache = json.loads(cpath.read_text())
        except Exception:
            cache = {}
    todo = [i for i in idx if str(i) not in cache]
    t0 = time.perf_counter()
    x1, y1, x2, y2 = (int(round(v)) for v in crop)
    for s in range(0, len(todo), 4):
        if deadline is not None and time.perf_counter() > deadline:
            log.warning("roi: budget hit after %d of %d frames", s, len(todo))
            break
        chunk = todo[s:s + 4]
        imgs = [frames.frames[i].load()[y1:y2, x1:x2] for i in chunk]
        outs = roi_detector(detector).detect_batch(imgs)
        for i, dets in zip(chunk, outs):
            cache[str(i)] = [[d.label, round(d.score, 3), d.box[0] + x1, d.box[1] + y1, d.box[2] + x1, d.box[3] + y1]
                             for d in dets if d.label in THR]
    if todo:
        tmp = cpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(cpath)
        log.info("roi: %d crops in %.1fs", len(todo), time.perf_counter() - t0)
    desk = layout.desk
    dw = desk[2] - desk[0]
    for i in idx:
        for label, score, bx1, by1, bx2, by2 in cache.get(str(i), []):
            b = (bx1, by1, bx2, by2)
            if score < THR[label] or (b[2] - b[0]) < 0.05 * dw:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if not (desk[0] <= cx <= desk[2] and desk[1] - 0.15 * (desk[3] - desk[1]) <= cy <= desk[3]):
                continue
            if any(inter_area(b, m) / max(1.0, area(b)) > 0.5 for m in masks):
                continue
            out[i] = True
            break
    return out
