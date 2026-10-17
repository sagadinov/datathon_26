"""Pixel measurements and series helpers for screen.py (P6/P7), kept apart to stay within the size limit.

Everything here is relative to the laptop box found by zones.build_layout and to the video's own
brightness levels; no pixel constants that assume a frame size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.pipeline.detect import Detection
from src.pipeline.track import iou
from src.pipeline.zones import SceneLayout, area, inter_area

Box = tuple[float, float, float, float]

WHITE_LEVEL = 170           # gray level above which a pixel counts as "white" (spec P6)
REGION_FRAC = 0.58          # upper part of the laptop box that holds the screen (spec: 55-60%)
MATCH_IOU = 0.3             # a per-frame laptop box belongs to the layout laptop when IoU >= this
MIN_DET_SCORE = 0.4         # same score as zones._pick uses for the laptop cluster


@dataclass
class Measure:
    ok: bool                # frame image could be read
    brightness: float       # frame mean gray
    mean: float             # mean gray of the screen region (NaN without a laptop)
    white: float            # fraction of region pixels > WHITE_LEVEL
    white_full: float       # same fraction over the whole laptop box
    blue: float             # mean(B) - mean(R) of the region (bluish screens of camera A / X-28)
    occ: float              # fraction of the region area covered by the largest overlapping person box


def screen_region(laptop: Box) -> Box:
    """Upper REGION_FRAC of the laptop box: the lid/screen, without the keyboard."""
    x1, y1, x2, y2 = laptop
    return (x1, y1, x2, y1 + REGION_FRAC * (y2 - y1))


def crop(img: np.ndarray, box: Box) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x2, y2 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img[y1:y2, x1:x2]


def measure_frame(path: Path, laptop: Box | None, persons: list[Detection]) -> Measure:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    nan = float("nan")
    if img is None:
        return Measure(False, 0.0, nan, 0.0, 0.0, 0.0, 0.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if laptop is None:
        return Measure(True, brightness, nan, 0.0, 0.0, 0.0, 0.0)
    region = screen_region(laptop)
    reg, full = crop(img, region), crop(gray, laptop)
    if reg is None or full is None:
        return Measure(True, brightness, nan, 0.0, 0.0, 0.0, 0.0)
    g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    occ = max([inter_area(d.box, region) / area(region) for d in persons] + [0.0])
    return Measure(True, brightness, float(g.mean()), float((g > WHITE_LEVEL).mean()),
                   float((full > WHITE_LEVEL).mean()), float(reg[..., 0].mean() - reg[..., 2].mean()), occ)


# ---- laptop anchor sanity ---------------------------------------------------------------------------
def match_laptop(dets: list[Detection], laptop: Box, min_score: float) -> Detection | None:
    """Best-scoring laptop detection of one frame that overlaps the layout laptop (IoU >= MATCH_IOU)."""
    cands = [d for d in dets if d.label == "laptop" and d.score >= min_score and iou(d.box, laptop) >= MATCH_IOU]
    return max(cands, key=lambda d: d.score) if cands else None


def laptop_presence(layout: SceneLayout, dets_per_frame: list[list[Detection]] | None) -> float:
    """Fraction of frames showing the laptop at the anchor. The cluster presence of the layout undercounts
    when the lid closes mid-video (open and closed boxes form two clusters: X-12 0.67 + 0.26), so the
    per-frame matches against the anchor are counted as well and the larger value is used."""
    base = float(layout.presence.get("laptop", 0.0))
    if not dets_per_frame or layout.laptop is None:
        return base
    hit = sum(1 for dets in dets_per_frame if match_laptop(dets, layout.laptop, MIN_DET_SCORE) is not None)
    return max(base, hit / max(1, len(dets_per_frame)))


def laptop_on_desk(laptop: Box, desk: Box | None) -> bool:
    """The laptop box sits on / next to the desk: centre x within the desk span (+10% of its width) and the
    bottom edge between half a desk height above the desk top and a quarter below its bottom."""
    if desk is None:
        return True
    dw, dh = max(1.0, desk[2] - desk[0]), max(1.0, desk[3] - desk[1])
    cx = (laptop[0] + laptop[2]) / 2
    return desk[0] - 0.1 * dw <= cx <= desk[2] + 0.1 * dw and desk[1] - 0.5 * dh <= laptop[3] <= desk[3] + 0.25 * dh


def laptop_hw_series(layout: SceneLayout, dets_per_frame: list[list[Detection]] | None, n: int) -> list[float]:
    """Per-frame height/width of the laptop box (matched detection, else the layout box), gaps filled from the
    neighbouring frames. Without detections the series is the constant layout ratio."""
    lp = layout.laptop
    base = (lp[3] - lp[1]) / max(1e-6, lp[2] - lp[0])
    if not dets_per_frame:
        return [base] * n
    vals: list[float | None] = []
    for dets in dets_per_frame[:n]:
        d = match_laptop(dets, lp, 0.3)
        vals.append((d.box[3] - d.box[1]) / max(1e-6, d.box[2] - d.box[0]) if d else None)
    vals += [None] * (n - len(vals))
    out: list[float] = []
    last: float | None = None
    for v in vals:
        last = v if v is not None else last
        out.append(last if last is not None else math.nan)
    first = next((v for v in out if not math.isnan(v)), base)
    return [first if math.isnan(v) else v for v in out]


# ---- 1-D series helpers -----------------------------------------------------------------------------------
def median_of(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def median3(values: list[bool]) -> list[bool]:
    """Median filter of width 3 (single-frame glitches removed), edges kept."""
    n = len(values)
    return [values[i] if i == 0 or i == n - 1 else (values[i - 1] + values[i] + values[i + 1]) >= 2 for i in range(n)]


def majority(seq: list[str], before: int, after: int) -> list[str]:
    """Majority vote in the window [i - before, i + after] (3 of 4 frames at the spec's 2 fps); when no
    class reaches the majority the previous output is kept (flicker, X-19 34.5-44 s)."""
    out: list[str] = []
    prev: str | None = None
    for i in range(len(seq)):
        win = seq[max(0, i - before): i + after + 1]
        need = len(win) // 2 + 1
        counts: dict[str, int] = {}
        for c in win:
            counts[c] = counts.get(c, 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] >= need or prev is None:
            prev = best if counts[best] >= need else seq[i]
        out.append(prev)
    return out
