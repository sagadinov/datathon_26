"""Helpers for safe.py: box clustering with per-frame presence, seated-height estimate, safe ROI pixel diff.

Everything is relative to box sizes; no pixel constants that assume a frame size.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from src.pipeline.track import iou
from src.pipeline.zones import area, inter_area

Box = tuple[float, float, float, float]


@dataclass
class Cluster:
    """A spatial cluster of boxes of one static object over the video."""
    box: Box                                   # median box
    frames: dict[int, float] = field(default_factory=dict)   # frame index -> best score in that frame
    boxes: list[Box] = field(default_factory=list)
    n_frames: int = 1

    @property
    def presence(self) -> float:
        return len(self.frames) / max(1, self.n_frames)

    @property
    def score(self) -> float:
        return float(np.mean(list(self.frames.values()))) if self.frames else 0.0

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def median_box(boxes: list[Box]) -> Box:
    arr = np.asarray(boxes, dtype=np.float32)
    return tuple(float(v) for v in np.median(arr, axis=0))


def cluster_frames(items: list[tuple[int, Box, float]], n_frames: int, iou_thresh: float = 0.6) -> list[Cluster]:
    """Greedy IoU clustering of (frame, box, score) items against the running median box of each cluster.
    Presence counts frames (several boxes of one frame in the same cluster count once)."""
    clusters: list[Cluster] = []
    for i, b, s in items:
        for c in clusters:
            if iou(b, c.box) >= iou_thresh:
                c.boxes.append(b)
                c.frames[i] = max(c.frames.get(i, 0.0), s)
                if len(c.boxes) % 8 == 0:
                    c.box = median_box(c.boxes)
                break
        else:
            clusters.append(Cluster(b, {i: s}, [b], n_frames))
    for c in clusters:
        c.box = median_box(c.boxes)
        c.n_frames = n_frames
    return sorted(clusters, key=lambda c: -c.presence)


def frac_inside(b: Box, zone: Box) -> float:
    a = area(b)
    return inter_area(b, zone) / a if a > 0 else 0.0


def box_gap(a: Box, b: Box) -> tuple[float, float]:
    """(x gap, y gap) between two boxes in pixels; 0 when they overlap along that axis."""
    gx = max(0.0, a[0] - b[2], b[0] - a[2])
    gy = max(0.0, a[1] - b[3], b[1] - a[3])
    return gx, gy


def x_overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def seated_ref_height(heights: list[float]) -> float | None:
    """Reference height of the seated cashier: the lower mode of a 2-means split of the box heights when the
    distribution is bimodal (standing/seated ratio >= 1.3), else the median. None when nothing is known."""
    hs = sorted(float(h) for h in heights if h > 0)
    if len(hs) < 3:
        return float(np.median(hs)) if hs else None
    lo, hi = hs[0], hs[-1]
    if hi < 1.3 * lo:
        return float(np.median(hs))
    c_lo, c_hi = float(np.percentile(hs, 20)), float(np.percentile(hs, 80))
    for _ in range(10):
        low = [h for h in hs if abs(h - c_lo) <= abs(h - c_hi)]
        high = [h for h in hs if abs(h - c_lo) > abs(h - c_hi)]
        if not low or not high:
            break
        c_lo, c_hi = float(np.mean(low)), float(np.mean(high))
    low = [h for h in hs if abs(h - c_lo) <= abs(h - c_hi)]
    if len(low) >= 3 and c_hi >= 1.3 * c_lo:
        return float(np.median(low))
    return float(np.median(hs))


class RoiDiff:
    """Mean absolute grey difference of the safe ROI against a baseline, with person boxes masked out.

    The baseline is the per-pixel median over the `n_base` least-suspicious early frames (lowest gated
    safe_door_open score, i.e. most likely closed). Frames whose global brightness differs from the baseline
    frames by more than `light_tol` (lights off / exposure jump) and frames with less than half of the ROI
    unmasked give None (no vote)."""

    def __init__(self, frames, roi: Box, persons_per_frame: list[list[Box]], order: list[int],
                 n_base: int = 5, light_tol: float = 0.25, changed_thr: float = 40.0):
        import cv2

        self._cv2 = cv2
        x1, y1, x2, y2 = roi
        self.roi = (max(0, int(x1)), max(0, int(y1)), int(np.ceil(x2)), int(np.ceil(y2)))
        self.frames, self.persons = frames, persons_per_frame
        self.light_tol, self.changed_thr = light_tol, changed_thr
        self.base: np.ndarray | None = None
        self.base_bright = 0.0
        self._build(order, n_base)

    def _load(self, i: int):
        img = self._cv2.imread(str(self.frames.frames[i].path), self._cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None, 0.0
        h, w = img.shape
        x1, y1, x2, y2 = self.roi
        x2, y2 = min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return None, None, 0.0
        crop = img[y1:y2, x1:x2].astype(np.float32)
        mask = np.ones(crop.shape, dtype=bool)
        for p in self.persons[i]:
            px1, py1 = max(x1, int(p[0])), max(y1, int(p[1]))
            px2, py2 = min(x2, int(np.ceil(p[2]))), min(y2, int(np.ceil(p[3])))
            if px2 > px1 and py2 > py1:
                mask[py1 - y1:py2 - y1, px1 - x1:px2 - x1] = False
        return crop, mask, float(img.mean())

    def _build(self, order: list[int], n_base: int) -> None:
        crops, brights = [], []
        for i in order:
            crop, mask, bright = self._load(i)
            if crop is None or mask.mean() < 0.6:
                continue
            crop = crop.copy()
            crop[~mask] = np.nan
            crops.append(crop)
            brights.append(bright)
            if len(crops) >= n_base:
                break
        if len(crops) >= 2:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.base = np.nanmedian(np.stack(crops), axis=0)
            self.base_bright = float(np.mean(brights))

    def diff(self, i: int) -> tuple[float, float] | None:
        """(mean absolute difference, fraction of pixels changed by >= changed_thr) over the unmasked ROI, or None."""
        if self.base is None:
            return None
        crop, mask, bright = self._load(i)
        if crop is None:
            return None
        valid = mask & ~np.isnan(self.base)
        if valid.mean() < 0.5 or abs(bright - self.base_bright) > self.light_tol * max(1.0, self.base_bright):
            return None
        d = np.abs(crop[valid] - self.base[valid])
        return float(d.mean()), float((d >= self.changed_thr).mean())


def fill_gaps(values: list[bool], max_gap: int) -> list[bool]:
    """Bridge False gaps of length <= max_gap between True runs (tolerate missed detections)."""
    out = list(values)
    runs, i = [], 0
    while i < len(values):
        if values[i]:
            j = i
            while j < len(values) and values[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    for (_, b1), (a2, _) in zip(runs, runs[1:]):
        if a2 - b1 <= max_gap:
            for k in range(b1, a2):
                out[k] = True
    return out
