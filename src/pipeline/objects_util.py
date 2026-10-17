"""Helpers for objects.py: box geometry, per-class IoU cluster tracks and small boolean-series utilities.

Kept separate so that objects.py stays within the 400-line budget. Nothing here depends on state.py
(state imports objects, so objects must not import state).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.detect import Detection
from src.pipeline.track import iou
from src.pipeline.zones import area, inter_area

Box = tuple[float, float, float, float]


# ---- geometry ------------------------------------------------------------------------------
def contain_frac(b: Box, zone: Box) -> float:
    """Fraction of the area of `b` that lies inside `zone`."""
    a = area(b)
    return inter_area(b, zone) / a if a > 0 else 0.0


def overlap_min(a: Box, b: Box) -> float:
    """Intersection over the smaller box (1.0 when one box contains the other)."""
    m = min(area(a), area(b))
    return inter_area(a, b) / m if m > 0 else 0.0


def box_dist(a: Box, b: Box) -> float:
    """Euclidean gap between two boxes (0 when they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def point_in(x: float, y: float, zone: Box) -> bool:
    return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]


def median_box(boxes) -> Box:
    arr = np.asarray(list(boxes), dtype=np.float32)
    return tuple(float(v) for v in np.median(arr, axis=0))


# ---- boolean series ------------------------------------------------------------------------
def runs(values: list[bool]) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] of True runs."""
    res, i = [], 0
    while i < len(values):
        if values[i]:
            j = i
            while j < len(values) and values[j]:
                j += 1
            res.append((i, j))
            i = j
        else:
            i += 1
    return res


def fill_gaps(values: list[bool], max_gap: int) -> list[bool]:
    """Bridge False gaps of length <= max_gap between True runs (tolerate missed detections)."""
    out = list(values)
    rs = runs(values)
    for (_, b1), (a2, _) in zip(rs, rs[1:]):
        if a2 - b1 <= max_gap:
            for k in range(b1, a2):
                out[k] = True
    return out


def running_median(values: list[float], valid: list[bool], half_window: int) -> list[float]:
    """Median of `values` over [i - w, i + w] restricted to `valid` frames (0.0 when none are valid)."""
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half_window), min(len(values), i + half_window + 1)
        vals = [values[j] for j in range(lo, hi) if valid[j]]
        out.append(float(np.median(vals)) if vals else 0.0)
    return out


# ---- per-class cluster tracks --------------------------------------------------------------
@dataclass
class Cluster:
    """One spatial cluster of same-class boxes over the video (a 'place' where that class was seen).

    rec[frame] = (best box, best score) of the frame; static/age are filled by ObjectModel.
    """
    cid: int
    label: str
    rec: dict[int, tuple[Box, float]] = field(default_factory=dict)
    last_frame: int = -1
    background: bool = False
    static: dict[int, bool] = field(default_factory=dict)
    age: dict[int, int] = field(default_factory=dict)
    on_desk: dict[int, bool] = field(default_factory=dict)
    masked: dict[int, bool] = field(default_factory=dict)
    _median: Box | None = None
    _median_n: int = 0

    @property
    def n(self) -> int:
        return len(self.rec)

    @property
    def frames(self) -> list[int]:
        return sorted(self.rec)

    def box(self, i: int) -> Box | None:
        r = self.rec.get(i)
        return r[0] if r else None

    def score(self, i: int) -> float:
        r = self.rec.get(i)
        return r[1] if r else 0.0

    def median(self) -> Box:
        """Median box, refreshed lazily as the cluster grows (cheap enough for the matching loop)."""
        if self._median is None or self.n >= max(self._median_n + 5, int(self._median_n * 1.25)):
            self._median = median_box(b for b, _ in self.rec.values())
            self._median_n = self.n
        return self._median

    def add(self, i: int, box: Box, score: float) -> None:
        if i not in self.rec or self.rec[i][1] < score:
            self.rec[i] = (box, score)
        self.last_frame = max(self.last_frame, i)

    def segments(self, max_gap: int = 2) -> list[list[int]]:
        """Visibility runs: lists of frames where consecutive sightings are <= max_gap frames apart."""
        segs: list[list[int]] = []
        for f in self.frames:
            if segs and f - segs[-1][-1] <= max_gap + 1:
                segs[-1].append(f)
            else:
                segs.append([f])
        return segs


def cluster_tracks(dets_per_frame: list[list[Detection]], labels: tuple[str, ...], min_score: float,
                   iou_thr: float = 0.5, contain_thr: float = 0.7, recent: int = 60
                   ) -> tuple[dict[int, Cluster], list[list[tuple[Detection, int]]]]:
    """Greedy per-class clustering of boxes over the whole video.

    A box joins the same-class cluster with the best IoU (>= iou_thr) against the cluster's MEDIAN box - never
    against its last box: chaining frame to frame would glue a bag carried in the hands to the place where it is
    put down 6 s later (X-15) and make the placed bag look like furniture. A slowly pushed object (X-17 pouch) splits
    into two clusters instead, which ObjectModel merges back into one run. A box that is a duplicate / sub-box of
    the median (>= contain_thr of the smaller inside the larger, areas within 4x) joins too, so NMS leftovers do
    not spawn 'new objects'. Candidates are clusters seen in the last `recent` frames or seen >= 3 times
    (furniture re-attaches after long occlusions).
    Returns (clusters by id, per-frame [(detection, cluster id)] for detections with label in `labels` and
    score >= min_score).
    """
    clusters: dict[int, Cluster] = {}
    by_label: dict[str, list[Cluster]] = {}
    assign: list[list[tuple[Detection, int]]] = []
    for i, dets in enumerate(dets_per_frame):
        row: list[tuple[Detection, int]] = []
        objs = sorted((d for d in dets if d.label in labels and d.score >= min_score), key=lambda d: -d.score)
        for d in objs:
            best, best_sim = None, 0.0
            for c in by_label.get(d.label, ()):
                if c.n < 3 and i - c.last_frame > recent:
                    continue
                med = c.median()
                sim = iou(d.box, med)
                if sim < iou_thr and overlap_min(d.box, med) >= contain_thr:
                    big, small = max(area(d.box), area(med)), min(area(d.box), area(med))
                    if small > 0 and big / small <= 4.0:
                        sim = iou_thr
                if sim >= iou_thr and sim > best_sim:
                    best, best_sim = c, sim
            if best is None:
                best = Cluster(len(clusters), d.label)
                clusters[best.cid] = best
                by_label.setdefault(d.label, []).append(best)
            best.add(i, d.box, d.score)
            row.append((d, best.cid))
        assign.append(row)
    return clusters, assign
