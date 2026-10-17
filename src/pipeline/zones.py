"""Scene layout and zones derived from detections only (no pixel constants).

Static objects (desk, laptop, counting machine, safe, door) are found by clustering their
boxes across the whole video and taking robust medians. Zones are expressed relative to
those boxes, so a different camera angle in the test set yields different pixel zones
automatically. Person-related zones use the people themselves: the cashier seat is the
median box of people who sit behind the laptop/counting machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.detect import Detection
from src.pipeline.track import Track, iou

Box = tuple[float, float, float, float]


def expand(b: Box, fx: float, fy: float | None = None) -> Box:
    fy = fx if fy is None else fy
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
    return (cx - w * fx / 2, cy - h * fy / 2, cx + w * fx / 2, cy + h * fy / 2)


def inter_area(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center_in(b: Box, zone: Box) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def overlap_frac(b: Box, zone: Box) -> float:
    """Fraction of `b` that lies inside `zone`."""
    a = area(b)
    return inter_area(b, zone) / a if a > 0 else 0.0


def cluster_boxes(boxes: list[Box], iou_thresh: float = 0.5) -> list[tuple[Box, int]]:
    """Greedy IoU clustering -> list of (median box, count), most frequent first."""
    clusters: list[list[Box]] = []
    for b in boxes:
        for c in clusters:
            if iou(b, _median(c)) >= iou_thresh:
                c.append(b)
                break
        else:
            clusters.append([b])
    out = [(_median(c), len(c)) for c in clusters]
    return sorted(out, key=lambda x: -x[1])


def _median(boxes: list[Box]) -> Box:
    arr = np.array(boxes, dtype=np.float32)
    return tuple(float(v) for v in np.median(arr, axis=0))


@dataclass
class SceneLayout:
    frame_w: int
    frame_h: int
    n_frames: int
    desk: Box | None = None
    laptop: Box | None = None
    counter_machine: Box | None = None
    safe: Box | None = None
    door: Box | None = None
    cashier_seat: Box | None = None          # median box of the person sitting behind the laptop
    presence: dict[str, float] = field(default_factory=dict)   # class -> fraction of frames seen

    # ---- derived zones -------------------------------------------------------------
    @property
    def anchor(self) -> Box | None:
        """Union of laptop and counting machine: what the cashier sits behind."""
        bs = [b for b in (self.laptop, self.counter_machine) if b]
        if not bs:
            return None
        return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))

    @property
    def desk_zone(self) -> Box | None:
        return self.desk

    @property
    def cashier_zone(self) -> Box | None:
        if self.cashier_seat:
            return expand(self.cashier_seat, 1.3, 1.2)
        return expand(self.anchor, 1.6, 2.2) if self.anchor else None

    @property
    def near_desk_zone(self) -> Box | None:
        return expand(self.desk, 1.3, 1.6) if self.desk else None

    @property
    def safe_zone(self) -> Box | None:
        return expand(self.safe, 1.6, 1.4) if self.safe else None

    @property
    def door_zone(self) -> Box | None:
        return expand(self.door, 1.1, 1.05) if self.door else None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d.update(cashier_zone=self.cashier_zone, near_desk_zone=self.near_desk_zone,
                 safe_zone=self.safe_zone, door_zone=self.door_zone, anchor=self.anchor)
        return d


def _pick(dets_per_frame: list[list[Detection]], label: str, min_score: float, min_frac: float,
          n_frames: int, prefer=None) -> tuple[Box | None, float]:
    boxes = [d.box for dets in dets_per_frame for d in dets if d.label == label and d.score >= min_score]
    if not boxes:
        return None, 0.0
    clusters = cluster_boxes(boxes)
    if prefer:
        clusters = sorted(clusters, key=prefer)
    box, count = clusters[0]
    frac = count / max(1, n_frames)
    return (box, frac) if frac >= min_frac else (None, frac)


def build_layout(dets_per_frame: list[list[Detection]], frame_w: int, frame_h: int,
                 tracks: dict[int, Track] | None = None) -> SceneLayout:
    n = len(dets_per_frame)
    lay = SceneLayout(frame_w, frame_h, n)
    frame_area = float(frame_w * frame_h)
    lay.laptop, lay.presence["laptop"] = _pick(dets_per_frame, "laptop", 0.4, 0.2, n)
    lay.counter_machine, lay.presence["counter_machine"] = _pick(dets_per_frame, "counter_machine", 0.3, 0.15, n)
    anchor = lay.anchor
    # desk: the cluster that overlaps the anchor most (the cashier's desk has the laptop on it),
    # not the big foreground table; ignore boxes covering most of the frame.
    def desk_pref(c):
        box, count = c
        if area(box) > 0.6 * frame_area:
            return (1, -count)
        ov = inter_area(box, anchor) / area(anchor) if anchor and area(anchor) > 0 else 0.0
        return (0, -ov, -count)
    lay.desk, lay.presence["desk"] = _pick(dets_per_frame, "desk", 0.3, 0.1, n, prefer=desk_pref)
    # safe: most frequent cluster that is not huge and not the desk
    def safe_pref(c):
        box, count = c
        bad = area(box) > 0.35 * frame_area or (lay.desk and iou(box, lay.desk) > 0.5)
        return (1 if bad else 0, -count)
    lay.safe, lay.presence["safe"] = _pick(dets_per_frame, "safe", 0.3, 0.1, n, prefer=safe_pref)
    # door: tall cluster in the upper part of the frame, not covering most of the frame
    def door_pref(c):
        box, count = c
        w, h = box[2] - box[0], box[3] - box[1]
        bad = area(box) > 0.5 * frame_area or h < w or box[1] > 0.5 * frame_h
        return (1 if bad else 0, -count)
    lay.door, lay.presence["door"] = _pick(dets_per_frame, "door", 0.25, 0.1, n, prefer=door_pref)
    # cashier seat: median box of people overlapping the anchor
    if anchor and tracks:
        seat_boxes = []
        for t in tracks.values():
            for b in t.boxes.values():
                if inter_area(b, anchor) / area(anchor) >= 0.15:
                    seat_boxes.append(b)
        if len(seat_boxes) >= max(3, 0.1 * n):
            lay.cashier_seat = _median(seat_boxes)
    return lay
