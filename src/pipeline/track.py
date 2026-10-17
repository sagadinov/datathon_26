"""Lightweight multi-object tracker for people at low frame rates (2 fps).

A greedy IoU + centre-distance matcher with a lost-track buffer is enough here: people move
slowly relative to a 0.5 s sampling step, there are at most 3-4 of them, and the camera is
static. No external dependency (supervision 0.31 dropped its ByteTrack wrapper).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.detect import Detection


def iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    track_id: int
    boxes: dict[int, tuple] = field(default_factory=dict)   # frame index -> box
    scores: dict[int, float] = field(default_factory=dict)
    last_seen: int = -1
    misses: int = 0

    @property
    def first_seen(self) -> int:
        return min(self.boxes) if self.boxes else -1

    @property
    def n_frames(self) -> int:
        return len(self.boxes)

    def box_at(self, i: int) -> tuple | None:
        return self.boxes.get(i)


class GreedyTracker:
    """Assign stable ids to per-frame boxes. `update()` returns the id for each input box."""

    def __init__(self, iou_thresh: float = 0.3, max_center_dist: float = 0.25, lost_frames: int = 6):
        self.iou_thresh, self.max_center_dist, self.lost_frames = iou_thresh, max_center_dist, lost_frames
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, frame_idx: int, boxes: list[tuple], scores: list[float], frame_diag: float) -> list[int]:
        active = [t for t in self.tracks.values() if t.misses <= self.lost_frames]
        ids = [-1] * len(boxes)
        if active and boxes:
            cost = np.zeros((len(active), len(boxes)))
            for i, t in enumerate(active):
                tb = t.boxes[t.last_seen]
                tc = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
                for j, b in enumerate(boxes):
                    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                    dist = np.hypot(tc[0] - bc[0], tc[1] - bc[1]) / frame_diag
                    ov = iou(tb, b)
                    # similarity: IoU, or proximity for fast movers; penalise long-lost tracks slightly
                    cost[i, j] = -(ov + max(0.0, 1 - dist / self.max_center_dist) * 0.5) + 0.02 * t.misses
            used_t, used_b = set(), set()
            for i, j in sorted(((i, j) for i in range(len(active)) for j in range(len(boxes))),
                               key=lambda ij: cost[ij]):
                if i in used_t or j in used_b or cost[i, j] >= -0.15:
                    continue
                used_t.add(i); used_b.add(j)
                ids[j] = active[i].track_id
        for j, b in enumerate(boxes):
            if ids[j] == -1:
                ids[j] = self._next
                self.tracks[self._next] = Track(self._next)
                self._next += 1
            t = self.tracks[ids[j]]
            t.boxes[frame_idx] = tuple(b)
            t.scores[frame_idx] = float(scores[j])
            t.last_seen, t.misses = frame_idx, 0
        matched = set(ids)
        for t in self.tracks.values():
            if t.track_id not in matched and t.last_seen < frame_idx:
                t.misses += 1
        return ids


def track_boxes(persons_per_frame: list[list[Detection]], frame_w: int, frame_h: int) -> tuple[list[list[int]], dict[int, Track]]:
    """Track already-filtered person detections (one list per frame). Returns per-frame ids (parallel to
    the input lists) and the tracks."""
    tracker = GreedyTracker()
    diag = float(np.hypot(frame_w, frame_h))
    per_frame_ids: list[list[int]] = []
    for i, people in enumerate(persons_per_frame):
        per_frame_ids.append(tracker.update(i, [d.box for d in people], [d.score for d in people], diag))
    return per_frame_ids, tracker.tracks


def track_people(dets_per_frame: list[list[Detection]], frame_w: int, frame_h: int,
                 min_score: float = 0.35, min_height_frac: float = 0.12) -> tuple[list[list[int]], dict[int, Track]]:
    """Track `person` detections. Returns per-frame [(det index -> track id)] and the tracks."""
    tracker = GreedyTracker()
    diag = float(np.hypot(frame_w, frame_h))
    per_frame_ids: list[list[int]] = []
    for i, dets in enumerate(dets_per_frame):
        people = [(k, d) for k, d in enumerate(dets)
                  if d.label == "person" and d.score >= min_score and (d.box[3] - d.box[1]) >= min_height_frac * frame_h]
        ids = tracker.update(i, [d.box for _, d in people], [d.score for _, d in people], diag)
        row = [-1] * len(dets)
        for (k, _), tid in zip(people, ids):
            row[k] = tid
        per_frame_ids.append(row)
    return per_frame_ids, tracker.tracks
