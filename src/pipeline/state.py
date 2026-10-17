"""Per-frame scene state ("timeline") built from detections, tracks and the scene layout.

The rule engine reasons over this timeline only. Everything here is relative to detected
objects (zones.py), so no pixel constants. Objects "on the desk" are detections of movable
classes whose centre lies inside the desk box; people get zone memberships and a cashier flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.detect import Detection
from src.pipeline.frames import FrameSet
from src.pipeline.track import Track, track_people
from src.pipeline.zones import SceneLayout, build_layout, center_in, expand, inter_area, area, overlap_frac

Box = tuple[float, float, float, float]
DESK_OBJECTS = ("cash", "money_bag", "bag", "cup", "book", "jacket", "phone")
OBJ_MIN_SCORE = {"cash": 0.30, "money_bag": 0.30, "bag": 0.30, "cup": 0.30, "book": 0.30, "jacket": 0.30, "phone": 0.30}


@dataclass
class PersonState:
    track_id: int
    box: Box
    score: float
    anchor_overlap: float = 0.0      # fraction of the laptop/counter anchor covered by this person
    desk_gap: float = 9.0            # distance to the desk box in desk-heights (0 = touching/overlapping)
    at_seat: bool = False            # box overlaps the cashier seat (median cashier box)
    in_cashier_zone: bool = False
    near_desk: bool = False
    at_door: bool = False
    at_safe: bool = False
    is_cashier: bool = False


@dataclass
class ObjState:
    label: str
    score: float
    box: Box
    on_desk: bool


@dataclass
class FrameState:
    index: int
    t: float
    persons: list[PersonState] = field(default_factory=list)
    objects: list[ObjState] = field(default_factory=list)     # movable objects (all, with on_desk flag)
    safe_door_open: float = 0.0                                # gated max score
    brightness: float = 0.0                                    # mean gray 0..255
    laptop_bright: float = 0.0                                 # mean gray inside the laptop box
    laptop_white: float = 0.0                                  # fraction of laptop-box pixels > 170
    hands_below_desk: int = 0

    @property
    def lights_off(self) -> bool:
        return self.brightness < 110

    @property
    def screen_on(self) -> bool:
        return self.laptop_white >= 0.10 and self.laptop_bright >= self.brightness - 5

    @property
    def cashier(self) -> PersonState | None:
        return next((p for p in self.persons if p.is_cashier), None)

    @property
    def n_persons(self) -> int:
        return len(self.persons)

    def on_desk(self, label: str, min_score: float | None = None) -> list[ObjState]:
        ms = OBJ_MIN_SCORE.get(label, 0.3) if min_score is None else min_score
        return [o for o in self.objects if o.label == label and o.on_desk and o.score >= ms]


@dataclass
class Timeline:
    layout: SceneLayout
    frames: list[FrameState]
    tracks: dict[int, Track]
    sample_fps: float

    def __len__(self) -> int:
        return len(self.frames)

    def cashier_track_id(self) -> int | None:
        votes: dict[int, int] = {}
        for f in self.frames:
            c = f.cashier
            if c:
                votes[c.track_id] = votes.get(c.track_id, 0) + 1
        return max(votes, key=votes.get) if votes else None


def _brightness(frames: FrameSet, laptop: Box | None) -> list[tuple[float, float, float]]:
    """(frame mean gray, laptop-box mean gray, fraction of bright pixels in the laptop box)."""
    import cv2

    out = []
    for f in frames:
        img = cv2.imread(str(f.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            out.append((0.0, 0.0, 0.0))
            continue
        lb, lw = 0.0, 0.0
        if laptop:
            x1, y1, x2, y2 = (int(round(v)) for v in laptop)
            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size:
                lb, lw = float(crop.mean()), float((crop > 170).mean())
        out.append((float(img.mean()), lb, lw))
    return out


def desk_gap(b: Box, desk: Box) -> float:
    dh = max(1.0, desk[3] - desk[1])
    dx = max(desk[0] - b[2], b[0] - desk[2], 0.0)
    dy = max(desk[1] - b[3], b[1] - desk[3], 0.0)
    return float(np.hypot(dx, dy) / dh)


def build_timeline(frames: FrameSet, dets_per_frame: list[list[Detection]], with_brightness: bool = True) -> Timeline:
    w, h = frames.meta.width, frames.meta.height
    if frames.frames:
        import cv2

        img = cv2.imread(str(frames.frames[0].path))
        if img is not None:
            h, w = img.shape[:2]
    ids_per_frame, tracks = track_people(dets_per_frame, w, h)
    layout = build_layout(dets_per_frame, w, h, tracks)
    bright = _brightness(frames, layout.laptop) if with_brightness else [(0.0, 0.0, 0.0)] * len(frames)
    anchor, desk = layout.anchor, layout.desk
    seat = layout.cashier_seat
    cz, ndz, sz, dz = layout.cashier_zone, layout.near_desk_zone, layout.safe_zone, layout.door_zone
    desk_in = expand(desk, 1.05, 1.1) if desk else None
    states: list[FrameState] = []
    for i, (frame, dets) in enumerate(zip(frames, dets_per_frame)):
        st = FrameState(i, frame.timestamp_sec, brightness=bright[i][0], laptop_bright=bright[i][1],
                        laptop_white=bright[i][2])
        persons = []
        for d, tid in zip(dets, ids_per_frame[i]):
            if tid < 0:
                continue
            p = PersonState(tid, d.box, d.score)
            if anchor:
                p.anchor_overlap = inter_area(d.box, anchor) / area(anchor)
            if desk:
                p.desk_gap = desk_gap(d.box, desk)
            if seat:
                p.at_seat = inter_area(d.box, seat) / max(1.0, area(seat)) > 0.4
            p.in_cashier_zone = bool(cz and (center_in(d.box, cz) or overlap_frac(d.box, cz) > 0.5))
            p.near_desk = bool(ndz and inter_area(d.box, ndz) > 0)
            p.at_door = bool(dz and overlap_frac(d.box, dz) > 0.6)
            if layout.safe:
                sb = layout.safe
                x_ov = max(0.0, min(d.box[2], sb[2]) - max(d.box[0], sb[0])) / max(1.0, sb[2] - sb[0])
                p.at_safe = x_ov >= 0.5 and inter_area(d.box, sz) > 0
            persons.append(p)
        # cashier: the person covering the anchor most (ties -> inside cashier zone)
        if persons:
            best = max(persons, key=lambda p: (p.anchor_overlap, p.in_cashier_zone, -p.at_door))
            if best.anchor_overlap >= 0.3 or (best.in_cashier_zone and best.anchor_overlap > 0.05):
                best.is_cashier = True
        st.persons = persons
        for d in dets:
            if d.label in DESK_OBJECTS and d.score >= 0.2:
                on = bool(desk_in and center_in(d.box, desk_in))
                st.objects.append(ObjState(d.label, d.score, d.box, on))
            elif d.label == "safe_door_open" and layout.safe:
                gate = expand(layout.safe, 1.5, 1.4)
                if center_in(d.box, gate) and area(d.box) < 1.5 * area(layout.safe):
                    st.safe_door_open = max(st.safe_door_open, d.score)
            elif d.label == "hand" and desk and d.score >= 0.25:
                if (d.box[1] + d.box[3]) / 2 > desk[3]:
                    st.hands_below_desk += 1
        states.append(st)
    return Timeline(layout, states, tracks, frames.sample_fps)


def smooth_bool(values: list[bool], min_run: int) -> list[bool]:
    """Keep only runs of True of length >= min_run (temporal debouncing)."""
    out = [False] * len(values)
    i = 0
    while i < len(values):
        if values[i]:
            j = i
            while j < len(values) and values[j]:
                j += 1
            if j - i >= min_run:
                for k in range(i, j):
                    out[k] = True
            i = j
        else:
            i += 1
    return out


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
