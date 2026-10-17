"""People primitives (spec: docs/violations_observed.md P1, P2, P3, P9).

P2 dedup_persons, P1 in_room, P3 CashierModel (one cashier per video, present / vacated / exit),
P9 roles_and_events (at_desk / far_side / at_door, entered / left). Thresholds are relative to the
desk, door and cashier sizes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.detect import Detection
from src.pipeline.track import Track, iou
from src.pipeline.zones import SceneLayout, area, inter_area

Box = tuple[float, float, float, float]


def dedup_persons(person_dets: list[Detection], frame_h: int, min_score: float = 0.35,
                  min_h_frac: float = 0.12) -> list[Detection]:
    """P2: drop weak/small boxes, then duplicates (IoU >= 0.45 or >= 70% inside a stronger box)."""
    cands = sorted([d for d in person_dets if d.score >= min_score and (d.box[3] - d.box[1]) >= min_h_frac * frame_h],
                   key=lambda d: -d.score)
    keep: list[Detection] = []
    for d in cands:
        if any(iou(d.box, k.box) >= 0.45 or inter_area(d.box, k.box) / max(1.0, area(d.box)) >= 0.7 for k in keep):
            continue
        keep.append(d)
    return keep


def desk_gap(b: Box, desk: Box | None) -> float:
    if desk is None:
        return 9.0
    dh = max(1.0, desk[3] - desk[1])
    dx = max(desk[0] - b[2], b[0] - desk[2], 0.0)
    dy = max(desk[1] - b[3], b[1] - desk[3], 0.0)
    return float(np.hypot(dx, dy) / dh)


def floor_line(layout: SceneLayout) -> float | None:
    """P1: people whose feet (box bottom) are above this line stand behind the glass."""
    H = layout.frame_h
    fl = None
    if layout.door_zone:
        fl = layout.door_zone[3] - 0.02 * H
    if layout.desk:
        dh = layout.desk[3] - layout.desk[1]
        desk_fl = layout.desk[1] + 0.06 * dh      # anyone touching the desk from behind is inside
        fl = desk_fl if fl is None else min(fl, desk_fl)
    return fl


def in_room(box: Box, layout: SceneLayout, ref_height: float | None) -> bool:
    """P1: inside the room unless the box sits in the door/glass area and is high or small."""
    fl = floor_line(layout)
    h = box[3] - box[1]
    small = ref_height is not None and h < 0.5 * ref_height
    high = fl is not None and box[3] < fl
    touching = desk_gap(box, layout.desk) <= 0.05
    if high and not touching:
        return False
    if layout.door:
        in_door = inter_area(box, layout.door) / max(1.0, area(box)) >= 0.5
        if in_door and (high or small):
            return False
    return True


class CashierModel:
    """P3: one cashier per video; per-frame box / present / vacated and the exit time."""

    def __init__(self, persons_raw: list[list[Detection]], ids_per_frame: list[list[int]], tracks: dict[int, Track],
                 layout: SceneLayout, in_room0: list[list[bool]], sample_fps: float):
        self.persons, self.ids, self.tracks, self.layout, self.fps = persons_raw, ids_per_frame, tracks, layout, sample_fps
        n = len(persons_raw)
        self.n = n
        seat, anchor, counter = layout.cashier_seat, layout.anchor, layout.counter_machine
        score: dict[int, int] = {}
        for i in range(n):
            for k, d in enumerate(persons_raw[i]):
                if not in_room0[i][k]:
                    continue
                cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
                hit = False
                if seat and seat[0] <= cx <= seat[2] and seat[1] <= cy <= seat[3]:
                    hit = True
                elif anchor and inter_area(d.box, anchor) / max(1.0, area(anchor)) >= 0.15:
                    hit = True
                elif seat is None and anchor is None and counter and inter_area(d.box, counter) > 0:
                    hit = True
                if hit:
                    score[ids_per_frame[i][k]] = score.get(ids_per_frame[i][k], 0) + 1
        self.track_id = max(score, key=score.get) if score else None
        if self.track_id is None and tracks:       # fallback: longest in-room track
            lens = {tid: sum(1 for i in range(n) for k, t in enumerate(ids_per_frame[i]) if t == tid and in_room0[i][k])
                    for tid in tracks}
            self.track_id = max(lens, key=lens.get) if lens and max(lens.values()) > 0 else None
        # seated reference: boxes of the cashier track that scored (seat/anchor) during the first 10 s of presence
        ref_boxes = []
        first = None
        for i in range(n):
            for k, t in enumerate(ids_per_frame[i]):
                if t == self.track_id and in_room0[i][k]:
                    first = i if first is None else first
                    if i - first <= 10 * sample_fps:
                        b = persons_raw[i][k].box
                        if (seat and seat[0] <= (b[0] + b[2]) / 2 <= seat[2]) or (anchor and inter_area(b, anchor) > 0) or not seat:
                            ref_boxes.append(b)
        if ref_boxes:
            arr = np.array(ref_boxes, dtype=np.float32)
            self.seat_box: Box | None = tuple(float(v) for v in np.median(arr, axis=0))
            self.ref_height: float | None = float(np.median(arr[:, 3] - arr[:, 1]))
        else:
            self.seat_box, self.ref_height = seat, None
        first_frames = int(round(2 * sample_fps))
        self.room_empty_start = not any(any(in_room0[i]) for i in range(min(n, first_frames)))
        self.box: list[Box | None] = [None] * n
        self.present = [False] * n
        self.vacated = [True] * n
        self.exit_t: float | None = None
        self._cashier_k: list[int] = [-1] * n

    def finalize(self, in_room: list[list[bool]]) -> None:
        n, seat, desk = self.n, self.seat_box, self.layout.desk
        last_box, hold = None, 0
        for i in range(n):
            best_k, best_iou = -1, 0.0
            for k, d in enumerate(self.persons[i]):
                if not in_room[i][k]:
                    continue
                s = iou(d.box, seat) if seat else 0.0
                if s >= 0.3 and s > best_iou:
                    best_k, best_iou = k, s
            if best_k < 0:
                for k, t in enumerate(self.ids[i]):
                    if t == self.track_id and in_room[i][k]:
                        best_k = k
                        break
            if best_k >= 0:
                b = self.persons[i][best_k].box
                last_box, hold = b, 0
                self._cashier_k[i] = best_k
            elif last_box is not None and hold < 2:
                b, hold = last_box, hold + 1
            else:
                b = None
            self.box[i] = b
            if b is not None:
                s = iou(b, seat) if seat else 0.0
                self.present[i] = s >= 0.3 or desk_gap(b, desk) <= 0.10
                tall = self.ref_height is not None and (b[3] - b[1]) >= 1.35 * self.ref_height
                self.vacated[i] = tall or s < 0.3
            else:
                self.present[i], self.vacated[i] = False, True
        self.exit_t = self._find_exit(in_room)

    def _find_exit(self, in_room: list[list[bool]]) -> float | None:
        dz, H = self.layout.door_zone, self.layout.frame_h
        have = [self._cashier_k[i] >= 0 for i in range(self.n)]
        if not any(have):
            return None
        last_seen = max(i for i in range(self.n) if have[i])
        for i in range(self.n):
            if not have[i]:
                continue
            b = self.persons[i][self._cashier_k[i]].box
            at_door = dz is not None and dz[0] <= (b[0] + b[2]) / 2 <= dz[2] and dz[1] <= (b[1] + b[3]) / 2 <= dz[3] \
                and b[3] <= dz[3] + 0.03 * H
            if not at_door:
                continue
            after = have[i + 1:i + 3]
            if (len(after) >= 2 and not any(after)) or (len(after) < 2 and i >= self.n - 1 - 3 * self.fps and not any(after)):
                if not any(have[i + 1:]) or sum(have[i + 1:]) <= 2:
                    return float(i + 1) / self.fps if i + 1 < self.n else float(i) / self.fps
        # fallback: the cashier disappears for good at least 1 s before the clip ends
        if last_seen < self.n - 2:
            return float(last_seen + 1) / self.fps
        return None

    def is_cashier(self, i: int, k: int) -> bool:
        return self._cashier_k[i] == k


@dataclass
class Roles:
    flags: list[list[tuple[float, bool, bool, bool, float]]]
    entered: list[tuple[float, int]] = field(default_factory=list)
    left: list[tuple[float, int]] = field(default_factory=list)

    def person_flags(self, i: int, k: int) -> tuple[float, bool, bool, bool, float]:
        return self.flags[i][k]


def roles_and_events(persons_raw: list[list[Detection]], ids_per_frame: list[list[int]], in_room: list[list[bool]],
                     cm: CashierModel, layout: SceneLayout, sample_fps: float) -> Roles:
    desk, dz, seat = layout.desk, layout.door_zone, cm.seat_box
    cy_desk = (desk[1] + desk[3]) / 2 if desk else None
    cy_seat = (seat[1] + seat[3]) / 2 if seat else None
    flags: list[list[tuple[float, bool, bool, bool, float]]] = []
    for i in range(len(persons_raw)):
        row = []
        for k, d in enumerate(persons_raw[i]):
            g = desk_gap(d.box, desk)
            at_desk = in_room[i][k] and g <= 0.10
            cy = (d.box[1] + d.box[3]) / 2
            far = bool(cy_desk is not None and cy_seat is not None and (cy - cy_desk) * (cy_seat - cy_desk) < 0)
            at_door = False
            if dz:
                dw = dz[2] - dz[0]
                cx = (d.box[0] + d.box[2]) / 2
                at_door = iou(d.box, dz) > 0.1 or (dz[0] - 0.5 * dw <= cx <= dz[2] + 0.5 * dw and cy <= dz[3])
            s_iou = iou(d.box, seat) if seat else 0.0
            row.append((g, at_desk, far, at_door, s_iou))
        flags.append(row)
    # visitor events per track (non-cashier)
    entered, left = [], []
    per_track: dict[int, list[tuple[int, int]]] = {}
    for i in range(len(persons_raw)):
        for k, t in enumerate(ids_per_frame[i]):
            if not cm.is_cashier(i, k):
                per_track.setdefault(t, []).append((i, k))
    n = len(persons_raw)
    for t, obs in per_track.items():
        if t == cm.track_id:
            continue
        obs.sort()
        inside = [in_room[i][k] for i, k in obs]
        door = [flags[i][k][3] for i, k in obs]
        was_in = False
        for j, (i, k) in enumerate(obs):
            if inside[j] and not was_in:
                came_via_door = any(door[max(0, j - 2):j]) or (door[j] and i > 0 and j == 0)
                if came_via_door:
                    entered.append((i / sample_fps, t))
                was_in = True
            elif was_in and not inside[j]:
                nxt = inside[j + 1] if j + 1 < len(inside) else False
                if not nxt:
                    left.append((i / sample_fps, t))
                    was_in = False
        if was_in:
            last_i = obs[-1][0]
            if last_i < n - 2:        # track lost for >= 2 frames before the end
                left.append(((last_i + 1) / sample_fps, t))
    return Roles(flags, sorted(entered), sorted(left))
