"""People primitives (spec: docs/violations_observed.md P1, P2, P3, P9).

P2  dedup_persons      - remove duplicate / ghost person boxes per frame
P1  in_room            - person is inside the cash room, not behind the glass door
P3  CashierModel       - ONE cashier per video, per-frame present / seat_vacated / box, exit time
P9  roles_and_events   - at_desk / far_side / at_door flags, entered / left events

All thresholds are relative (desk height, door height, cashier height); pixels only in comments.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline.detect import Detection
from src.pipeline.track import Track, iou
from src.pipeline.zones import SceneLayout, area, inter_area

Box = tuple[float, float, float, float]


def dedup_persons(person_dets: list[Detection], frame_h: int, min_score: float = 0.35,
                  min_h_frac: float = 0.12) -> list[Detection]:
    """P2: keep person boxes with score >= min_score and height >= min_h_frac * frame_h, then drop a box
    if IoU >= 0.45 with a more confident box, or >= 70% of its area lies inside a more confident box
    (X-43: IoU 0.54 'upper half' duplicate; X-3/8/14/24/26/27/32/33/45). Return the survivors sorted by score."""
    raise NotImplementedError


def floor_line(layout: SceneLayout) -> float | None:
    """P1: y of the 'floor line': door_zone.y2 - 0.02*H when the door is known (camera A ~278-280 px,
    camera B ~290 px, X-32 door.y2=342); fallback desk.y0 - 0.1*desk_h; None if neither is known."""
    raise NotImplementedError


def in_room(box: Box, layout: SceneLayout, ref_height: float | None) -> bool:
    """P1: True when the person stands inside the room: box.y2 >= floor_line AND (ref_height is None or
    box height >= 0.5 * ref_height). Additionally OUTSIDE when the box lies entirely inside the door zone
    and box.y1 < door.y0 + 0.38 * door_h (X-13/X-15: outside y1 <= 121, inside standing y1 >= 126)."""
    raise NotImplementedError


class CashierModel:
    """P3: cashier identity for the whole video (not per frame).

    Construction (first pass, in_room0 computed without a reference height):
      * candidate tracks = tracks with in_room frames; score each by the number of frames whose box centre
        lies inside layout.cashier_seat (or whose box overlaps layout.anchor by >= 0.15 of the anchor when
        the seat is unknown); fallback (X-30/X-31, no laptop): first in_room track overlapping counter_machine
        >= 2 frames; merge ids that hand their box over between consecutive frames (IoU > 0.4).
      * self.track_id   - main cashier track id (None if no person at all)
      * self.ref_height - median box height of the cashier while seated (first 10 s of presence)
      * self.seat_box   - median cashier box while seated (reference for seat IoU)
      * self.room_empty_start - no in_room person during the first 2 s (X-30/31/32/33/40)
    finalize(in_room) (second pass with the final in_room flags) fills per-frame lists of length n:
      * self.box[i]      - cashier box this frame (in-room box of the cashier track or any in-room box with
                           IoU(seat_box) >= 0.3), else None; hold the previous box for <= 2 frames of dropout
      * self.present[i]  - cashier box exists AND (IoU(seat_box) >= 0.3 OR desk_gap <= 0.10)
      * self.vacated[i]  - present box height >= 1.35 * ref_height OR IoU(seat_box) < 0.3 (standing / away);
                           True when no cashier box (X-28 75->130 px, X-12 82->126-137)
      * self.exit_t      - first t where the cashier box centre entered door_zone with y2 <= door_zone.y2 + 20 px
                           and afterwards no in-room cashier for >= 2 frames (or the clip ends within 3 s with
                           >= 2 empty frames): X-11 18.0, X-22 16.5, X-28 10-11.5, X-15 15.5, X-26 15, X-24 10.5,
                           X-13 61, X-23 14.5, X-29 13, X-27 22. None when the cashier never leaves.
    is_cashier(i, k): the k-th dedup'd person box of frame i is the cashier box of that frame.
    """

    def __init__(self, persons_raw: list[list[Detection]], ids_per_frame: list[list[int]], tracks: dict[int, Track],
                 layout: SceneLayout, in_room0: list[list[bool]], sample_fps: float):
        raise NotImplementedError

    def finalize(self, in_room: list[list[bool]]) -> None:
        raise NotImplementedError

    def is_cashier(self, i: int, k: int) -> bool:
        raise NotImplementedError


@dataclass
class Roles:
    """P9 output. flags[i][k] = (desk_gap, at_desk, far_side, at_door, seat_iou) for the k-th person of frame i."""
    flags: list[list[tuple[float, bool, bool, bool, float]]]
    entered: list[tuple[float, int]] = field(default_factory=list)   # (t, track_id): non-cashier became in_room
    left: list[tuple[float, int]] = field(default_factory=list)      # (t, track_id): non-cashier left the room

    def person_flags(self, i: int, k: int) -> tuple[float, bool, bool, bool, float]:
        return self.flags[i][k]


def roles_and_events(persons_raw: list[list[Detection]], ids_per_frame: list[list[int]], in_room: list[list[bool]],
                     cm: CashierModel, layout: SceneLayout, sample_fps: float) -> Roles:
    """P9: per-person flags and visitor events.
    desk_gap: distance box<->desk in desk heights (0 = touching); at_desk = in_room and desk_gap <= 0.10;
    far_side: centre on the opposite side of the desk from the cashier seat (camera A: client y2 276-285 <
    desk.y0 while cashier y2 ~428; camera B: client in front of the desk); at_door: box overlaps door_zone
    (IoU > 0.1) or centre within 0.5 door width of it; seat_iou: IoU with cm.seat_box.
    entered(t, id): a non-cashier track becomes in_room right after >= 1 frame in door_zone (X-4 57.0,
    X-38 32-33, X-18 17.5-18, X-30 12.5). left(t, id): an in_room non-cashier stops being in_room for >= 2 frames
    or moves into door_zone and is lost (X-4 52-53, X-34 12-13, X-38 28 & 54, X-17 12.0). End a visitor track at
    the door so its id is not reused for the next visitor."""
    raise NotImplementedError
