"""Safe anchor, 'attending the safe' and 'door open by state change' (spec P4, P8)."""
from __future__ import annotations

from src.pipeline.detect import Detection
from src.pipeline.frames import FrameSet
from src.pipeline.zones import SceneLayout, area, inter_area

Box = tuple[float, float, float, float]


def pick_safe_anchor(dets_per_frame: list[list[Detection]], layout: SceneLayout) -> Box | None:
    """P8: choose the safe box or None. Candidates: clusters of 'safe' and 'safe_door_open' boxes with aspect
    w/h 0.6-1.0, adjacent to the back/side edge of the desk (distance <= 0.5 desk width), bottom edge not below
    desk.y1 + 0.1*desk_h, centre NOT inside the permanent glass/door boxes (safe_door_open 0.28-0.40 on the whole
    glass every frame in camera A: (6,1,118,313), (1,0,221,319)), not white (needs no pixels: use the door box),
    merge vertically adjacent clusters (X-39: (232,258,339,433)). Presence of the chosen cluster must be >= 0.5
    of frames, else return None (X-11/13/14/15/31/46 anchor on the glass door; X-24-29 on the white cabinet under
    the desk; X-12/43 cupboard behind the glass; X-8 shifted 60 px). Update layout.presence['safe']."""
    raise NotImplementedError


def attending_safe_series(persons_raw: list[list[Detection]], hands_per_frame: list[list[Detection]], cm,
                          layout: SceneLayout) -> list[bool]:
    """P4 per frame: someone is at the safe: any in-room person box with x-overlap >= 0.25 of the safe width, or
    whose edge is within 0.3 safe widths of the safe edge (X-48 218-246 vs 229), or a hand box (score >= 0.25)
    with centre inside expand(safe, 1.5, 1.4) (X-39), or the cashier seated within one desk height of the safe
    with the box extending towards the safe (X-35 x1 199-202 -> 254-268). All False when layout.safe is None.
    Note: cm.box[i] is the cashier box; persons_raw boxes are the dedup'd people."""
    raise NotImplementedError


def door_open_series(dets_per_frame: list[list[Detection]], frames: FrameSet, layout: SceneLayout,
                     persons_raw: list[list[Detection]], with_pixels: bool = True) -> tuple[list[bool], list[float]]:
    """P8 per frame: (door_open, gated safe_door_open score). door_open = >= 2 of 3 votes in >= 3 of the last 5 frames:
    1. a new safe_door_open box with IoU(B0) < 0.5 shifted >= 10% of the safe width towards the room/desk, or the
       cabinet silhouette >= 20% wider than its baseline (X-39 (250,259,338,366)->(235-246,289,291,365-376);
       X-13 (162,199,221,272)->(145,217,183,271); X-48 (230,281,268,374)->(221,301,266,381); X-35 75->95 px);
    2. gated score - p20(gated score over the video) >= 0.10 (X-5/48 0.33-0.42->0.48-0.53, X-19 0.23-0.34->0.47-0.53,
       X-20) OR baseline-box score drop >= 0.15 (X-39 0.45->0.24-0.34);
    3. with_pixels: mean absolute pixel difference of the safe ROI vs the median of the first 5 s, in frames where no
       person box overlaps the ROI (dark interior visible: X-19 37.5-40, X-20).
    Gated score = max safe_door_open score of boxes whose centre is inside expand(safe,1.5,1.4) and area < 1.5 x safe
    area; 0.0 when layout.safe is None (then door_open is all False)."""
    raise NotImplementedError
