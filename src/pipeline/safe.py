"""Safe anchor, 'attending the safe' and 'door open by state change' (spec P4, P8).

All thresholds are relative to the safe / desk / person box sizes (pixel values in comments are
examples at 350x640). Pixels are only read for vote 3 of door_open (via safe_util.RoiDiff).
"""
from __future__ import annotations

import numpy as np

from src.pipeline.detect import Detection
from src.pipeline.frames import FrameSet
from src.pipeline.safe_util import Cluster, RoiDiff, box_gap, cluster_frames, fill_gaps, frac_inside, x_overlap
from src.pipeline.track import iou
from src.pipeline.zones import SceneLayout, area, center_in, expand, inter_area

Box = tuple[float, float, float, float]

ANCHOR_MIN_PRESENCE = 0.5      # fraction of frames the chosen cluster must be seen in
ASPECT = (0.6, 1.0)            # w/h of a cabinet box
DESK_GAP = 0.3                 # max gap safe<->desk, in desk widths (real cabinets 0.00-0.06, cupboards behind glass 0.33+)
PERSON_MIN_SCORE, PERSON_MIN_H = 0.35, 0.12
HAND_MIN_SCORE = 0.25
PIXEL_DIFF_THR = 12.0          # mean abs grey difference of the door ROI vs its closed baseline (vote 3)
PIXEL_CHANGED_FRAC = 0.05      # ... and at least this fraction of ROI pixels changed by >= 40 grey levels


def _dims(b: Box) -> tuple[float, float]:
    return b[2] - b[0], b[3] - b[1]


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _person_boxes(dets_per_frame: list[list[Detection]], frame_h: int) -> list[list[Box]]:
    return [[d.box for d in dets if d.label == "person" and d.score >= PERSON_MIN_SCORE
             and d.box[3] - d.box[1] >= PERSON_MIN_H * frame_h] for dets in dets_per_frame]


# ----------------------------------------------------------------------------------------------- P8 anchor
def _cabinet_like(box: Box, desk: Box | None, glass: list[Box]) -> bool:
    """Geometry filter for a safe/cabinet candidate (see pick_safe_anchor)."""
    w, h = _dims(box)
    if w <= 0 or h <= 0 or not (ASPECT[0] <= w / h <= ASPECT[1]):
        return False
    if desk is not None:
        dw, dh = _dims(desk)
        gx, gy = box_gap(box, desk)
        if max(gx, gy) > DESK_GAP * dw:
            return False                     # cupboard behind the glass wall, far from the desk
        if box[1] > desk[1] + 0.05 * dh:
            return False                     # top edge below the desk top: a low drawer unit, not a safe
        if box[3] > desk[3] + 0.1 * dh:
            return False                     # bottom edge well below the desk: the white cabinet under the desk
    cy = (box[1] + box[3]) / 2
    for g in glass:
        if frac_inside(box, g) >= 0.9 and (cy - g[1]) / max(1.0, g[3] - g[1]) <= 0.65:
            return False                     # inside the permanent glass/door box, in its upper part: behind the glass
    return True


def _glass_like(box: Box, frame_area: float) -> bool:
    """Permanent safe_door_open boxes on the glass wall: tall panels (>= 8% of the frame, w/h <= 0.55) or the whole
    wall (>= 20% of the frame)."""
    w, h = _dims(box)
    a = area(box)
    return a >= 0.2 * frame_area or (a >= 0.08 * frame_area and h > 0 and w / h <= 0.55)


def _vertically_adjacent(a: Box, b: Box) -> bool:
    """b sits right above or below a (same column, touching or slightly overlapping), e.g. a drawer unit."""
    wa, ha = _dims(a)
    wb, hb = _dims(b)
    if x_overlap(a, b) < 0.7 * min(wa, wb):
        return False
    hmin = min(ha, hb)
    _, gy = box_gap(a, b)
    y_over = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if gy > 0.3 * hmin or y_over > 0.5 * hmin:
        return False
    cya, cyb = (a[1] + a[3]) / 2, (b[1] + b[3]) / 2
    if abs(cya - cyb) < 0.5 * hmin:
        return False
    uw, uh = _dims(_union(a, b))
    return 0.5 <= uw / uh <= 1.2


def pick_safe_anchor(dets_per_frame: list[list[Detection]], layout: SceneLayout) -> Box | None:
    """P8: choose the safe box or None. Candidates: clusters of 'safe' and 'safe_door_open' boxes with aspect
    w/h 0.6-1.0, adjacent to the back/side edge of the desk (distance <= 0.3 desk width), top edge not below the
    desk top, bottom edge not below desk.y2 + 0.1*desk_h, not lying inside the permanent glass/door boxes
    (safe_door_open 0.28-0.40 on the whole glass every frame in camera A: (6,1,118,313), (1,0,221,319)); ranked by
    presence x score, penalising clusters that sit on people (chairs / heads). A vertically adjacent cluster that the
    detector scores at least as high is merged (X-39: cabinet (250,259,338,367) on the drawer unit (232,329,334,433)
    -> (232,259,338,433)). Presence of the chosen cluster must be >= 0.5 of frames, else None (X-30; the glass door
    in X-11/13/14/15/31/46, the white cabinet under the desk in X-24-29, the cupboard behind the glass in X-12/43 and
    the shifted anchor of X-8 are all rejected by the filters). Updates layout.presence['safe']."""
    n = len(dets_per_frame)
    layout.presence["safe"] = 0.0
    items = [(i, d.box, d.score) for i, dets in enumerate(dets_per_frame)
             for d in dets if d.label in ("safe", "safe_door_open")]
    if not items or n == 0:
        return None
    clusters = cluster_frames(items, n, 0.6)
    frame_area = float(layout.frame_w * layout.frame_h)
    glass = [c.box for c in clusters if c.presence >= 0.5 and _glass_like(c.box, frame_area)]
    if layout.door is not None:
        glass.append(layout.door)
    persons = _person_boxes(dets_per_frame, layout.frame_h)
    best: Cluster | None = None
    best_rank, best_presence = 0.0, 0.0
    for c in clusters:
        if not _cabinet_like(c.box, layout.desk, glass):
            continue
        best_presence = max(best_presence, c.presence)
        if c.presence < ANCHOR_MIN_PRESENCE:
            continue
        occupied = float(np.mean([any(center_in(c.box, p) for p in persons[i]) for i in range(n)]))
        rank = c.presence * c.score * (1.0 - 0.5 * occupied)
        if rank > best_rank:
            best, best_rank = c, rank
    if best is None:
        layout.presence["safe"] = best_presence
        return None
    layout.presence["safe"] = best.presence
    box = best.box
    for c in clusters:
        if c is best or c.presence < ANCHOR_MIN_PRESENCE or c.score < best.score:
            continue
        if _vertically_adjacent(best.box, c.box):
            box = _union(box, c.box)
    return box


# ------------------------------------------------------------------------------------- P4 attending the safe
def _near_safe(b: Box, safe: Box, sw: float) -> bool:
    """Person box x-overlaps the safe by >= 0.25 of its width, or its edge is within 0.3 safe widths of the safe
    edge, while overlapping the safe vertically."""
    if min(b[3], safe[3]) - max(b[1], safe[1]) <= 0:
        return False
    gx, _ = box_gap(b, safe)
    return x_overlap(b, safe) >= 0.25 * sw or gx <= 0.3 * sw


def _edge_toward(b: Box, seat: Box, safe: Box) -> float:
    """How far the box's safe-side edge extends beyond the seat reference towards the safe (pixels)."""
    safe_right = (safe[0] + safe[2]) / 2 > (seat[0] + seat[2]) / 2
    return b[2] - seat[2] if safe_right else seat[0] - b[0]


def _outside_fallback(b: Box, layout: SceneLayout) -> bool:
    """P1 fallback when in_room flags are not given: a box lying entirely inside the door zone with its top in the
    upper 38% of the door is behind the glass (X-13/X-15: outside y1 <= 121, inside standing y1 >= 126)."""
    door, zone = layout.door, layout.door_zone
    if door is None or zone is None:
        return False
    return frac_inside(b, zone) >= 0.98 and b[1] < door[1] + 0.38 * (door[3] - door[1])


def attending_safe_series(persons_raw: list[list[Detection]], hands_per_frame: list[list[Detection]], cm,
                          layout: SceneLayout, in_room: list[list[bool]] | None = None) -> list[bool]:
    """P4 per frame: someone is at the safe. An upright in-room person box (top edge above the safe top by >= 0.1
    safe heights; a seated person's head is at or below the cabinet top in every camera) with x-overlap >= 0.25 of
    the safe width or an edge within 0.3 safe widths of the safe (X-20 0-4.5 s, X-14 34.5-43, X-21 1-6.5, X-23
    7.5-10.5, X-13 46.5-50.5), or a hand box (score >= 0.25) with centre inside expand(safe, 1.5, 1.2) (X-39), or -
    only when the safe stands beside the desk (x-disjoint from it, camera A) - the seated cashier whose box extends
    >= 0.2 safe widths beyond the seat reference towards the safe and ends within 0.3 widths of it (X-48 x2 218-246
    vs 229, X-35 x2 199-202 -> 254-268, X-19, X-39). When the safe stands behind the desk (camera B / A'), a seated
    person never attends (X-20 from 5.0 s: seated at the desk with the open safe behind her). Gaps of <= 2 frames
    are bridged. All False when layout.safe is None.
    `cm` provides cm.box[i] (cashier box or None) and optionally cm.seat_box (else layout.cashier_seat is the seat
    reference); `in_room[i][k]` flags are honoured when given, else a door-zone fallback is used."""
    n = len(persons_raw)
    safe = layout.safe
    if safe is None:
        return [False] * n
    sw, sh = _dims(safe)
    desk = layout.desk
    beside = desk is None or x_overlap(safe, desk) < 0.2 * sw or safe[3] > desk[1] + 0.3 * (desk[3] - desk[1])
    seat = getattr(cm, "seat_box", None) or layout.cashier_seat
    hand_gate = expand(safe, 1.5, 1.2)
    out: list[bool] = []
    for i in range(n):
        at = any(d.score >= HAND_MIN_SCORE and center_in(d.box, hand_gate) for d in hands_per_frame[i])
        for k, d in enumerate(persons_raw[i]):
            if at:
                break
            b = d.box
            if in_room is not None:
                if not in_room[i][k]:
                    continue
            elif _outside_fallback(b, layout):
                continue
            upright = b[1] < safe[1] - 0.1 * sh          # head above the safe top: standing (seated heads are at/below it)
            if upright:
                at = _near_safe(b, safe, sw)
            elif beside and seat is not None:
                at = _edge_toward(b, seat, safe) >= 0.2 * sw and _near_safe(b, safe, sw)
        out.append(at)
    return fill_gaps(out, 2)


# ------------------------------------------------------------------------------------------ P8 door_open
def _door_signals(dets_per_frame: list[list[Detection]], frames: FrameSet | None, layout: SceneLayout,
                  persons_raw: list[list[Detection]], with_pixels: bool) -> dict:
    """Per-frame door_open votes and the signals behind them (see door_open_series)."""
    n = len(dets_per_frame)
    safe = layout.safe
    sw, sh = _dims(safe)
    gate, max_area = expand(safe, 1.5, 1.4), 1.5 * area(safe)
    gated_boxes = [[d for d in dets if d.label == "safe_door_open" and center_in(d.box, gate) and d.area < max_area]
                   for dets in dets_per_frame]
    gated = [max((d.score for d in g), default=0.0) for g in gated_boxes]
    clusters = cluster_frames([(i, d.box, d.score) for i, g in enumerate(gated_boxes) for d in g], n, 0.5)
    b0c = clusters[0] if clusters and clusters[0].presence >= 0.3 else None
    b0 = b0c.box if b0c else safe
    b0_score = [max((d.score for d in g if iou(d.box, b0) >= 0.5), default=0.0) for g in gated_boxes]
    b0_base = float(np.median([s for s in b0_score if s > 0])) if b0c is not None and b0c.presence >= 0.5 else None
    ref_cx = (layout.desk[0] + layout.desk[2]) / 2 if layout.desk else layout.frame_w / 2
    room_left = ref_cx < (safe[0] + safe[2]) / 2          # the room/desk lies to the left of the safe

    def shift(b: Box) -> float:
        return (b0[0] - b[0]) if room_left else (b[2] - b0[2])

    persons = [[d.box for d in ps] for ps in persons_raw]
    covered = [any(inter_area(p, safe) / area(safe) >= 0.2 for p in persons[i]) for i in range(n)]
    p20 = float(np.percentile(gated, 20))
    v1 = [any(d.score >= 0.25 and frac_inside(d.box, b0) >= 0.3 and shift(d.box) >= 0.1 * sw for d in g)
          for g in gated_boxes]
    v2 = [gated[i] - p20 >= 0.10 or (b0_base is not None and not covered[i] and b0_base - b0_score[i] >= 0.15)
          for i in range(n)]
    diffs: list[tuple[float, float] | None] = [None] * n
    if with_pixels and frames is not None and len(frames.frames) == n:
        # baseline: closed-looking (lowest gated score) clean frames of the first 5 s; when a person covers the
        # safe there (X-23), the last 5 s (the safe is normally closed at the end), then the rest of the video
        k = min(n, max(2, int(round(5 * frames.sample_fps))))
        order = (sorted(range(k), key=lambda i: (gated[i], i)) + sorted(range(max(k, n - k), n), key=lambda i: (gated[i], -i))
                 + sorted(range(k, max(k, n - k)), key=lambda i: (gated[i], i)))
        rd = RoiDiff(frames, b0, persons, order)       # the ROI is where the detector sees the door (B0)
        diffs = [rd.diff(i) for i in range(n)]
    # the door state persists while a person hides the ROI: carry the last pixel verdict over <= 3 s of occlusion
    v3, last, age = [False] * n, False, 10 ** 6
    hold = int(round(3 * (frames.sample_fps if frames is not None else 2.0)))
    for i, d in enumerate(diffs):
        if d is not None:
            last, age = d[0] >= PIXEL_DIFF_THR and d[1] >= PIXEL_CHANGED_FRAC, 0
        else:
            age += 1
        v3[i] = last and age <= hold
    return {"gated": gated, "b0": b0, "b0_base": b0_base, "b0_score": b0_score, "p20": p20,
            "v1": v1, "v2": v2, "v3": v3, "diff": diffs}


def door_open_series(dets_per_frame: list[list[Detection]], frames: FrameSet, layout: SceneLayout,
                     persons_raw: list[list[Detection]], with_pixels: bool = True) -> tuple[list[bool], list[float]]:
    """P8 per frame: (door_open, gated safe_door_open score). door_open = >= 2 of 3 votes in >= 3 of 5 frames
    (centred window):
    1. a gated safe_door_open box (score >= 0.25, >= 30% of it over the baseline box B0) whose room-side edge is
       shifted >= 10% of the safe width towards the room/desk relative to B0 (X-39 (250,259,338,366) -> (235-246,289,
       291,365-376); X-13 (162,199,221,272) -> (145,217,183,271); X-48 (230,281,268,374) -> (221,301,266,381));
    2. gated score - p20(gated score over the video) >= 0.10 (X-5/48 0.33-0.42 -> 0.48-0.53, X-19 0.23-0.34 ->
       0.47-0.53, X-20) OR, when B0 is a stable cluster (presence >= 0.5) and no person covers the safe: its score
       dropped >= 0.15 below its median (X-39 0.45 -> 0.24-0.34) or it vanished while another gated box appeared;
    3. with_pixels: mean absolute grey difference of the door ROI (B0; person boxes masked, lights-off frames
       skipped, verdict carried over <= 3 s of occlusion) vs the per-pixel median of the most-closed-looking clean
       frames of the first 5 s (else the last 5 s) >= PIXEL_DIFF_THR with >= PIXEL_CHANGED_FRAC of the pixels changed
       (the swung door panel / interior: X-19 37.5-40, X-20, X-23, X-13).
    B0 = the most frequent gated safe_door_open cluster if seen in >= 30% of frames, else the anchor itself.
    Gated score = max safe_door_open score of boxes whose centre is inside expand(safe,1.5,1.4) and area < 1.5 x safe
    area; 0.0 when layout.safe is None (then door_open is all False)."""
    n = len(dets_per_frame)
    if layout.safe is None or n == 0:
        return [False] * n, [0.0] * n
    s = _door_signals(dets_per_frame, frames, layout, persons_raw, with_pixels)
    votes = [int(s["v1"][i]) + int(s["v2"][i]) + int(s["v3"][i]) >= 2 for i in range(n)]
    door = [sum(votes[max(0, i - 2):i + 3]) >= 3 for i in range(n)]
    return door, s["gated"]
