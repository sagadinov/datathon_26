"""Object primitives (spec: docs/violations_observed.md P5, P10 partly).

Background (baseline) objects, anchor masks, worn clothing, on_desk by the bottom edge, new static desk
objects, cash in the cashier's hands, hand at the counting machine.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.detect import Detection
from src.pipeline.frames import FrameSet
from src.pipeline.objects_util import (Cluster, box_dist, center, cluster_tracks, contain_frac, fill_gaps,
                                       median_box, point_in, running_median)
from src.pipeline.track import iou
from src.pipeline.zones import SceneLayout, area, expand, inter_area

Box = tuple[float, float, float, float]
MOVABLE = ("cash", "money_bag", "bag", "cup", "book", "jacket", "phone", "document")
STATIC = ("counter_machine", "laptop")   # clustered only to anchor the baseline and the masks, never ObjStates
WORN = ("jacket", "bag")                 # classes that are 'worn clothing' when inside a person box
HAND_LABELS = ("cash", "phone", "cup")   # cash in hands is detected as any of these (phone/cup under occlusion)


@dataclass
class ObjState:
    label: str
    score: float
    box: Box
    on_desk: bool                  # bottom edge inside the (slightly expanded) desk box
    cluster_id: int = -1           # stable id of the spatial cluster this box belongs to
    background: bool = False       # baseline furniture/equipment - never starts an event
    masked: bool = False           # inside counter/laptop/phone masks or worn clothing
    static: bool = False           # IoU >= 0.6 with this cluster's box in the previous frame
    age: int = 0                   # consecutive frames the cluster has been seen (1 = new)
    in_cashier: bool = False       # centre inside the cashier's box

    @property
    def is_event_object(self) -> bool:
        return not self.background and not self.masked


class ObjectModel:
    """P5: per-frame movable objects with background / mask / static flags, plus hand signals.

    Baseline (background): per class, greedy IoU clusters (IoU >= 0.5 with the cluster's median box, see
    objects_util.cluster_tracks) over the whole video, for MOVABLE and STATIC (counter_machine, laptop) classes.
    Clusters of any class at the same place (IoU >= 0.5 of medians) form one 'place' whose presence is the union of
    their sightings: the detector flips classes on furniture (X-25 card terminal = counter_machine while the cashier
    sits, money_bag after she stands up; money_bag <-> bag <-> phone on the desk phone). A place is background when
    present in >= 80% of frames OR (seen in >= 1/3 of the first 3 s AND still seen in the last third of the
    clip) - furniture is there at both ends, a real object brought in at the start and taken away later (X-25 cup +
    magazine) is not, and a bag carried in the hands for 6 s and then put down (X-15) forms a separate cluster from
    its place on the desk. This kills money_bag on the counting machine (all 42 videos, 0.30-0.58), money_bag/bag on
    the desk phone (X-43/45), cash on the shelf behind the glass (camera A, [27,140,53,154]), cup on labels/sockets,
    bag/cup/money_bag on the foreground chair.
    Masks: an object with IoU >= 0.3 with the counter_machine box expanded (1.2 x 1.5: hopper + output tray) or the
    laptop box, or >= 60% of its area inside one of them, or inside a background phone / counter_machine / laptop
    cluster (a second device on the desk), is masked (X-12 money_bag 100% inside the counter; X-31 cup = machine
    hopper; X-43 banknotes in the output tray; X-25 card terminal).
    jacket/bag with >= 70% of area inside a person box (or IoU > 0.3) are worn clothing -> masked (not money_bag:
    in camera A the pouch on the desk lies inside the cashier's 2D box, X-17).
    on_desk: bottom edge (x-centre, y2) inside the desk box widened by 2.5% per side, extended 15% of its height
    beyond the FAR edge (top: X-40 bag on the far corner; X-17 phone on the far edge) but only 2% beyond the near
    edge (bottom): a box whose bottom edge is below the near edge stands in front of the desk - on the cashier's
    body / chair (X-38, X-16 money_bag blobs on the back of the seated cashier, camera A).
    static: IoU >= 0.6 with the same cluster's box in the previous frame (<= 2 missed frames bridged);
    age = consecutive frames seen (gaps <= 2 frames do not reset it).

    Attributes (all length n):
      objects[i]        - list[ObjState] for MOVABLE classes with score >= 0.2
      cash_in_hand[i]   - a {cash, phone(small: area <= 4% of the cashier box), cup} event object with centre inside
                          the cashier box and above desk.y0 + 0.6*desk_h (X-1 [134,316,163,342] inside the cashier
                          box), score >= 0.20 (cash in hands scores 0.22-0.33, X-13 stacks at the hip 0.23-0.48 as
                          'phone'/'cup'); gaps <= 2 frames bridged because it flickers
      hand_at_counter[i]- a hand box (score >= 0.20) within 0.25 x counter size of counter_machine in >= 4 of the
                          last 10 frames (2 of 5 s), OR the cashier's anchor overlap rises to >= 0.4 and >= 0.08 above
                          its running median (X-34 12.5-15.5, X-38 10-11, 41-43) - proxy for 'counting'
    Methods:
      new_desk_object(labels, thr, k) -> list[bool]: per frame, a non-background, non-masked, on_desk object of
          one of `labels` with score >= thr, static, age >= k (X-17 money_bag 7.0 0.50 then >= 0.30; X-26 cash
          0.30-0.32; X-38 0.31-0.43; X-46 cup 0.43/0.46 then >= 0.36). Once confirmed the run is extended back to
          the cluster's first on-desk sighting (the placement) and forward while the object stays with score
          >= max(0.2, thr - 0.1) (enter/hold hysteresis), holding over <= 2 missed frames. Runs of the same label
          that touch in time (gap <= 2 frames) and in space (boxes within a quarter of their size) are one object
          that was nudged (X-17 pouch pushed 7.0-9.5) and are merged.
      object_runs(labels, thr, k) -> list[tuple[int, int, Box]]: (start, end_exclusive, median box) of such objects.
    """

    def __init__(self, dets_per_frame: list[list[Detection]], persons_raw: list[list[Detection]], cm,
                 layout: SceneLayout, frames: FrameSet):
        self.n = n = len(dets_per_frame)
        self.fps = float(frames.sample_fps or 2.0)
        self.layout = layout
        cbox = list(getattr(cm, "box", []) or []) if cm is not None else []
        self.cashier_box: list[Box | None] = [cbox[i] if i < len(cbox) else None for i in range(n)]
        self.clusters, assign = cluster_tracks(dets_per_frame, MOVABLE + STATIC, 0.2)
        self._static_and_age()
        self._baseline()
        self.mask_boxes = self._mask_boxes()
        desk = layout.desk
        if desk:
            dw, dh = desk[2] - desk[0], desk[3] - desk[1]
            self.desk_zone: Box | None = (desk[0] - 0.025 * dw, desk[1] - 0.15 * dh, desk[2] + 0.025 * dw, desk[3] + 0.02 * dh)
        else:
            self.desk_zone = None
        self.objects: list[list[ObjState]] = []
        for i in range(n):
            row: list[ObjState] = []
            cb = self.cashier_box[i]
            for d, cid in assign[i]:
                if d.label in STATIC:
                    continue
                c = self.clusters[cid]
                cx, cy = center(d.box)
                st = ObjState(d.label, d.score, d.box, self._on_desk(d.box), cid, c.background,
                              self._masked(d.box, d.label, persons_raw[i]), c.static.get(i, False), c.age.get(i, 1),
                              bool(cb and point_in(cx, cy, cb)))
                if c.rec[i][0] == d.box:          # the cluster's best box of this frame carries its flags
                    c.on_desk[i], c.masked[i] = st.on_desk, st.masked
                row.append(st)
            self.objects.append(row)
        self._runs_cache: dict = {}
        self.cash_in_hand = self._cash_in_hand()
        self.hand_at_counter = self._hand_at_counter(dets_per_frame)

    # ---- construction helpers --------------------------------------------------------------
    def _static_and_age(self) -> None:
        for c in self.clusters.values():
            prev = None
            for f in c.frames:
                if prev is not None and f - prev <= 3:
                    c.age[f] = c.age[prev] + 1
                    c.static[f] = iou(c.rec[f][0], c.rec[prev][0]) >= 0.6
                else:
                    c.age[f], c.static[f] = 1, False
                prev = f

    def _baseline(self) -> None:
        n = self.n
        head = max(2, int(round(3 * self.fps)))
        tail_start = n - max(head, n // 3)
        places: list[tuple[Box, set[int], list[Cluster]]] = []     # (median of the largest cluster, frames, clusters)
        for c in sorted(self.clusters.values(), key=lambda c: -c.n):
            m = c.median()
            for p in places:
                if iou(m, p[0]) >= 0.5:
                    p[1].update(c.frames)
                    p[2].append(c)
                    break
            else:
                places.append((m, set(c.frames), [c]))
        for _, fr, cs in places:
            frac = len(fr) / max(1, n)
            head_frac = sum(1 for f in fr if f < head) / head
            seen_tail = any(f >= tail_start for f in fr)
            bg = frac >= 0.8 or (head_frac >= 1 / 3 and seen_tail)
            for c in cs:
                c.background = bg

    def _mask_boxes(self) -> list[Box]:
        lay = self.layout
        masks: list[Box] = []
        if lay.counter_machine:
            masks.append(expand(lay.counter_machine, 1.2, 1.5))
        if lay.laptop:
            masks.append(lay.laptop)
        for c in self.clusters.values():
            if c.background and c.n >= 3 and c.label in ("phone", "laptop"):
                masks.append(c.median())
            elif c.background and c.n >= 3 and c.label == "counter_machine":
                masks.append(expand(c.median(), 1.2, 1.5))
        return masks

    def _masked(self, box: Box, label: str, persons: list[Detection]) -> bool:
        for m in self.mask_boxes:
            if iou(box, m) >= 0.3 or contain_frac(box, m) >= 0.6:
                return True
        if label in WORN:
            for p in persons:
                if contain_frac(box, p.box) >= 0.7 or iou(box, p.box) > 0.3:
                    return True
        return False

    def _on_desk(self, box: Box) -> bool:
        if self.desk_zone is None:
            return False
        return point_in((box[0] + box[2]) / 2, box[3], self.desk_zone)

    # ---- hand signals (P10) ----------------------------------------------------------------
    def _cash_in_hand(self) -> list[bool]:
        desk = self.layout.desk
        ylim = desk[1] + 0.6 * (desk[3] - desk[1]) if desk else float("inf")
        raw = []
        for i in range(self.n):
            cb = self.cashier_box[i]
            hit = False
            if cb:
                cb_area = area(cb)
                for o in self.objects[i]:
                    if o.label not in HAND_LABELS or o.score < 0.20 or not o.is_event_object:
                        continue
                    if o.label == "phone" and area(o.box) > 0.04 * cb_area:
                        continue
                    cx, cy = center(o.box)
                    if point_in(cx, cy, cb) and cy < ylim:
                        hit = True
                        break
            raw.append(hit)
        return fill_gaps(raw, 2)

    def _hand_at_counter(self, dets_per_frame: list[list[Detection]]) -> list[bool]:
        n, lay = self.n, self.layout
        cmach, anchor = lay.counter_machine, lay.anchor
        ok = bool(cmach and anchor and area(anchor) > 0)
        self.anchor_overlap = [inter_area(cb, anchor) / area(anchor) if cb and ok else 0.0 for cb in self.cashier_box]
        if not ok:
            return [False] * n
        tol = 0.25 * min(cmach[2] - cmach[0], cmach[3] - cmach[1])
        near = [any(d.label == "hand" and d.score >= 0.20 and box_dist(d.box, cmach) <= tol for d in dets)
                for dets in dets_per_frame]
        win, need = max(1, int(round(5 * self.fps))), max(1, int(round(2 * self.fps)))
        votes = [sum(near[max(0, i - win + 1): i + 1]) >= need for i in range(n)]
        ao = self.anchor_overlap
        base = running_median(ao, [cb is not None for cb in self.cashier_box], int(round(10 * self.fps)))
        rise = [cb is not None and a >= 0.4 and a >= b + 0.08 for cb, a, b in zip(self.cashier_box, ao, base)]
        return [v or r for v, r in zip(votes, rise)]

    # ---- new static desk objects (P5) --------------------------------------------------------
    def _eligible(self, c: Cluster, f: int) -> bool:
        return c.on_desk.get(f, False) and not c.masked.get(f, False)

    def _cluster_runs(self, labels, thr: float, k: int) -> list[tuple[int, int, list[int]]]:
        """[(start, end_exclusive, cluster ids)] of confirmed new desk objects, merged and sorted by start."""
        key = (frozenset(labels), float(thr), int(k))
        if key in self._runs_cache:
            return self._runs_cache[key]
        hold = max(0.2, thr - 0.1)
        out: list[tuple[int, int, int]] = []
        for c in self.clusters.values():
            if c.label not in labels or c.background:
                continue
            for seg in c.segments(2):
                pos = 0
                while pos < len(seg):
                    fi = next((j for j in range(pos, len(seg)) if self._eligible(c, seg[j]) and c.score(seg[j]) >= thr
                               and c.static[seg[j]] and c.age[seg[j]] >= k), None)
                    if fi is None:
                        break
                    j = fi                                   # back to the first on-desk sighting of this run
                    while j > pos and self._eligible(c, seg[j - 1]):
                        j -= 1
                    start, last_good = seg[j], seg[fi]
                    for f in range(seg[fi] + 1, seg[-1] + 1):   # forward: hold while the object stays
                        if self._eligible(c, f) and c.score(f) >= hold:
                            last_good = f
                        elif f - last_good > 2:
                            break
                    out.append((start, last_good + 1, c.cid))
                    pos = next((q for q in range(fi, len(seg)) if seg[q] > last_good), len(seg))
        merged = self._merge_runs(sorted(out))
        self._runs_cache[key] = merged
        return merged

    def _merge_runs(self, rs: list[tuple[int, int, int]]) -> list[tuple[int, int, list[int]]]:
        """Merge same-label runs that touch in time (gap <= 2 frames) and in space (one nudged object)."""
        merged: list[tuple[int, int, list[int]]] = []
        for a, b, cid in rs:
            c = self.clusters[cid]
            for j, (ma, mb, cids) in enumerate(merged):
                m = self.clusters[cids[-1]]
                if m.label != c.label or a > mb + 2:
                    continue
                ba, bb = c.median(), m.median()
                size = max(ba[2] - ba[0], ba[3] - ba[1], bb[2] - bb[0], bb[3] - bb[1])
                if box_dist(ba, bb) <= 0.25 * size:
                    merged[j] = (min(ma, a), max(mb, b), cids + [cid])
                    break
            else:
                merged.append((a, b, [cid]))
        return sorted(merged)

    def new_desk_object(self, labels: set[str], thr: float, k: int) -> list[bool]:
        out = [False] * self.n
        for a, b, _ in self._cluster_runs(labels, thr, k):
            for i in range(a, min(b, self.n)):
                out[i] = True
        return out

    def object_runs(self, labels: set[str], thr: float, k: int) -> list[tuple[int, int, Box]]:
        res = []
        for a, b, cids in self._cluster_runs(labels, thr, k):
            boxes = [self.clusters[cid].rec[f][0] for cid in cids for f in range(a, b) if f in self.clusters[cid].rec]
            res.append((a, b, median_box(boxes) if boxes else self.clusters[cids[0]].median()))
        return res
