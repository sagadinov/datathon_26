"""Object primitives (spec: docs/violations_observed.md P5, P10 partly).

Background (baseline) objects, anchor masks, worn clothing, on_desk by the bottom edge, new static desk
objects, cash in the cashier's hands, hand at the counting machine.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.detect import Detection
from src.pipeline.frames import FrameSet
from src.pipeline.zones import SceneLayout, area, inter_area

Box = tuple[float, float, float, float]
MOVABLE = ("cash", "money_bag", "bag", "cup", "book", "jacket", "phone", "document")


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

    Baseline (background): per class, greedy IoU clusters (IoU >= 0.5 with the running median box) over the
    whole video; a cluster is background when present in >= 70% of frames OR in >= 60% of the first 3 s.
    This kills money_bag on the counting machine (all 42 videos, 0.30-0.58), money_bag/bag on the desk
    phone (X-43/45), cash on the shelf behind the glass (camera A, [27,140,53,154]), cup on labels/sockets,
    bag/cup/money_bag on the foreground chair.
    Masks: an object with IoU >= 0.3 with counter_machine or laptop boxes, or with a phone cluster, is masked
    (X-12 money_bag 100% inside the counter; X-31 cup = machine hopper). jacket/bag with >= 70% of area inside a
    person box (or IoU > 0.3) are worn clothing -> masked.
    on_desk: bottom edge (x-centre, y2) inside expand(desk, 1.05, 1.15) (X-40 bag on the far corner; X-17 phone
    on the far edge).
    static: IoU >= 0.6 with the same cluster's box in the previous frame; age = consecutive frames seen.

    Attributes (all length n):
      objects[i]        - list[ObjState] for MOVABLE classes with score >= 0.2
      cash_in_hand[i]   - a {cash, phone(small), cup} box with centre inside the cashier box and above
                          desk.y0 + 0.6*desk_h (X-1 [134,316,163,342] inside the cashier box), score >= 0.25
      hand_at_counter[i]- a hand box (score >= 0.20) within 15 px of counter_machine in >= 4 of the last 10 frames,
                          OR cashier anchor_overlap rises >= 0.4 (X-34 12.5-15.5, X-38 10-11) - proxy for 'counting'
    Methods:
      new_desk_object(labels, thr, k) -> list[bool]: per frame, a non-background, non-masked, on_desk object of
          one of `labels` with score >= thr, static, age >= k (X-17 money_bag 7.0 0.50 then >= 0.30; X-26 cash
          0.30-0.32; X-38 0.31-0.43; X-46 cup 0.43/0.46 then >= 0.36). Hold True over <= 2 missed frames.
      object_runs(labels, thr, k) -> list[tuple[int, int, Box]]: (start, end_exclusive, median box) of such objects.
    """

    def __init__(self, dets_per_frame: list[list[Detection]], persons_raw: list[list[Detection]], cm,
                 layout: SceneLayout, frames: FrameSet):
        raise NotImplementedError

    def new_desk_object(self, labels: set[str], thr: float, k: int) -> list[bool]:
        raise NotImplementedError

    def object_runs(self, labels: set[str], thr: float, k: int) -> list[tuple[int, int, Box]]:
        raise NotImplementedError
