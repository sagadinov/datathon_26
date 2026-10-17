"""Per-frame scene state ("timeline") v2: people, objects, screen, safe -> FrameState list.

Orchestrates the modules people.py / objects.py / screen.py / safe.py (specified in
docs/violations_observed.md, sections 3 and 6). Rules (rules.py) read only this timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline import objects as objmod
from src.pipeline.objects import ObjState  # noqa: F401  (re-exported for rules/evidence)
from src.pipeline import people, safe, screen
from src.pipeline.detect import Detection
from src.pipeline.frames import FrameSet
from src.pipeline.track import Track, track_boxes
from src.pipeline.zones import SceneLayout, build_layout, inter_area, area

Box = tuple[float, float, float, float]


@dataclass
class PersonState:
    track_id: int
    box: Box
    score: float
    in_room: bool = False          # P1: inside the cash room (not behind the glass / door)
    is_cashier: bool = False       # this box is the cashier's (video-level identity, P3)
    at_desk: bool = False          # desk_gap <= 0.10 (touching the desk)
    far_side: bool = False         # on the client side of the desk (P9)
    at_door: bool = False          # overlapping the door zone (crossing / standing in the doorway)
    desk_gap: float = 9.0
    seat_iou: float = 0.0

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class FrameState:
    index: int
    t: float
    persons: list[PersonState] = field(default_factory=list)
    objects: list[ObjState] = field(default_factory=list)
    # people / cashier (P1-P3, P9)
    cashier_present: bool = False  # an in-room box matches the cashier (seat IoU or identity)
    seat_vacated: bool = False     # cashier standing / not sitting at the seat
    cashier_box: Box | None = None
    someone_at_desk: bool = False  # any in-room person touching the desk
    n_in_room: int = 0
    n_clients_at_desk: int = 0     # in-room non-cashiers touching the desk
    # safe (P4, P8)
    attending_safe: bool = False
    door_open: bool = False
    safe_score: float = 0.0
    # screen / light (P6, P7)
    screen: str = "unknown"        # active | dim | off | unknown
    lid_closed: bool = False
    brightness: float = 0.0
    lights_off: bool = False
    # hands (P10)
    hands_visible: int = 0         # hand boxes inside the cashier box
    hand_at_counter: bool = False
    cash_in_hand: bool = False

    @property
    def cashier(self) -> PersonState | None:
        return next((p for p in self.persons if p.is_cashier), None)

    def event_objects(self, label: str, min_score: float = 0.0) -> list[ObjState]:
        return [o for o in self.objects if o.label == label and o.on_desk and o.is_event_object and o.score >= min_score]


@dataclass
class Timeline:
    layout: SceneLayout
    frames: list[FrameState]
    tracks: dict[int, Track]
    sample_fps: float
    cashier_track_id: int | None = None
    exit_t: float | None = None                 # P3: cashier left the room (None if never)
    entered: list[tuple[float, int]] = field(default_factory=list)   # (t, track_id) non-cashier entered
    left: list[tuple[float, int]] = field(default_factory=list)      # (t, track_id) non-cashier left
    lights_off_events: list[float] = field(default_factory=list)
    lock_events: list[float] = field(default_factory=list)
    room_empty_start: bool = False              # nobody in the room during the first seconds
    cash_roi: list | None = None                # roi.py second look: valuables on the desk per frame
    seat_box: tuple | None = None               # median seated cashier box (people.CashierModel)

    def __len__(self) -> int:
        return len(self.frames)

    def sec(self, s: float) -> int:
        return max(1, int(round(s * self.sample_fps)))


def build_timeline(frames: FrameSet, dets_per_frame: list[list[Detection]], with_pixels: bool = True) -> Timeline:
    w, h = frames.meta.width, frames.meta.height
    if frames.frames:
        import cv2

        img = cv2.imread(str(frames.frames[0].path))
        if img is not None:
            h, w = img.shape[:2]
    n = len(frames)
    # 1-2. people: dedup per frame, then tracking on the dedup'd boxes
    persons_raw = [people.dedup_persons([d for d in dets if d.label == "person"], h) for dets in dets_per_frame]
    ids_per_frame, tracks = track_boxes(persons_raw, w, h)
    # 3. layout (+ safe anchor sanity, P8)
    layout = build_layout(dets_per_frame, w, h, tracks)
    layout.safe = safe.pick_safe_anchor(dets_per_frame, layout)
    # 4-5. in_room (P1) and the cashier model (P3); ref height comes from the cashier
    in_room0 = [[people.in_room(d.box, layout, None) for d in ps] for ps in persons_raw]
    cm = people.CashierModel(persons_raw, ids_per_frame, tracks, layout, in_room0, frames.sample_fps)
    in_room = [[people.in_room(d.box, layout, cm.ref_height) for d in ps] for ps in persons_raw]
    cm.finalize(in_room)
    # 6. roles / enter-leave events (P9)
    roles = people.roles_and_events(persons_raw, ids_per_frame, in_room, cm, layout, frames.sample_fps)
    # 7. hands
    hands_per_frame = [[d for d in dets if d.label == "hand" and d.score >= 0.20] for dets in dets_per_frame]
    # 8. objects (P5)
    om = objmod.ObjectModel(dets_per_frame, persons_raw, cm, layout, frames)
    # 9. screen / lights (P6, P7)
    scr = screen.screen_states(frames, layout, persons_raw) if with_pixels else None
    # 10. safe (P4, P8)
    attending = safe.attending_safe_series(persons_raw, hands_per_frame, cm, layout)
    door_open, safe_scores = safe.door_open_series(dets_per_frame, frames, layout, persons_raw, with_pixels)

    states: list[FrameState] = []
    for i in range(n):
        st = FrameState(i, frames.frames[i].timestamp_sec)
        for k, d in enumerate(persons_raw[i]):
            p = PersonState(ids_per_frame[i][k], d.box, d.score, in_room=in_room[i][k])
            p.is_cashier = cm.is_cashier(i, k)
            p.desk_gap, p.at_desk, p.far_side, p.at_door, p.seat_iou = roles.person_flags(i, k)
            st.persons.append(p)
        st.cashier_present, st.seat_vacated, st.cashier_box = cm.present[i], cm.vacated[i], cm.box[i]
        st.n_in_room = sum(p.in_room for p in st.persons)
        st.someone_at_desk = any(p.in_room and p.at_desk for p in st.persons)
        st.n_clients_at_desk = sum(p.in_room and p.at_desk and not p.is_cashier for p in st.persons)
        st.objects = om.objects[i]
        st.cash_in_hand = om.cash_in_hand[i]
        st.hands_visible = sum(1 for d in hands_per_frame[i] if cm.box[i] and _center_in(d.box, cm.box[i]))
        st.hand_at_counter = om.hand_at_counter[i]
        st.attending_safe, st.door_open, st.safe_score = attending[i], door_open[i], safe_scores[i]
        if scr is not None:
            s = scr[i]
            st.screen, st.lid_closed, st.brightness, st.lights_off = s.screen, s.lid_closed, s.brightness, s.lights_off
        states.append(st)
    tl = Timeline(layout, states, tracks, frames.sample_fps, cashier_track_id=cm.track_id, exit_t=cm.exit_t,
                  entered=roles.entered, left=roles.left, room_empty_start=cm.room_empty_start)
    tl.seat_box = cm.seat_box
    if scr is not None:
        tl.lights_off_events = [i / frames.sample_fps for i in screen.lights_off_events(scr)]
        tl.lock_events = [i / frames.sample_fps for i in screen.lock_events(scr, cm)]
    return tl


def _center_in(b: Box, zone: Box) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


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


def fill_gaps(values: list[bool], max_gap: int) -> list[bool]:
    """Bridge False gaps of length <= max_gap between True runs (tolerate missed detections)."""
    out = list(values)
    rs = runs(values)
    for (a1, b1), (a2, b2) in zip(rs, rs[1:]):
        if a2 - b1 <= max_gap:
            for k in range(b1, a2):
                out[k] = True
    return out
