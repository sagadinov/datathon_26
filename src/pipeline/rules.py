"""Rule engine v2: scene timeline -> temporal events (spec docs/violations_observed.md, section 4).

Every rule is a pure function over per-frame series derived once (Series). Durations are in seconds
(converted with the sample rate), object presence comes from the objects module flags (background /
masked / static / age), people from the cashier model (present / vacated / exit) and roles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.state import FrameState, Timeline, fill_gaps, runs
from src.pipeline.zones import area, inter_area

CFG = {
    "cash_thr": 0.30, "valuables_k": 2, "cup_enter": 0.40, "cup_hold": 0.30, "book_thr": 0.40,
    "items_thr": 0.35, "items_k": 3, "away_gap": 0.10,
    "T1A_min_s": 3.5, "T1B_static_s": 4.0, "T1B_noclient_s": 2.0,
    "T2_min_s": 3.0, "T2_absent_s": 8.0, "T3_min_s": 2.0, "T4_min_s": 2.0, "T4b_min_s": 1.5,
    "T5_window_s": 20.0, "T6_min_s": 1.5, "T6_ctx_s": 5.0, "T7_gone_s": 1.5, "T7_hidden_s": 1.5,
    "T8_client_s": 3.0, "T8_window_s": 10.0, "T8_count_s": 2.0, "T9_leave_s": 10.0, "T9_stay_s": 2.0,
    "T10_dwell_s": 1.0, "T10_reach_s": 10.0,
}


@dataclass
class Event:
    type_id: str
    t_start: float
    t_end: float
    score: float
    frame_idx: int                      # main evidence frame
    details: dict = field(default_factory=dict)
    evidence_idx: list[int] = field(default_factory=list)   # optional explicit evidence frames


def _box_iou(a, b) -> float:
    inter = inter_area(a, b)
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


class Series:
    """All per-frame boolean/score series the rules need, computed once from the timeline."""

    def __init__(self, tl: Timeline, cfg: dict = CFG):
        self.tl, self.cfg, F = tl, cfg, tl.frames
        self.n, self.t, self.sec = len(F), [s.t for s in F], tl.sec
        desk_w = (tl.layout.desk[2] - tl.layout.desk[0]) if tl.layout.desk else 100.0
        # --- objects -----------------------------------------------------------------
        self.valuables = self.new_obj({"cash", "money_bag", "bag", "book"}, cfg["cash_thr"], cfg["valuables_k"])
        self.cash_desk = self.new_obj({"cash"}, cfg["cash_thr"], cfg["valuables_k"])
        self.bag_desk = self.new_obj({"money_bag", "bag"}, cfg["cash_thr"], cfg["items_k"])
        self.items = self.new_obj({"bag", "money_bag", "jacket", "book"}, cfg["items_thr"], cfg["items_k"])
        self.cash_ctx = [any(o.label == "cash" and o.is_event_object and o.on_desk and o.score >= 0.40 for o in s.objects)
                         or s.cash_in_hand for s in F]
        cash_boxes = [[o.box for o in s.objects if o.label == "cash" and o.score >= 0.3] for s in F]

        def cup_ok(o, s):
            w, h = o.box[2] - o.box[0], o.box[3] - o.box[1]
            if w < 0.07 * desk_w or h < 0.8 * w or not o.is_event_object:
                return False
            if any(_box_iou(o.box, cb) >= 0.5 for cb in cash_boxes[s.index]):
                return False
            return o.on_desk or o.in_cashier
        cup_hi = [any(o.label == "cup" and o.score >= cfg["cup_enter"] and cup_ok(o, s) for o in s.objects) for s in F]
        cup_lo = [any(o.label == "cup" and o.score >= cfg["cup_hold"] and cup_ok(o, s) for o in s.objects) for s in F]
        self.cup = self._hysteresis(cup_hi, cup_lo)
        max_cash_area = max((area(b) for cb in cash_boxes for b in cb), default=0.0)
        self.book = [any(o.label == "book" and o.score >= cfg["book_thr"] and o.on_desk and o.is_event_object and o.static
                         and area(o.box) >= 2 * max_cash_area
                         and not any(_box_iou(o.box, cb) >= 0.5 for cb in cash_boxes[s.index]) for o in s.objects) for s in F]
        # --- people ----------------------------------------------------------------------
        self.present = [s.cashier_present for s in F]
        self.vacated = [s.seat_vacated for s in F]
        self.cashier_gap = [next((p.desk_gap for p in s.persons if p.is_cashier and p.in_room), 9.0) for s in F]
        self.attending = [s.attending_safe for s in F]
        self.counting = [s.hand_at_counter for s in F]
        self.away = fill_gaps([((not pr) or (va and gap > cfg["away_gap"])) and not at and not cnt
                               for pr, va, gap, at, cnt in zip(self.present, self.vacated, self.cashier_gap,
                                                              self.attending, self.counting)], 1)
        self.others_at_desk = [s.n_clients_at_desk for s in F]
        self.visitors = [sum(1 for p in s.persons if p.in_room and not p.is_cashier) for s in F]
        self.visitor_near = [sum(1 for p in s.persons if p.in_room and not p.is_cashier and p.desk_gap <= 0.3) for s in F]
        self.intruder = [any(p.in_room and not p.is_cashier and p.seat_iou >= 0.15 for p in s.persons) for s in F]
        self.n_in_room = [s.n_in_room for s in F]
        self.hands_hidden = [s.hands_visible == 0 for s in F]
        self.cash_in_hand = [s.cash_in_hand for s in F]
        # --- screen / safe ---------------------------------------------------------------
        self.screen_lit = [s.screen in ("active", "dim") and not s.lid_closed for s in F]
        self.screen_active = [s.screen == "active" for s in F]
        self.lights_off = [s.lights_off for s in F]
        self.door_open = fill_gaps([s.door_open for s in F], 1)
        self.exit_i = self._t_to_i(tl.exit_t) if tl.exit_t is not None else None

    # helpers -----------------------------------------------------------------------------
    def new_obj(self, labels: set[str], thr: float, k: int) -> list[bool]:
        raw = [any(o.label in labels and o.on_desk and o.is_event_object and o.score >= thr and o.age >= k
                   for o in s.objects) for s in self.tl.frames]
        return fill_gaps(raw, 2)

    @staticmethod
    def _hysteresis(hi: list[bool], lo: list[bool]) -> list[bool]:
        out = [False] * len(lo)
        for a, b in runs(fill_gaps(lo, 1)):
            if any(hi[a:b]):
                for k in range(a, b):
                    out[k] = True
        return out

    def _t_to_i(self, t: float) -> int:
        return int(np.argmin([abs(x - t) for x in self.t]))

    def obj_boxes(self, i: int, labels: set[str], thr: float):
        return [o for o in self.tl.frames[i].objects if o.label in labels and o.on_desk and o.is_event_object and o.score >= thr]


def _event(S: Series, type_id: str, a: int, b: int, score: float, details: dict | None = None,
           evidence: list[int] | None = None) -> Event:
    mid = min(b - 1, a + S.sec(1.0))
    return Event(type_id, S.t[a], S.t[b - 1], score, mid, details or {}, evidence or [])


def _runs_min(mask: list[bool], S: Series, min_s: float, or_until_end: bool = False, end_min: int = 2):
    """Runs of True at least min_s long; with or_until_end also runs that reach the exit / the end."""
    out = []
    for a, b in runs(mask):
        reach = or_until_end and (b >= S.n or (S.exit_i is not None and b >= S.exit_i)) and b - a >= end_min
        if b - a >= S.sec(min_s) or reach:
            out.append((a, b))
    return out


def rule_t1(S: Series) -> list[Event]:
    """T1-A valuables unattended (cashier away); T1-B next visitor enters while valuables lie on the desk."""
    out = []
    mask = [v and aw and not at and oth == 0 for v, aw, at, oth in zip(S.valuables, S.away, S.attending, S.others_at_desk)]
    for a, b in _runs_min(mask, S, S.cfg["T1A_min_s"], or_until_end=True):
        obj = S.obj_boxes(a, {"cash", "money_bag", "bag", "book"}, S.cfg["cash_thr"])
        label = max(obj, key=lambda o: o.score).label if obj else "cash"
        out.append(_event(S, "T1", a, b, min(0.95, 0.55 + 0.4 * (b - a) / max(1, S.n)),
                          {"variant": "A", "object": label, "away_s": round((b - a) / S.tl.sample_fps, 1)},
                          [max(0, a - S.sec(1.0)), a, b - 1]))
    k_static, k_noclient = S.sec(S.cfg["T1B_static_s"]), S.sec(S.cfg["T1B_noclient_s"])
    for t_in, tid in S.tl.entered:
        i = S._t_to_i(t_in)
        if i < k_static or sum(S.valuables[i - k_static:i]) < 0.75 * k_static:
            continue
        if any(S.others_at_desk[max(0, i - k_noclient):i]):
            continue
        j = i
        while j < S.n and S.valuables[j]:
            j += 1
        if j - i < 2:
            continue
        obj = S.obj_boxes(i - 1, {"cash", "money_bag", "bag", "book"}, S.cfg["cash_thr"])
        label = max(obj, key=lambda o: o.score).label if obj else "cash"
        out.append(_event(S, "T1", i, max(i + 2, j), 0.75, {"variant": "B", "object": label, "entered_t": round(t_in, 1)},
                          [max(0, i - k_static), i, min(S.n - 1, i + S.sec(2.0))]))
    return out


def rule_t2(S: Series) -> list[Event]:
    """Cashier leaves (final departure / exit) with the screen lit, or with items left on the desk."""
    out = []
    mask = [aw and not at for aw, at in zip(S.away, S.attending)]
    for a, b in _runs_min(mask, S, S.cfg["T2_min_s"], or_until_end=True):
        exited = S.exit_i is not None and a <= S.exit_i + 2
        absent_long = (b - a) >= S.sec(S.cfg["T2_absent_s"])
        if not (exited or absent_long or b >= S.n):
            continue
        win_a = max(a, (S.exit_i - S.sec(2.0)) if S.exit_i is not None else a)
        win = range(win_a, S.n)
        lit = sum(S.screen_lit[i] for i in win) >= 0.5 * len(win) if len(win) else False
        dark_on = any(S.lights_off[i] and S.screen_active[i] for i in win)
        items = sum(S.cup[i] or S.book[i] for i in win) >= 0.8 * len(win) if len(win) else False
        if not (lit or dark_on or items):
            continue
        why = "screen" if (lit or dark_on) else "items"
        out.append(_event(S, "T2", a, b, 0.8 if why == "screen" else 0.6,
                          {"why": why, "lights_off": bool(any(S.lights_off[i] for i in win)),
                           "exit_t": round(S.tl.exit_t, 1) if S.tl.exit_t is not None else None},
                          [a, S.exit_i if S.exit_i is not None else b - 1, S.n - 1]))
        break   # one departure event is enough
    return out


def rule_t3(S: Series) -> list[Event]:
    """Safe door open while nobody attends the safe."""
    if S.tl.layout.safe is None:
        return []
    mask = fill_gaps([d and not at for d, at in zip(S.door_open, S.attending)], 1)
    out = []
    for a, b in _runs_min(mask, S, S.cfg["T3_min_s"]):
        best = max(range(a, b), key=lambda i: S.tl.frames[i].safe_score)
        out.append(_event(S, "T3", a, b, min(0.95, 0.5 + S.tl.frames[best].safe_score), {}, [a, best, b - 1]))
    return out


def rule_t4(S: Series) -> list[Event]:
    """(a) two visitors at once while the cashier works; (b) a visitor on the cashier's side."""
    out = []
    k = S.sec(S.cfg["T4_min_s"])
    mask = [v >= 2 and pr for v, pr in zip(S.visitors, S.present)]
    for a, b in runs(fill_gaps(mask, 1)):
        if b - a < k:
            continue
        first_near = next((i for i in range(a, b) if S.visitor_near[i] >= 2), a)
        out.append(_event(S, "T4", first_near, b, 0.75, {"variant": "a", "persons": 2 + 1},
                          [a, first_near, min(S.n - 1, first_near + S.sec(2.0))]))
    mask_b = [intr and pr for intr, pr in zip(S.intruder, S.present)]
    for a, b in _runs_min(fill_gaps(mask_b, 1), S, S.cfg["T4b_min_s"]):
        if any(a <= e.frame_idx < b for e in out):
            continue
        out.append(_event(S, "T4", a, b, 0.7, {"variant": "b", "persons": 2}, [a, min(S.n - 1, a + S.sec(1.0)), b - 1]))
    return out


def rule_t5(S: Series) -> list[Event]:
    """Someone enters and leaves a personal bag / jacket on the desk."""
    out = []
    entries = [S._t_to_i(t) for t, _ in S.tl.entered]
    if S.tl.room_empty_start:
        first_in = next((i for i, s in enumerate(S.tl.frames) if s.n_in_room > 0), None)
        if first_in is not None:
            entries.append(first_in)
    if not entries:
        return out
    win = S.sec(S.cfg["T5_window_s"])
    for a, b in runs(S.items):
        if not any(0 <= a - e <= win for e in entries):
            continue
        obj = S.obj_boxes(a, {"bag", "money_bag", "jacket", "book"}, S.cfg["items_thr"])
        label = max(obj, key=lambda o: o.score).label if obj else "bag"
        start = max(0, a - (S.cfg["items_k"] - 1))
        out.append(_event(S, "T5", start, b, 0.65, {"object": "bag" if label in ("bag", "money_bag") else label},
                          [max(0, start - S.sec(3.0)), start, b - 1]))
    return out


def rule_t6(S: Series) -> list[Event]:
    """Cup / magazine on the work desk while cash is being handled."""
    out = []
    ctx = S.sec(S.cfg["T6_ctx_s"])
    for a, b in _runs_min([c or bk for c, bk in zip(S.cup, S.book)], S, S.cfg["T6_min_s"]):
        has_ctx = any(S.cash_ctx[max(0, a - ctx):min(S.n, b + ctx)]) or any(S.counting[max(0, a - ctx):min(S.n, b + ctx)])
        if not has_ctx:
            continue
        best = max(range(a, b), key=lambda i: max((o.score for o in S.tl.frames[i].objects if o.label in ("cup", "book")), default=0))
        out.append(_event(S, "T6", a, b, 0.65, {"object": "cup" if any(S.cup[a:b]) else "book"}, [a, best, b - 1]))
    return out


def rule_t7(S: Series) -> list[Event]:
    """A: cash in hand then hands hidden; B: a long-lying desk bundle vanishes; C: cash held low while standing."""
    out, fps = [], S.tl.sample_fps
    gone_k, hid_k = S.sec(S.cfg["T7_gone_s"]), S.sec(S.cfg["T7_hidden_s"])
    # A
    for a, b in runs(fill_gaps(S.cash_in_hand, 1)):
        if b - a < 2:
            continue
        horizon = min(S.n, b + S.sec(3.0))
        no_cash = [not S.cash_desk[i] and S.others_at_desk[i] == 0 for i in range(b, horizon)]
        hidden = [S.hands_hidden[i] and not S.counting[i] for i in range(b, horizon)]
        if sum(no_cash) >= gone_k and any(sum(hidden[j:j + hid_k]) >= hid_k for j in range(max(1, len(hidden) - hid_k + 1))):
            out.append(_event(S, "T7", b - 1, horizon, 0.55, {"variant": "A"}, [b - 1, min(S.n - 1, b + 1), min(S.n - 1, b + S.sec(2.0))]))
    # B
    for a, b in runs(S.cash_desk):
        if b - a < S.sec(10.0) or b >= S.n - 1:
            continue
        gone = all(not S.cash_desk[i] for i in range(b, min(S.n, b + 3)))
        if not gone:
            continue
        w0, w1 = max(0, b - S.sec(1.5)), min(S.n, b + S.sec(1.5))
        if any(S.others_at_desk[w0:w1]) or any(S.door_open[w0:w1]) or any(S.attending[w0:w1]) or any(S.counting[w0:w1]):
            continue
        out.append(_event(S, "T7", b, min(S.n, b + S.sec(2.0)), 0.55, {"variant": "B"}, [b - 1, b, min(S.n - 1, b + S.sec(1.5))]))
    # C
    for a, b in runs([cih and va for cih, va in zip(S.cash_in_hand, S.vacated)]):
        if b - a < S.sec(2.0) or any(S.counting[max(0, a - S.sec(10.0)):a]):
            continue
        if any(S.cash_desk[max(0, a - S.sec(5.0)):a]) and not S.cash_desk[a]:
            out.append(_event(S, "T7", a, b, 0.5, {"variant": "C"}, [max(0, a - 1), a, b - 1]))
    return out


def rule_t8(S: Series) -> list[Event]:
    """Client leaves while the cashier keeps counting the cash."""
    out = []
    client_k, win, count_k = S.sec(S.cfg["T8_client_s"]), S.sec(S.cfg["T8_window_s"]), S.sec(S.cfg["T8_count_s"])
    at_desk = [o > 0 for o in S.others_at_desk]
    for a, b in runs(fill_gaps(at_desk, 2)):
        if b - a < client_k or b >= S.n:
            continue
        cash_seen = any(S.cash_desk[max(0, b - S.sec(5.0)):b]) or any(S.cash_in_hand[max(0, b - S.sec(5.0)):b]) \
            or any(S.cash_ctx[max(0, b - S.sec(5.0)):b])
        if not cash_seen:
            continue
        counting = [S.counting[i] and S.present[i] and S.others_at_desk[i] == 0 for i in range(b, min(S.n, b + win))]
        for j in range(len(counting) - count_k + 1):
            if all(counting[j:j + count_k]):
                start = b + j
                end = start
                while end < S.n and S.counting[end]:
                    end += 1
                out.append(_event(S, "T8", start, max(end, start + 1), 0.6, {"client_left_t": round(S.t[b - 1], 1)},
                                  [max(0, b - 1), start, min(S.n - 1, start + S.sec(2.0))]))
                break
    return out


def rule_t9(S: Series) -> list[Event]:
    """A visitor puts a money bag on the desk and leaves within seconds; the bag stays."""
    out = []
    leave_k, stay_k = S.sec(S.cfg["T9_leave_s"]), S.sec(S.cfg["T9_stay_s"])
    for a, b in runs(S.bag_desk):
        a0 = max(0, a - (S.cfg["items_k"] - 1))            # placement frame
        if not any(S.others_at_desk[max(0, a0 - 2):a0 + 3]):
            continue
        gone = next((i for i in range(a0, min(S.n, a0 + leave_k)) if S.visitors[i] == 0 and
                     (i + 1 >= S.n or S.visitors[i + 1] == 0)), None)
        if gone is None or b - gone < stay_k:
            continue
        out.append(_event(S, "T9", a0, b, 0.7, {"visitor_left_at": round(S.t[gone], 1)},
                          [a0, min(S.n - 1, a0 + S.sec(1.0)), min(S.n - 1, gone)]))
    return out


def rule_t10(S: Series) -> list[Event]:
    """Cashier enters the empty room and lets go of the door at once (door not closed / locked)."""
    tl = S.tl
    if not tl.room_empty_start or tl.cashier_track_id is None:
        return []
    first = next((i for i, s in enumerate(tl.frames) if any(p.is_cashier and p.in_room for p in s.persons)), None)
    if first is None or first == 0:
        return []
    before = [any(p.is_cashier and p.at_door for p in tl.frames[j].persons) for j in range(max(0, first - 3), first)]
    if not any(before):
        return []
    dwell = 0
    while first + dwell < S.n and any(p.is_cashier and p.at_door for p in tl.frames[first + dwell].persons):
        dwell += 1
    if dwell > S.sec(S.cfg["T10_dwell_s"]):
        return []
    reach = any(tl.frames[j].cashier_present for j in range(first, min(S.n, first + S.sec(S.cfg["T10_reach_s"]))))
    if not reach:
        return []
    i = min(S.n - 1, first + dwell)
    return [_event(S, "T10", i, min(S.n, i + S.sec(2.0)), 0.55, {"dwell_s": round(dwell / tl.sample_fps, 1)},
                   [max(0, first - 2), i, min(S.n - 1, i + S.sec(2.0))])]


RULES = [rule_t1, rule_t2, rule_t3, rule_t4, rule_t5, rule_t6, rule_t7, rule_t8, rule_t9, rule_t10]


def run_rules(tl: Timeline, cfg: dict = CFG, enabled: list[str] | None = None) -> list[Event]:
    S = Series(tl, cfg)
    events: list[Event] = []
    for rule in RULES:
        tid = rule.__name__.split("_")[-1].upper()
        if enabled and tid not in enabled:
            continue
        events.extend(rule(S))
    return sorted(events, key=lambda e: e.t_start)
