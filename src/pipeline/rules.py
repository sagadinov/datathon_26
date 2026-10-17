"""Rule engine T1-T9: scene timeline -> temporal events (t_start, t_end, score, type).

Every rule is a pure function of per-frame boolean/score series derived from the Timeline,
debounced with minimum durations expressed in seconds (converted with the sample rate), so the
same thresholds work at 2 fps or at a reduced rate under the time budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.state import FrameState, Timeline, runs

CFG = {
    "cash_score": 0.35, "money_bag_score": 0.35, "cup_score": 0.42, "book_score": 0.42,
    "bag_score": 0.38, "jacket_score": 0.38, "safe_open_score": 0.47, "safe_open_rel": 0.25,
    "near_gap": 0.3,
    "present_k": 3, "present_n": 5,                 # object present if seen in k of the last n frames
    "T1_min_s": 2.0, "T1_gap": 0.25,                # cashier farther than this (desk heights) = away
    "T2_min_s": 1.5, "T3_min_s": 3.0, "T4_min_s": 2.0, "T5_min_s": 3.0, "T6_min_s": 3.0,
    "T8_min_s": 3.0, "T8_client_min_s": 2.0, "T9_leave_s": 6.0, "T7_min_s": 1.0,
}


@dataclass
class Event:
    type_id: str
    t_start: float
    t_end: float
    score: float
    frame_idx: int                      # best evidence frame
    details: dict = field(default_factory=dict)


def _sec(tl: Timeline, s: float) -> int:
    return max(1, int(round(s * tl.sample_fps)))


def _present(series: list[float], thr: float, k: int, n: int) -> list[bool]:
    """True where the score was >= thr in at least k of the last n frames (debounced presence)."""
    hits = [v >= thr for v in series]
    out = []
    for i in range(len(hits)):
        out.append(sum(hits[max(0, i - n + 1): i + 1]) >= min(k, i + 1))
    return out


def _obj_score(st: FrameState, label: str) -> float:
    return max((o.score for o in st.objects if o.label == label and o.on_desk), default=0.0)


class Series:
    """All per-frame series the rules need, computed once."""

    def __init__(self, tl: Timeline, cfg: dict = CFG):
        self.tl, self.cfg, F = tl, cfg, tl.frames
        k, n = cfg["present_k"], cfg["present_n"]
        self.t = [s.t for s in F]
        self.cash_s = [_obj_score(s, "cash") for s in F]
        self.bag_s = [_obj_score(s, "money_bag") for s in F]
        self.cash = _present(self.cash_s, cfg["cash_score"], k, n)
        self.money_bag = _present(self.bag_s, cfg["money_bag_score"], k, n)
        self.valuables = [a or b for a, b in zip(self.cash, self.money_bag)]
        self.cup = _present([_obj_score(s, "cup") for s in F], cfg["cup_score"], k, n)
        self.book = _present([_obj_score(s, "book") for s in F], cfg["book_score"], k, n)
        self.pbag = _present([_obj_score(s, "bag") for s in F], cfg["bag_score"], k, n)
        self.jacket = _present([_obj_score(s, "jacket") for s in F], cfg["jacket_score"], k, n)
        # safe door: relative to this video's own closed-door baseline (20th percentile), with a floor
        so = [s.safe_door_open for s in F]
        base = float(np.percentile(so, 20)) if so else 0.0
        self.safe_thr = max(cfg["safe_open_score"], base + cfg["safe_open_rel"])
        self.safe_open = _present(so, self.safe_thr, k, n)
        self.lights_off = [s.lights_off for s in F]
        self.screen_on = [s.screen_on for s in F]
        self.n_persons = [s.n_persons for s in F]
        self.cashier_near = [any(p.desk_gap <= cfg["T1_gap"] for p in s.persons) for s in F]
        self.cashier_seat = [any(p.at_seat or (p.is_cashier and p.desk_gap == 0 and p.anchor_overlap >= 0.15)
                                 for p in s.persons) for s in F]
        self.someone_at_safe = [any(p.at_safe for p in s.persons) for s in F]
        self.near_count = [sum(p.desk_gap <= cfg["near_gap"] for p in s.persons) for s in F]
        self.cashier_side = [sum(p.at_seat for p in s.persons) for s in F]
        self.client = [any((not p.is_cashier) and p.desk_gap <= cfg["near_gap"] for p in s.persons) for s in F]
        self.hands_below = [s.hands_below_desk > 0 for s in F]


def _events_from_runs(mask: list[bool], tl: Timeline, type_id: str, min_s: float, score_fn, details_fn=None) -> list[Event]:
    out = []
    for a, b in runs(mask):
        if b - a < _sec(tl, min_s):
            continue
        sc = float(score_fn(a, b))
        mid = a + min(b - a - 1, _sec(tl, 1.0))      # evidence ~1 s into the event
        out.append(Event(type_id, tl.frames[a].t, tl.frames[b - 1].t, sc, mid, details_fn(a, b) if details_fn else {}))
    return out


def rule_t1(S: Series) -> list[Event]:
    """Valuables (cash / money bag) on the desk while no one attends the desk."""
    mask = [v and not near for v, near in zip(S.valuables, S.cashier_near)]
    return _events_from_runs(mask, S.tl, "T1", S.cfg["T1_min_s"],
                             lambda a, b: min(0.95, 0.5 + max(max(S.cash_s[a:b]), max(S.bag_s[a:b]))),
                             lambda a, b: {"object": "cash" if max(S.cash_s[a:b]) >= max(S.bag_s[a:b]) else "money_bag",
                                           "away_s": round((b - a) / S.tl.sample_fps, 1)})


def rule_t2(S: Series) -> list[Event]:
    """Lights switched off (or room empty) with the laptop screen still on and nobody at the seat."""
    mask = [scr and not seat and (off or n == 0) for scr, seat, off, n in
            zip(S.screen_on, S.cashier_seat, S.lights_off, S.n_persons)]
    return _events_from_runs(mask, S.tl, "T2", S.cfg["T2_min_s"], lambda a, b: 0.8,
                             lambda a, b: {"lights_off": bool(S.lights_off[a])})


def rule_t3(S: Series) -> list[Event]:
    """Safe door open while nobody stands at the safe."""
    mask = [o and not at for o, at in zip(S.safe_open, S.someone_at_safe)]
    return _events_from_runs(mask, S.tl, "T3", S.cfg["T3_min_s"],
                             lambda a, b: min(0.95, 0.4 + max(S.tl.frames[i].safe_door_open for i in range(a, b))))


def rule_t4(S: Series) -> list[Event]:
    """A third person near the desk, or two people on the cashier's side at once."""
    mask = [nc >= 3 or cs >= 2 for nc, cs in zip(S.near_count, S.cashier_side)]
    return _events_from_runs(mask, S.tl, "T4", S.cfg["T4_min_s"], lambda a, b: 0.7,
                             lambda a, b: {"persons": max(S.near_count[a:b])})


def rule_t5(S: Series) -> list[Event]:
    """Personal bag or jacket lying on the desk."""
    mask = [p or j for p, j in zip(S.pbag, S.jacket)]
    return _events_from_runs(mask, S.tl, "T5", S.cfg["T5_min_s"], lambda a, b: 0.6,
                             lambda a, b: {"object": "bag" if any(S.pbag[a:b]) else "jacket"})


def rule_t6(S: Series) -> list[Event]:
    """Cup / drink or a magazine on the work desk."""
    mask = [c or bk for c, bk in zip(S.cup, S.book)]
    return _events_from_runs(mask, S.tl, "T6", S.cfg["T6_min_s"], lambda a, b: 0.6,
                             lambda a, b: {"object": "cup" if any(S.cup[a:b]) else "book"})


def rule_t7(S: Series) -> list[Event]:
    """Hands below the desk edge while valuables were on the desk shortly before (cash leaving the view)."""
    recent = [any(S.valuables[max(0, i - 6): i + 1]) for i in range(len(S.valuables))]
    mask = [h and r and seat for h, r, seat in zip(S.hands_below, recent, S.cashier_seat)]
    return _events_from_runs(mask, S.tl, "T7", S.cfg["T7_min_s"], lambda a, b: 0.5)


def rule_t8(S: Series) -> list[Event]:
    """Client left while cash stays on the desk and the cashier keeps working with it."""
    tl, cfg = S.tl, S.cfg
    min_client = _sec(tl, cfg["T8_client_min_s"])
    mask = [False] * len(tl)
    had_client, since = 0, None
    for i in range(len(tl)):
        if S.client[i]:
            had_client += 1
            since = None
        else:
            if had_client >= min_client:
                since = i if since is None else since
            if since is not None and S.cash[i] and S.cashier_seat[i]:
                mask[i] = True
    return _events_from_runs(mask, tl, "T8", cfg["T8_min_s"], lambda a, b: 0.55)


def rule_t9(S: Series) -> list[Event]:
    """A visitor brings a money bag, it appears on the desk, and the visitor leaves within seconds."""
    tl, cfg = S.tl, S.cfg
    out = []
    for a, b in runs(S.money_bag):
        before = any(S.client[max(0, a - 6): a + 1])
        leave_by = min(len(tl), a + _sec(tl, cfg["T9_leave_s"]))
        gone = [i for i in range(a, leave_by) if not any(S.client[i: i + 2])]
        if before and gone and b >= gone[0] + 2:
            out.append(Event("T9", tl.frames[a].t, tl.frames[b - 1].t, 0.6, min(b - 1, a + 2),
                             {"visitor_left_at": tl.frames[gone[0]].t}))
    return out


RULES = [rule_t1, rule_t2, rule_t3, rule_t4, rule_t5, rule_t6, rule_t7, rule_t8, rule_t9]


def run_rules(tl: Timeline, cfg: dict = CFG, enabled: list[str] | None = None) -> list[Event]:
    S = Series(tl, cfg)
    events: list[Event] = []
    for rule in RULES:
        tid = rule.__name__[-2:].upper()
        if enabled and tid not in enabled:
            continue
        events.extend(rule(S))
    return sorted(events, key=lambda e: e.t_start)
