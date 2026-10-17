"""T7 "cash out of sight" with the cashier's wrists from the pose model (docs/violations_observed.md, T7-A / P10).

    rule_t7_pose(S, wrists) -> list[Event]       wrists = pose.wrists_series(frames, tl)

Variant A: cash in the cashier's hands (S.cash_in_hand) for >= 2 frames, then within 3 s the hands go out of sight
and the cash does not come back (desk / hands / counting machine) for T7P_return_s or until the clip ends.

"Out of sight" depends on where the camera stands relative to the cashier (derived from boxes, no pixels):
  * behind view (camera A: the cashier's box reaches below the desk's near edge): hands in the lap are occluded
    by her own back, so both wrist confidences collapse to ~0.4-0.5 (same point, 20-30 px apart) AND the elbows
    drop to the lower torso (elbow_rel >= 0.52). Typing / handling things on the desk hides the wrists too, but
    then the elbows stay high (X-35 0.26-0.49, X-39 0.12-0.29, X-47 0.23-0.45 vs lap 0.57-0.77 in X-1/2/5/19).
  * front view (camera B: the desk between camera and cashier): wrists stay visible (conf 0.7-0.8) at the far
    desk edge in every clip, so the spec's criterion wrist_y >= desk.y0 - 5 px holds all the time, for X-44 and
    for the clean X-10/20/21/43/45 alike -> disabled (cfg "front_edge"); no front-view events are produced.
Exclusions: a hand extended across the desk (wrist well above the shoulder line in the image) in the 2 s before
the hands drop = hand-to-hand transfer or feeding the counting machine (X-3 17-18, X-5 24-26.5); safe open /
attended in the window (cash into the safe, X-8); cash seen again within T7P_return_s (X-5 28-31: lap 3.5 s,
then back on the desk).
Timecode: the first cash-in-hand frame after the cash was last seen in a legitimate place (desk / counting
machine) inside the run, but not earlier than 5 s before the hands drop (X-1: 20.0 for label 22).
"""
from __future__ import annotations

import math

import numpy as np

from src.pipeline.rules import Event, Series
from src.pipeline.state import fill_gaps, runs

CFG_T7P = {
    "hidden_conf": 0.60,     # max(lw, rw) confidence below this = wrists not seen
    "elbow_low": 0.52,       # (max elbow y - shoulder y) / torso >= this = elbows at the lower torso (lap)
    "reach_rel": -0.50,      # (wrist y - shoulder y) / torso <= this with conf >= 0.6 = arm extended across the desk
    "reach_s": 2.0,          # look-back for the reach exclusion before the hands drop
    "after_s": 3.0,          # the hands must drop within this time after the last cash-in-hand frame
    "lap_min": 3,            # frames out of sight (2 are enough when the clip ends there)
    "lap_min_end": 2,
    "return_s": 4.0,         # cash must stay unseen this long after the hands drop (or until the clip ends)
    "safe_s": 2.0,           # look-back for safe open / attending
    "cih_min": 2,
    "edge_px": 5.0,          # front view: wrist y >= desk.y0 - edge_px
    "score": 0.6,
}


def _kp(w: dict, name: str):
    return w.get("kps", {}).get(name) or (math.nan, math.nan, 0.0)


def view_from_behind(S: Series) -> bool | None:
    """True = camera behind the cashier (her box reaches below the desk's near edge), False = front/side view."""
    desk = S.tl.layout.desk
    if not desk:
        return None
    bottoms = [s.cashier_box[3] for s in S.tl.frames if s.cashier_box is not None and s.cashier_present and not s.seat_vacated]
    if len(bottoms) < 3:
        return None
    dh = desk[3] - desk[1]
    return float(np.median(bottoms)) > desk[3] + 0.05 * dh


def pose_features(S: Series, wrists: list[dict], cfg: dict = CFG_T7P) -> dict[str, list]:
    """Per-frame scalar features from the wrist dicts (nan / False when the cashier was not seen)."""
    n = S.n
    conf, elb_rel, wr_rel, reach, edge = [], [], [], [], []
    desk = S.tl.layout.desk
    for i in range(n):
        w = wrists[i] if i < len(wrists) else None
        if not w or w.get("box") is None or not w.get("kps"):
            conf.append(0.0); elb_rel.append(math.nan); wr_rel.append(math.nan); reach.append(False); edge.append(False)
            continue
        lw, rw = w["lw"], w["rw"]
        c = max(lw[2], rw[2])
        sho = (_kp(w, "l_sho")[1] + _kp(w, "r_sho")[1]) / 2
        hip = (_kp(w, "l_hip")[1] + _kp(w, "r_hip")[1]) / 2
        torso = max(1.0, hip - sho)
        conf.append(c)
        elb_rel.append((max(_kp(w, "l_elb")[1], _kp(w, "r_elb")[1]) - sho) / torso)
        wr_rel.append((max(lw[1], rw[1]) - sho) / torso)
        reach.append(any(k[2] >= cfg["hidden_conf"] and (k[1] - sho) / torso <= cfg["reach_rel"] for k in (lw, rw)))
        edge.append(bool(desk) and c >= cfg["hidden_conf"] and max(lw[1], rw[1]) >= desk[1] - cfg["edge_px"])
    return {"conf": conf, "elb_rel": elb_rel, "wr_rel": wr_rel, "reach": reach, "edge": edge}


def out_of_sight(S: Series, feats: dict, behind: bool | None, cfg: dict = CFG_T7P) -> list[bool]:
    """Per frame: the cashier's hands are out of sight (lap, below the desk line)."""
    hidden = [0.0 < c < cfg["hidden_conf"] for c in feats["conf"]]
    if behind is False:
        # Front view: the wrists rest on the desk line in every camera-B clip, positive or clean (X-44 as well as
        # X-10/20/21/43/45), so the criterion carries no information; it is kept only for experiments.
        return list(feats["edge"]) if cfg.get("front_edge", False) else [False] * len(hidden)
    low = [not math.isnan(e) and e >= cfg["elbow_low"] for e in feats["elb_rel"]]
    return [h and l for h, l in zip(hidden, low)]


def rule_t7_pose(S: Series, wrists: list[dict], cfg: dict = CFG_T7P) -> list[Event]:
    """T7-A with pose: cash in hand -> hands out of sight -> cash does not come back (see module docstring)."""
    if not wrists or S.n == 0:
        return []
    n, sec = S.n, S.sec
    behind = view_from_behind(S)
    feats = pose_features(S, wrists, cfg)
    lap = out_of_sight(S, feats, behind, cfg)
    cash_client = [any(o.label == "cash" and o.score >= 0.3 and _in_visitor(o.box, s) for o in s.objects) for s in S.tl.frames]
    cash_seen = [cd or cih or cnt or cc for cd, cih, cnt, cc in zip(S.cash_desk, S.cash_in_hand, S.counting, cash_client)]
    safe = [d or a for d, a in zip(S.door_open, S.attending)]
    out: list[Event] = []
    for a, b in runs(fill_gaps(S.cash_in_hand, 1)):
        if b - a < cfg["cih_min"] or not S.seated[b - 1]:
            continue
        horizon = min(n, b + sec(cfg["after_s"]))
        for s in range(b, horizon):
            if not lap[s] or (s > b and lap[s - 1]):
                continue                         # not the start of an out-of-sight run
            e = s
            while e < n and lap[e]:
                e += 1
            if (e - s) < (cfg["lap_min_end"] if e >= n else cfg["lap_min"]):
                continue
            clear_to = min(n, max(e, s + sec(cfg["return_s"])))
            if any(cash_seen[b:clear_to]):
                continue                         # the cash came back: counted / put on the desk / machine
            if any(feats["reach"][max(0, s - sec(cfg["reach_s"])):s]):
                continue                         # hand-to-hand transfer / counting machine just before
            if any(safe[max(0, b - sec(cfg["safe_s"])):e]):
                continue                         # cash into the safe
            if not all(S.seated[s:e]):
                continue
            t_ev = _event_frame(S, a, b, s, sec)
            details = {"variant": "A-pose", "view": "behind" if behind else "front", "hands_hidden_from": S.t[s],
                       "hidden_frames": e - s, "last_cash_in_hand": S.t[b - 1],
                       "wrist_conf": round(float(np.mean(feats["conf"][s:e])), 2)}
            out.append(Event("T7", S.t[t_ev], S.t[e - 1], cfg["score"], b - 1, details,
                             [b - 1, s, min(n - 1, s + 2)]))
            break
    return out


def _in_visitor(box, s) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(p.in_room and not p.is_cashier and p.box[0] <= cx <= p.box[2] and p.box[1] <= cy <= p.box[3] for p in s.persons)


def _event_frame(S: Series, a: int, b: int, s: int, sec) -> int:
    """First cash-in-hand frame after the cash was last on the desk / in the machine, >= 5 s before the drop."""
    last_ok = max((i for i in range(a, b) if S.cash_desk[i] or S.counting[i]), default=a - 1)
    return max(min(last_ok + 1, b - 1), s - sec(5.0), a)
