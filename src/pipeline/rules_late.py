"""Rules T7 (cash out of sight, gated off without pose) and T10 (door not closed on entry).

Split out of rules.py to keep files under 400 lines; they use the same Series / Event helpers.
"""
from __future__ import annotations

from src.pipeline.rules import Event, Series, _event, _runs_min  # imported late by rules.py itself
from src.pipeline.state import fill_gaps, runs
from src.pipeline.zones import inter_area

def rule_t7(S: Series) -> list[Event]:
    """A: cash in hand then hands hidden; B: a long-lying desk bundle vanishes; C: cash held low while standing."""
    out, fps = [], S.tl.sample_fps
    gone_k, hid_k = S.sec(S.cfg["T7_gone_s"]), S.sec(S.cfg["T7_hidden_s"])
    # A
    for a, b in (runs(fill_gaps(S.cash_in_hand, 1)) if S.cfg.get("T7_A", False) else []):
        if b - a < 3 or not any(S.seated[a:b]) or not any(S.hands_seen[a:b]):
            continue
        horizon = min(S.n, b + S.sec(3.0))
        no_cash = [not S.cash_desk[i] and S.others_at_desk[i] == 0 and not S.cash_in_hand[i] for i in range(b, horizon)]
        hidden = [S.hands_hidden[i] and not S.counting[i] and S.seated[i] for i in range(b, horizon)]
        if sum(no_cash) >= gone_k and any(sum(hidden[j:j + hid_k]) >= hid_k for j in range(max(1, len(hidden) - hid_k + 1))):
            out.append(_event(S, "T7", b - 1, horizon, 0.55, {"variant": "A"}, [b - 1, min(S.n - 1, b + 1), min(S.n - 1, b + S.sec(2.0))]))
    # B (disabled by default: desk cash is rarely detected and the variant fired on clean videos)
    for a, b in (runs(S.cash_desk) if S.cfg.get("T7_B", False) else []):
        if b - a < S.sec(10.0) or b >= S.n - 1:
            continue
        gone = all(not S.cash_desk[i] for i in range(b, min(S.n, b + 3)))
        if not gone:
            continue
        w0, w1 = max(0, b - S.sec(1.5)), min(S.n, b + S.sec(1.5))
        if any(S.others_at_desk[w0:w1]) or any(S.door_open[w0:w1]) or any(S.attending[w0:w1]) or any(S.counting[w0:w1]):
            continue
        if not all(S.seated[w0:w1]) or any(S.lights_off[w0:w1]):
            continue
        last = S.obj_boxes(b - 1, {"cash"}, S.cfg["cash_thr"])
        cb = S.tl.frames[b].cashier_box
        if last and cb is not None:
            ob = max(last, key=lambda o: o.score).box
            ex = (ob[0] - 0.5 * (ob[2] - ob[0]), ob[1] - 0.5 * (ob[3] - ob[1]), ob[2] + 0.5 * (ob[2] - ob[0]), ob[3] + 0.5 * (ob[3] - ob[1]))
            if inter_area(ex, cb) > 0:
                continue   # the cashier's body covers the spot: occlusion, not disappearance
        out.append(_event(S, "T7", b, min(S.n, b + S.sec(2.0)), 0.55, {"variant": "B"}, [b - 1, b, min(S.n - 1, b + S.sec(1.5))]))
    # C (disabled: too many false alarms without a pose model; kept for VLM-backed runs)
    for a, b in ([] if not S.cfg.get("T7_C", False) else runs([cih and va for cih, va in zip(S.cash_in_hand, S.vacated)])):
        if b - a < S.sec(2.0) or any(S.counting[max(0, a - S.sec(10.0)):a]):
            continue
        if any(S.cash_desk[max(0, a - S.sec(5.0)):a]) and not S.cash_desk[a]:
            out.append(_event(S, "T7", a, b, 0.5, {"variant": "C"}, [max(0, a - 1), a, b - 1]))
    return out


def rule_t10(S: Series) -> list[Event]:
    """Cashier enters the empty room and lets go of the door at once (door not closed / locked)."""
    tl = S.tl
    if not tl.room_empty_start or tl.cashier_track_id is None:
        return []
    first = next((i for i, s in enumerate(tl.frames) if any(p.is_cashier and p.in_room for p in s.persons)), None)
    if first is None or first == 0:
        return []
    cid = tl.cashier_track_id
    before = [any(p.track_id == cid and p.at_door for p in tl.frames[j].persons) for j in range(max(0, first - 4), first)]
    if not any(before):
        return []
    # dwell = frames after crossing during which the cashier has not yet moved towards the desk/camera
    H = tl.layout.frame_h
    b0 = tl.frames[first].cashier_box
    dwell = 0
    while first + dwell < S.n:
        b = tl.frames[first + dwell].cashier_box
        if b is None or b0 is None or b[1] >= b0[1] + 0.05 * H:
            break
        dwell += 1
    if dwell > S.sec(S.cfg["T10_dwell_s"]):
        return []
    reach = any(tl.frames[j].cashier_present for j in range(first, min(S.n, first + S.sec(S.cfg["T10_reach_s"]))))
    if not reach:
        return []
    i = min(S.n - 1, first + dwell)
    return [_event(S, "T10", i, min(S.n, i + S.sec(2.0)), 0.55, {"dwell_s": round(dwell / tl.sample_fps, 1)},
                   [max(0, first - 2), i, min(S.n - 1, i + S.sec(2.0))])]

