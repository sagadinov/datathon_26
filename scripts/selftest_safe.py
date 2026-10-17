"""Self-test of src/pipeline/safe.py on cached frames + detections (never runs the detector).

    .venv/bin/python scripts/selftest_safe.py              # run all acceptance checks (spec P4/P8)
    .venv/bin/python scripts/selftest_safe.py --show 20 0 12   # per-frame signals of X-20 between 0 and 12 s

Stand-ins for modules that may still be stubs: persons = person dets with score >= 0.35 and height >= 0.12*H
(no dedup), cashier = the track with the most frames overlapping layout.anchor (box held from any person with
IoU(seat) >= 0.3 when the track is lost), hands = hand dets with score >= 0.20.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.labels import index_videos  # noqa: E402
from src.pipeline import safe  # noqa: E402
from src.pipeline.detect_cache import load_any_cached  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402
from src.pipeline.track import iou, track_boxes  # noqa: E402
from src.pipeline.zones import area, build_layout, inter_area  # noqa: E402


class StandInCashier:
    """Minimal CashierModel stand-in: box[i], seat_box, ref_height, track_id."""

    def __init__(self, persons, ids, tracks, layout):
        n = len(persons)
        anchor = layout.anchor
        best, best_k = None, 0
        for tid, t in tracks.items():
            k = sum(1 for b in t.boxes.values() if anchor and inter_area(b, anchor) / area(anchor) >= 0.15)
            if k > best_k:
                best, best_k = tid, k
        self.track_id = best
        self.seat_box = layout.cashier_seat
        self.box = []
        for i in range(n):
            b = tracks[best].boxes.get(i) if best is not None else None
            if b is None and self.seat_box is not None:
                cands = [p.box for p in persons[i] if iou(p.box, self.seat_box) >= 0.3]
                b = cands[0] if cands else None
            self.box.append(b)


def load_video(n: int):
    frames = extract_frames(index_videos()[n])
    dets = load_any_cached(frames)
    assert dets is not None, f"no cached detections for X-{n}"
    import cv2

    img = cv2.imread(str(frames.frames[0].path))
    h, w = img.shape[:2]
    persons = [[d for d in ds if d.label == "person" and d.score >= 0.35 and d.box[3] - d.box[1] >= 0.12 * h]
               for ds in dets]
    ids, tracks = track_boxes(persons, w, h)
    layout = build_layout(dets, w, h, tracks)
    layout.safe = safe.pick_safe_anchor(dets, layout)
    cm = StandInCashier(persons, ids, tracks, layout)
    hands = [[d for d in ds if d.label == "hand" and d.score >= 0.20] for ds in dets]
    attending = safe.attending_safe_series(persons, hands, cm, layout)
    door, scores = safe.door_open_series(dets, frames, layout, persons)
    t = [f.timestamp_sec for f in frames.frames]
    return dict(n=n, frames=frames, dets=dets, persons=persons, layout=layout, cm=cm, hands=hands, t=t,
                attending=attending, door=door, scores=scores)


def fmt(b):
    return "None" if b is None else "(%d,%d,%d,%d)" % tuple(round(v) for v in b)


def show(n: int, t0: float, t1: float) -> None:
    v = load_video(n)
    lay = v["layout"]
    print(f"X-{n}: safe={fmt(lay.safe)} presence={lay.presence.get('safe', 0):.2f} desk={fmt(lay.desk)} "
          f"door={fmt(lay.door)} seat={fmt(lay.cashier_seat)}")
    if lay.safe is None:
        return
    sig = safe._door_signals(v["dets"], v["frames"], lay, v["persons"], True)
    print(f"B0={fmt(sig['b0'])} b0_base={sig['b0_base']} p20={sig['p20']:.2f}")
    print("    t  att door gated v1 v2 v3  diff frac | persons (x1,y1,x2,y2)h")
    for i, t in enumerate(v["t"]):
        if not (t0 <= t <= t1):
            continue
        d = sig["diff"][i]
        ps = " ".join(f"{fmt(p.box)}h{int(p.box[3]-p.box[1])}" for p in v["persons"][i])
        dd = "   --     " if d is None else f"{d[0]:5.1f} {d[1]:.2f}"
        print(f"{t:5.1f}  {int(v['attending'][i])}   {int(v['door'][i])}   {sig['gated'][i]:.2f}  {int(sig['v1'][i])}  "
              f"{int(sig['v2'][i])}  {int(sig['v3'][i])} {dd} | {ps}")


# ------------------------------------------------------------------------------------------------ checks
def _window(v, t0, t1, key):
    return [v[key][i] for i, t in enumerate(v["t"]) if t0 - 1e-6 <= t <= t1 + 1e-6]


def check_all() -> int:
    failures: list[str] = []
    results: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        results.append(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    def anchor_near(n, box, thr=0.8):
        v = load_video(n)
        a = v["layout"].safe
        ok = a is not None and iou(a, box) >= thr
        expect(ok, f"X-{n} anchor {fmt(a)} ~ {fmt(box)} (IoU {0 if a is None else iou(a, box):.2f})")
        return v

    def coverage(v, t0, t1, key):
        w = _window(v, t0, t1, key)
        return sum(w) / max(1, len(w)), len(w)

    # (a) anchors
    anchor_near(20, (141, 188, 197, 265))
    for n in (18, 16, 17):
        anchor_near(n, (237, 242, 323, 350))
    anchor_near(39, (232, 258, 339, 433))
    for n in (11, 13, 14, 15, 31, 46):
        v = load_video(n)
        a, door = v["layout"].safe, v["layout"].door
        not_door = a is None or door is None or iou(a, door) < 0.3
        expect(not_door, f"X-{n} anchor {fmt(a)} is not the glass door {fmt(door)} (presence {v['layout'].presence['safe']:.2f})")
    v = load_video(13)
    expect(v["layout"].safe is not None and iou(v["layout"].safe, (162, 199, 221, 272)) >= 0.8,
           f"X-13 anchor {fmt(v['layout'].safe)} ~ (162,199,221,272)")
    for n in range(24, 30):
        v = load_video(n)
        a = v["layout"].safe
        expect(a is None or iou(a, (12, 338, 102, 433)) < 0.3,
               f"X-{n} anchor {fmt(a)} is not the white cabinet under the desk (presence {v['layout'].presence['safe']:.2f})")

    # (b) attending_safe
    att_windows = {5: [(46, 52.5)], 48: [(46, 52.5)], 14: [(34.5, 43)], 19: [(37.5, 40)], 21: [(1, 6.5)],
                   35: [(40.5, 47)], 39: [(29.5, 35.5), (66.5, 71.5)]}
    for n, wins in att_windows.items():
        v = load_video(n)
        for t0, t1 in wins:
            cov, k = coverage(v, t0, t1, "attending")
            expect(cov >= 0.999, f"X-{n} attending True {t0}-{t1} s ({cov:.0%} of {k} frames)")
    v = load_video(20)
    cov_after, k = coverage(v, 5.0, 1e9, "attending")
    expect(cov_after == 0.0, f"X-20 attending False from 5.0 s on ({1 - cov_after:.0%} False of {k} frames)")
    cov, k = coverage(v, 3.5, 4.5, "attending")
    expect(cov >= 0.999, f"X-20 attending True 3.5-4.5 s ({cov:.0%} of {k} frames)")

    # (c) door_open
    v = load_video(20)
    cov, k = coverage(v, 5.0, 1e9, "door")
    expect(cov >= 0.999, f"X-20 door_open True from 5.0 s to the end ({cov:.0%} of {k} frames)")
    first = next((t for i, t in enumerate(v["t"]) if v["door"][i]), None)
    expect(first is not None and first <= 5.0, f"X-20 door_open already True at {first} s (needed by 4.5-5.0 s; the "
                                               f"panel starts moving at 2.0 s while the cashier attends)")
    t3 = next((t for i, t in enumerate(v["t"]) if v["door"][i] and not v["attending"][i]), None)
    expect(t3 is not None and 4.5 <= t3 <= 5.5, f"X-20 first door_open & !attending at {t3} s (T3 cue 5.0-5.5)")
    # spec windows cover the whole safe episode (opening ... locking); the door panel is physically open (checked on
    # the frames) during X-5/48 47-50, X-39 30-33 and 67-69.5, X-35 41.5-44.5, X-19 37.5-40
    door_windows = {5: [(47, 50)], 48: [(47, 50)], 19: [(37.5, 40)], 39: [(30, 33), (67, 69.5)], 35: [(41.5, 44.5)]}
    for n, wins in door_windows.items():
        v = load_video(n)
        for t0, t1 in wins:
            cov, k = coverage(v, t0, t1, "door")
            expect(cov >= 0.7, f"X-{n} door_open True around {t0}-{t1} s ({cov:.0%} of {k} frames)")
    for n in (34, 37, 38, 4, 3):
        v = load_video(n)
        cov, k = coverage(v, 0, 1e9, "door")
        expect(cov == 0.0, f"X-{n} door_open False everywhere ({sum(v['door'])} True of {k} frames, "
                           f"max score {max(v['scores']):.2f}, anchor {fmt(v['layout'].safe)})")
    v = load_video(23)
    if v["layout"].safe is None:
        expect(not any(v["door"]), "X-23 no anchor -> door_open all False")
    else:
        cov, k = coverage(v, 7.5, 10.5, "door")
        expect(cov >= 0.6, f"X-23 door_open True 7.5-10.5 s with anchor {fmt(v['layout'].safe)} ({cov:.0%} of {k} frames)")

    # T3 silence on clean videos with safe episodes: no run of door_open & !attending >= 4 frames
    for n in (21, 23, 13, 25, 27, 8, 12, 14):
        v = load_video(n)
        bad = [v["door"][i] and not v["attending"][i] for i in range(len(v["t"]))]
        run, best, start = 0, 0, None
        for i, b in enumerate(bad):
            run = run + 1 if b else 0
            if run > best:
                best, start = run, v["t"][i - run + 1]
        expect(best < 4, f"X-{n} (clean) longest door_open & !attending run = {best} frames at {start} s")

    print("\n".join(results))
    print(f"\n{len(results) - len(failures)} passed, {len(failures)} failed")
    return len(failures)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs=3, metavar=("N", "T0", "T1"))
    args = ap.parse_args()
    if args.show:
        show(int(args.show[0]), float(args.show[1]), float(args.show[2]))
        return
    sys.exit(1 if check_all() else 0)


if __name__ == "__main__":
    main()
