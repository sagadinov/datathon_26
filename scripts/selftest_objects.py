"""Offline self-test for src/pipeline/objects.py against the per-video expectations of the spec
(docs/violations_observed.md, P5 / P10 and the T1/T5/T6/T7/T8/T9 sections).

Uses cached frames + detections only (never runs the detector). Stand-ins for the still-stubbed modules:
persons = person dets with score >= 0.35 and height >= 0.12*H; in-room = bottom edge not above
desk.y0 - 0.1*desk_h and not a box fully inside the door zone with y1 in its upper 38%; cashier box = the in-room
person with the largest overlap with layout.anchor (the one nearest to the anchor when nobody overlaps it), held
for <= 2 frames of dropout.

    .venv/bin/python scripts/selftest_objects.py                 # all checks, exit code 1 when a hard check fails
    .venv/bin/python scripts/selftest_objects.py --cache oldest  # against the oldest cache file per video (v1 vocabulary)
    .venv/bin/python scripts/selftest_objects.py --cache 82a00e8276 --show 17 26   # one config hash; print timelines only

--cache newest (default) | oldest | <config hash>: which detection cache file to read for every video. Two caches exist
for the dataset (v1 vocabulary a2bcb32805 = the numbers quoted in the spec; the extended D1 vocabulary 82a00e8276 =
what the current detect.py config produces), and the detector scores of small objects differ between them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from eval.labels import index_videos  # noqa: E402
from src.common import CACHE_DIR  # noqa: E402
from src.pipeline.detect import Detection  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402
from src.pipeline.objects import ObjectModel  # noqa: E402
from src.pipeline.objects_util import contain_frac, runs  # noqa: E402
from src.pipeline.track import iou, track_boxes  # noqa: E402
from src.pipeline.zones import build_layout, inter_area  # noqa: E402

T1_LABELS, T5_LABELS, T9_LABELS = {"cash", "money_bag", "bag", "book"}, {"bag", "money_bag", "jacket", "book"}, {"money_bag", "bag"}


class StandInCashier:
    """Minimal replacement for people.CashierModel: only `.box[i]` is used by ObjectModel."""

    def __init__(self, persons, layout, h):
        self.box = []
        held, miss = None, 0
        desk, door = layout.desk, layout.door_zone
        for ps in persons:
            inroom = []
            for p in ps:
                b = p.box
                if desk and b[3] < desk[1] - 0.1 * (desk[3] - desk[1]):
                    continue
                if door and contain_frac(b, door) >= 0.99 and layout.door and b[1] < layout.door[1] + 0.38 * (layout.door[3] - layout.door[1]):
                    continue
                inroom.append(p)
            cb = None
            if inroom and layout.anchor:
                a = layout.anchor
                acx = (a[0] + a[2]) / 2
                cb = max(inroom, key=lambda p: (inter_area(p.box, a), -abs((p.box[0] + p.box[2]) / 2 - acx))).box
            if cb is None and held is not None and miss < 2:
                cb, miss = held, miss + 1
            elif cb is not None:
                held, miss = cb, 0
            self.box.append(cb)


CACHE_SEL = "newest"


def load_cached(frames, sel):
    """Detections from one cache file of this frame set, never running the detector (see --cache)."""
    import json

    key = frames.frames[0].path.parent.name
    cands = sorted((CACHE_DIR / "detections").glob(f"{key}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if sel == "oldest":
        cands = cands[::-1]
    elif sel != "newest":
        hit = [p for p in cands if p.stem.endswith(sel)]
        if not hit:
            print(f"   (no cache {sel} for {key}: falling back to the newest file)")
        cands = hit or cands
    for p in cands:
        data = json.loads(p.read_text())
        if len(data["frames"]) == len(frames):
            return [[Detection.from_list(r) for r in fr["dets"]] for fr in data["frames"]], p.stem.rsplit("_", 1)[-1]
    return None, None


def load(n):
    frames = extract_frames(index_videos()[n])
    dets, cfg = load_cached(frames, CACHE_SEL)
    assert dets is not None, f"no cached detections for X-{n} (cache={CACHE_SEL})"
    img = cv2.imread(str(frames.frames[0].path))
    h, w = img.shape[:2]
    persons = [[d for d in fr if d.label == "person" and d.score >= 0.35 and d.box[3] - d.box[1] >= 0.12 * h] for fr in dets]
    _, tracks = track_boxes(persons, w, h)
    layout = build_layout(dets, w, h, tracks)
    cm = StandInCashier(persons, layout, h)
    return frames, dets, layout, cm, ObjectModel(dets, persons, cm, layout, frames)


def fb(b):
    return "none" if b is None else "[" + ",".join(str(int(round(v))) for v in b) + "]"


def t_of(frames, i):
    return frames.frames[i].timestamp_sec


def idx(frames, t):
    return min(range(len(frames)), key=lambda i: abs(t_of(frames, i) - t))


def print_timeline(n, frames, om, t0, t1, labels=None, min_score=0.3, extra=None):
    print(f"--- X-{n} {t0}-{t1}s  (flags: B=background M=masked D=on_desk S=static C=in_cashier; age)")
    for i in range(len(frames)):
        t = t_of(frames, i)
        if not (t0 <= t <= t1):
            continue
        objs = [o for o in om.objects[i] if o.score >= min_score and (labels is None or o.label in labels)]
        s = " ".join(f"{o.label}:{o.score:.2f}{fb(o.box)}{'B' if o.background else ''}{'M' if o.masked else ''}"
                     f"{'D' if o.on_desk else ''}{'S' if o.static else ''}{'C' if o.in_cashier else ''}/{o.age}" for o in objs)
        ex = " ".join(f"{k}={'T' if v[i] else '.'}" for k, v in (extra or {}).items())
        print(f"{t:6.1f} cash={fb(om.cashier_box[i]):22s} cih={'T' if om.cash_in_hand[i] else '.'} hac={'T' if om.hand_at_counter[i] else '.'} "
              f"ao={om.anchor_overlap[i]:.2f} {ex} | {s}")


def sec_runs(frames, flags):
    return [(t_of(frames, a), t_of(frames, b - 1)) for a, b in runs(flags)]


def labeled_runs(frames, om, labels, thr, k):
    """(t_start, t_end, label, median box) of the confirmed new desk objects."""
    return [(t_of(frames, a), t_of(frames, b - 1), om.clusters[cids[0]].label, fb(om.object_runs(labels, thr, k)[j][2]))
            for j, (a, b, cids) in enumerate(om._cluster_runs(labels, thr, k))]


def frac_true(frames, flags, t0, t1):
    sel = [flags[i] for i in range(len(frames)) if t0 <= t_of(frames, i) <= t1]
    return sum(sel) / max(1, len(sel)), len(sel)


results = []


def check(name, ok, detail, hard=True):
    results.append((name, bool(ok), detail, hard))
    print(f"[{'PASS' if ok else ('FAIL' if hard else 'INFO')}] {name}: {detail}")


# ---------------------------------------------------------------------------------------------
def test_background():
    for n in (12, 10, 11, 25, 27, 29):
        frames, dets, lay, cm, om = load(n)
        cmach = lay.counter_machine
        bad = tot = 0
        for i in range(len(frames)):
            for o in om.objects[i]:
                if o.label == "money_bag" and cmach and contain_frac(o.box, cmach) >= 0.6:
                    tot += 1
                    bad += not (o.background or o.masked)
        check(f"X-{n} money_bag on the counting machine is background/masked", bad == 0, f"{tot - bad}/{tot} flagged")
    for n in (1, 3, 4, 5, 17, 18, 48):
        frames, dets, lay, cm, om = load(n)
        ref = (35, 109, 62, 123) if n in (17, 18) else (27, 140, 53, 154)   # spec pixel examples, camera A shots
        tot = bad = 0
        for i in range(len(frames)):
            for o in om.objects[i]:
                if o.label == "cash" and iou(o.box, ref) >= 0.4:
                    tot += 1
                    bad += not o.background
        check(f"X-{n} shelf cash behind the glass is background", tot > 0 and bad == 0, f"{tot - bad}/{tot} flagged")
    for n, phone in ((43, (55, 327, 82, 347)), (45, (72, 313, 99, 334))):
        frames, dets, lay, cm, om = load(n)
        tot = bad = 0
        for i in range(len(frames)):
            for o in om.objects[i]:
                if o.label in ("money_bag", "bag") and iou(o.box, phone) >= 0.3:
                    tot += 1
                    bad += not (o.background or o.masked)
        check(f"X-{n} bag/money_bag on the desk phone is background/masked", tot > 0 and bad == 0, f"{tot - bad}/{tot} flagged")


def test_new_desk_object():
    # X-17: T9 pouch placed at 7.0
    frames, dets, lay, cm, om = load(17)
    nd = om.new_desk_object(T9_LABELS, 0.30, 3)
    print_timeline(17, frames, om, 6.0, 14.5, T9_LABELS | {"cash"}, 0.25, {"nd": nd})
    rs = sec_runs(frames, nd)
    fr, _ = frac_true(frames, nd, 7.5, 99)
    check("X-17 new money_bag from ~7.0 to the end (thr 0.30, k 3)", rs and rs[0][0] <= 7.5 and fr >= 0.9,
          f"runs={rs} true-frac after 7.5s={fr:.2f} object_runs={[(t_of(frames, a), t_of(frames, b - 1), fb(m)) for a, b, m in om.object_runs(T9_LABELS, 0.30, 3)]}")
    # X-26: T1 cash placed 7.5-8.5, scores 0.30-0.32
    frames, dets, lay, cm, om = load(26)
    nd = om.new_desk_object(T1_LABELS, 0.30, 2)
    print_timeline(26, frames, om, 7.0, 19.5, T1_LABELS, 0.2, {"nd": nd})
    rs = sec_runs(frames, nd)
    fr, _ = frac_true(frames, nd, 9.5, 99)
    check("X-26 new cash on the desk from <= 9.5 s to the end (thr 0.30, k 2)", rs and rs[0][0] <= 9.5 and fr >= 0.9,
          f"runs={labeled_runs(frames, om, T1_LABELS, 0.30, 2)} true-frac after 9.5s={fr:.2f}")
    sc = [max([o.score for o in om.objects[i] if o.label == "cash" and o.on_desk and o.is_event_object] or [0]) for i in range(len(frames))]
    check("X-26 cash on the desk at thr 0.25 (report)", True, f"runs={labeled_runs(frames, om, {'cash'}, 0.25, 2)}; max on-desk cash score per frame 9-19.5s="
          f"{[round(sc[i], 2) for i in range(len(frames)) if 9 <= t_of(frames, i) <= 19.5]}", hard=False)
    # X-38: cash on the desk from 26.5 (detector sees it in few frames)
    frames, dets, lay, cm, om = load(38)
    nd = om.new_desk_object(T1_LABELS, 0.30, 2)
    rs = sec_runs(frames, nd)
    print_timeline(38, frames, om, 26.0, 30.0, {"cash"}, 0.2, {"nd": nd})
    print_timeline(38, frames, om, 49.0, 59.0, {"cash"}, 0.2, {"nd": nd})
    check("X-38 cash on the desk (report): any run in 45-59 s", any(a <= 59 and b >= 45 for a, b in rs), f"runs={labeled_runs(frames, om, T1_LABELS, 0.30, 2)}", hard=False)
    rs25 = labeled_runs(frames, om, {"cash"}, 0.25, 2)
    check("X-38 cash runs at thr 0.25 (report)", True, f"runs={rs25}", hard=False)
    # X-4: bundle on the desk from 44 s (weak)
    frames, dets, lay, cm, om = load(4)
    nd = om.new_desk_object(T1_LABELS, 0.30, 2)
    rs = [(a, b) for a, b in sec_runs(frames, nd) if b >= 40]
    check("X-4 bundle on the desk from ~44 s (report)", any(a <= 60 and b >= 59 for a, b in rs), f"runs after 40s={[r for r in labeled_runs(frames, om, T1_LABELS, 0.30, 2) if r[1] >= 40]}", hard=False)
    # X-46: cup (thr 0.40 enter / 0.30 hold)
    frames, dets, lay, cm, om = load(46)
    nd = om.new_desk_object({"cup"}, 0.40, 3)
    print_timeline(46, frames, om, 9.5, 18.5, {"cup", "cash"}, 0.3, {"nd": nd})
    rs = sec_runs(frames, nd)
    raw = [t_of(frames, i) for i in range(len(frames)) if any(o.label == "cup" and o.score >= 0.40 and o.is_event_object and (o.on_desk or o.in_cashier) for o in om.objects[i])]
    fr, _ = frac_true(frames, nd, rs[0][0] if rs else 99, 99)
    check("X-46 new cup from <= 12.5 s to the end (thr 0.40, k 3, hold 0.30)", rs and rs[0][0] <= 12.5 and fr >= 0.85,
          f"runs={rs} true-frac from start={fr:.2f}; raw cup>=0.40 on desk/in hand at {raw}")
    check("X-46 cup at thr 0.35 (report)", True, f"runs={labeled_runs(frames, om, {'cup'}, 0.35, 3)}", hard=False)
    # X-40: bag/money_bag on the far desk corner from 14.5
    frames, dets, lay, cm, om = load(40)
    nd = om.new_desk_object(T5_LABELS, 0.35, 3)
    print_timeline(40, frames, om, 13.5, 20.0, T5_LABELS, 0.3, {"nd": nd})
    rs = sec_runs(frames, nd)
    fr, _ = frac_true(frames, nd, 15.0, 18.5)
    check("X-40 new bag on the far desk corner from 14.5 s (thr 0.35, k 3)", rs and 14.0 <= rs[0][0] <= 15.5 and fr == 1.0,
          f"runs={rs} true-frac 15-18.5s={fr:.2f}")
    # clean videos: no new money_bag/bag
    for n in (10, 43, 45):
        frames, dets, lay, cm, om = load(n)
        nd = om.new_desk_object(T9_LABELS, 0.35, 3)
        check(f"X-{n} NO new_desk_object(money_bag|bag, 0.35, 3)", not any(nd), f"runs={sec_runs(frames, nd)}")
    # X-25: real cup + magazine present (not background), T6 context is the rules' job
    frames, dets, lay, cm, om = load(25)
    cup = sum(1 for i in range(len(frames)) for o in om.objects[i] if o.label == "cup" and o.score >= 0.30 and o.on_desk and o.is_event_object)
    book = sum(1 for i in range(len(frames)) for o in om.objects[i] if o.label == "book" and o.score >= 0.30 and o.on_desk and o.is_event_object)
    check("X-25 cup + magazine reported as real on-desk objects", cup >= 5 and book >= 3, f"cup frames={cup} book frames={book} "
          f"cup runs(0.30,2)={sec_runs(frames, om.new_desk_object({'cup'}, 0.30, 2))} book runs(0.30,2)={sec_runs(frames, om.new_desk_object({'book'}, 0.30, 2))}")


def test_worn():
    for n, t0, t1 in ((19, 0, 999), (35, 0, 999), (40, 1.5, 10.5)):
        frames, dets, lay, cm, om = load(n)
        persons = [[d for d in fr if d.label == "person" and d.score >= 0.35] for fr in dets]
        on_person = other = on_person_masked = 0
        for i in range(len(frames)):
            if not (t0 <= t_of(frames, i) <= t1):
                continue
            for o in om.objects[i]:
                if o.label == "jacket" and o.score >= 0.40:
                    if any(contain_frac(o.box, p.box) >= 0.5 for p in persons[i]):
                        on_person += 1
                        on_person_masked += o.masked
                    else:
                        other += 1
        check(f"X-{n} worn jackets masked ({t0}-{t1}s)", on_person > 0 and on_person_masked == on_person,
              f"jacket>=0.40 on a person: {on_person_masked}/{on_person} masked; not on a person (unmasked, off-desk): {other}")


def test_hands():
    frames, dets, lay, cm, om = load(1)
    print_timeline(1, frames, om, 19.5, 23.5, {"cash", "cup", "phone"}, 0.25)
    i22 = idx(frames, 22.0)
    fr, _ = frac_true(frames, om.cash_in_hand, 20.0, 22.5)
    check("X-1 cash_in_hand around 22.0 s", om.cash_in_hand[i22] and fr >= 0.6, f"t=22.0 -> {om.cash_in_hand[i22]}, true-frac 20-22.5s={fr:.2f}, runs={sec_runs(frames, om.cash_in_hand)}")
    frames, dets, lay, cm, om = load(13)
    print_timeline(13, frames, om, 44.5, 49.5, {"cash", "cup", "phone"}, 0.25)
    fr, _ = frac_true(frames, om.cash_in_hand, 45.0, 49.0)
    check("X-13 cash_in_hand mostly True in 45-49 s", fr >= 0.5, f"true-frac={fr:.2f}, runs={sec_runs(frames, om.cash_in_hand)}")
    frames, dets, lay, cm, om = load(10)
    fr, _ = frac_true(frames, om.cash_in_hand, 0, 999)
    check("X-10 cash_in_hand rate while counting at the desk (report)", True, f"true-frac={fr:.2f}, runs={sec_runs(frames, om.cash_in_hand)}", hard=False)
    hfr, _ = frac_true(frames, om.hand_at_counter, 0, 999)
    check("X-10 hand_at_counter True in most frames", hfr >= 0.7, f"true-frac={hfr:.2f}, runs={sec_runs(frames, om.hand_at_counter)}")
    frames, dets, lay, cm, om = load(34)
    print_timeline(34, frames, om, 11.5, 17.0, set(), 0.99)
    fr, _ = frac_true(frames, om.hand_at_counter, 12.5, 15.5)
    check("X-34 hand_at_counter True during 12.5-15.5 s", fr == 1.0, f"true-frac={fr:.2f}, runs={sec_runs(frames, om.hand_at_counter)}")
    frames, dets, lay, cm, om = load(26)
    fr, _ = frac_true(frames, om.hand_at_counter, 9.0, 999)
    check("X-26 hand_at_counter False after 9 s", fr == 0.0, f"true-frac after 9s={fr:.2f}, runs={sec_runs(frames, om.hand_at_counter)}")


def main():
    global CACHE_SEL
    args = sys.argv[1:]
    if "--cache" in args:
        k = args.index("--cache")
        CACHE_SEL = args[k + 1]
        del args[k:k + 2]
    print(f"detection cache: {CACHE_SEL}")
    if "--show" in args:
        for n in map(int, args[args.index("--show") + 1:]):
            frames, dets, lay, cm, om = load(n)
            print_timeline(n, frames, om, 0, 1e9)
        return
    test_background()
    test_new_desk_object()
    test_worn()
    test_hands()
    hard_fail = [r for r in results if r[3] and not r[1]]
    print(f"\n{len([r for r in results if r[1]])}/{len(results)} checks passed; hard failures: {len(hard_fail)}")
    for name, _, detail, _ in hard_fail:
        print("  FAILED:", name, "-", detail)
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
