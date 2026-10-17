"""Self-test of src/pipeline/blob.py (desk_blob, spec P5) on cached frames + detections; never runs the detector.

    .venv/bin/python scripts/selftest_blob.py                 # positives (X-4/38/26/15) + clean controls
    .venv/bin/python scripts/selftest_blob.py --videos 4 38   # a subset
    .venv/bin/python scripts/selftest_blob.py --all           # all 42 videos
    .venv/bin/python scripts/selftest_blob.py --show 26 8 16  # per-frame candidates of X-26 between 8 and 16 s

Prints, per video: runtime, thresholds, t-ranges of stable new blobs with boxes; for the positives whether a blob
exists at the label time and in the preceding seconds; for the clean controls whether a blob exists while the
cashier is away (S.away) or while a visitor enters with nobody else at the desk. Annotated frames go to
logs/debug/blob/X-<n>_<t>.jpg.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.labels import index_videos, load_labels  # noqa: E402
from src.common import LOGS_DIR  # noqa: E402
from src.pipeline.blob import DeskBlobModel  # noqa: E402
from src.pipeline.detect_cache import cached_detections  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402
from src.pipeline.rules import Series  # noqa: E402
from src.pipeline.state import build_timeline, runs  # noqa: E402

POSITIVES = {4: (57.0, 44.0), 38: (33.0, 26.5), 26: (9.0, 8.5), 15: (15.0, 6.0)}   # n -> (label t, bundle from t)
CONTROLS = [5, 48, 10, 12, 14, 19, 23, 25, 27, 29, 43, 45]
OUT = LOGS_DIR / "debug" / "blob"


def fb(b):
    return "(" + ",".join(str(int(round(v))) for v in b) + ")"


def load(n: int):
    import cv2

    cv2.setNumThreads(2)
    fs = extract_frames(index_videos()[n])
    tl = build_timeline(fs, cached_detections(fs))
    S = Series(tl)
    t0 = time.perf_counter()
    m = DeskBlobModel(fs, tl)
    dt = time.perf_counter() - t0
    return fs, tl, S, m, dt


def save_frame(fs, tl, m, i: int, n: int, tag: str = "") -> None:
    import cv2

    OUT.mkdir(parents=True, exist_ok=True)
    img = fs.frames[i].load()
    d = tl.layout.desk
    if d:
        cv2.rectangle(img, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (200, 200, 0), 1)
    for p in tl.frames[i].persons:
        b = p.box
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0) if p.is_cashier else (255, 0, 0), 1)
    for box, info in m.cands[i]:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 165, 255), 1)
    for box in m.boxes[i]:
        cv2.rectangle(img, (int(box[0]) - 1, int(box[1]) - 1), (int(box[2]) + 1, int(box[3]) + 1), (0, 0, 255), 2)
    t = fs.frames[i].timestamp_sec
    cv2.putText(img, f"X-{n} t={t:.1f} {tag}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(OUT / f"X-{n}_{t:05.1f}{tag}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def describe(n: int, fs, tl, S, m, dt: float, save: bool = True) -> dict:
    t = m.t
    stable = [bool(b) for b in m.boxes]
    print(f"X-{n}: {fs.meta.duration_sec:.0f}s {len(fs)} frames  blob stage {dt:.2f}s  {m.stats}")
    rs = runs(stable)
    for a, b in rs:
        boxes = {fb(bx) for i in range(a, b) for bx in m.boxes[i]}
        print(f"    blob {t[a]:5.1f}-{t[b - 1]:5.1f}s ({b - a} fr) boxes {' '.join(sorted(boxes))[:120]}")
    if not rs:
        print("    no stable blobs")
    res = {"runs": [(t[a], t[b - 1]) for a, b in rs]}
    if n in POSITIVES:
        tl_, t_from = POSITIVES[n]
        i_lab = min(range(len(t)), key=lambda i: abs(t[i] - tl_))
        pre = [stable[i] for i in range(len(t)) if t_from <= t[i] < tl_]
        print(f"    label {tl_}s: blob={stable[i_lab]} boxes={[fb(b) for b in m.boxes[i_lab]]}; "
              f"preceding {t_from}-{tl_}s: {sum(pre)}/{len(pre)} frames")
        res["at_label"], res["pre_frac"] = stable[i_lab], (sum(pre) / len(pre) if pre else 0.0)
        if save:
            for ti in (t_from, (t_from + tl_) / 2, tl_, min(t[-1], tl_ + 3)):
                save_frame(fs, tl, m, fs.nearest(ti).index, n)
    else:
        away = [s and a for s, a in zip(stable, S.away)]
        enters = []
        for t_in, tid in tl.entered:
            i = min(range(len(t)), key=lambda i: abs(t[i] - t_in))
            win = range(max(0, i - 4), min(len(t), i + 5))
            hit = [j for j in win if stable[j] and S.others_at_desk[j] == 0]
            enters.append((t_in, bool(hit)))
        away_runs = [(t[a], t[b - 1]) for a, b in runs(away)]
        bad = bool(away_runs) or any(h for _, h in enters)
        print(f"    {'BLOB while unattended' if bad else 'ok'}: blob&away runs {away_runs}; enters (t, blob&nobody at desk) {enters}")
        res["bad"] = bad
        if save:
            for a, b in rs[:3]:
                save_frame(fs, tl, m, a, n)
            for a, b in runs(away)[:2]:
                save_frame(fs, tl, m, a, n, "_away")
    return res


def show(n: int, t0: float, t1: float) -> None:
    fs, tl, S, m, dt = load(n)
    print(f"X-{n}: {m.stats} blob stage {dt:.2f}s; desk={fb(tl.layout.desk)}")
    for i, t in enumerate(m.t):
        if not (t0 <= t <= t1):
            continue
        ps = " ".join(fb(p.box) + ("c" if p.is_cashier else "") for p in tl.frames[i].persons)
        c = "; ".join(f"{fb(b)} a{v['area']} c{v['contrast']} r{v['ring_changed']} s{v['static_d']} d{v['mean_dist']}"
                      for b, v in m.cands[i])
        st = " ".join(fb(b) for b in m.boxes[i])
        print(f"{t:5.1f} away={int(S.away[i])} oth={S.others_at_desk[i]} stable=[{st}] cands: {c} | persons {ps}")
        save_frame(fs, tl, m, i, n, "_show")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--show", nargs=3, metavar=("N", "T0", "T1"))
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    if args.show:
        show(int(args.show[0]), float(args.show[1]), float(args.show[2]))
        return
    if args.all:
        videos = sorted(index_videos())
    elif args.videos:
        videos = args.videos
    else:
        videos = list(POSITIVES) + CONTROLS
    labels = {l.n: l for l in load_labels()}
    summary = []
    for n in videos:
        fs, tl, S, m, dt = load(n)
        lab = labels.get(n)
        tag = f"label={'viol ' + str(lab.t_sec) + ' ' + str(lab.type_id) if lab and lab.violation else 'clean'}"
        print(f"--- {tag}")
        r = describe(n, fs, tl, S, m, dt, save=not args.no_save)
        r["clean"] = lab is not None and not lab.violation
        summary.append((n, dt, len(fs), r))
    print("\nSUMMARY")
    for n, dt, nf, r in summary:
        if "at_label" in r:
            print(f"  X-{n:<3} {nf:4d} fr {dt:5.2f}s  POSITIVE at_label={r['at_label']} pre={r['pre_frac']:.0%}")
        elif "bad" in r:
            kind = "CLEAN control" if r["clean"] else "other-type violation"
            print(f"  X-{n:<3} {nf:4d} fr {dt:5.2f}s  {kind} {'BLOB while unattended' if r['bad'] else 'ok'} runs={len(r['runs'])}")
        else:
            print(f"  X-{n:<3} {nf:4d} fr {dt:5.2f}s  runs={r['runs']}")


if __name__ == "__main__":
    main()
