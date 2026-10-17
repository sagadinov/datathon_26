"""Offline self-test of the pose-based T7 rule (src/pipeline/pose.py + src/pipeline/rules_t7.py).

Cached frames + cached detections only (never runs the detector); the pose model does run (CPU, ~15 ms/frame).

    .venv/bin/python scripts/selftest_pose.py                 # positives + controls + extra clean videos
    .venv/bin/python scripts/selftest_pose.py 1 2 44 47       # chosen videos
    .venv/bin/python scripts/selftest_pose.py --show 1        # also print the per-frame wrist table

Prints per video: the T7 events (time, variant, view), the wrist statistics (frames with a cashier box, mean
wrist confidence, out-of-sight runs, reach frames) and the pose runtime; then a summary against the ground
truth below. Exit code 1 when a control fires.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.labels import index_videos  # noqa: E402
from src.pipeline.detect_cache import cached_detections  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402
from src.pipeline.pose import load_pose, wrists_series  # noqa: E402
from src.pipeline.rules import Series  # noqa: E402
from src.pipeline.rules_t7 import out_of_sight, pose_features, rule_t7_pose, view_from_behind  # noqa: E402
from src.pipeline.state import build_timeline, runs  # noqa: E402

T7_TRUTH = {1: 22.0, 2: 10.0, 44: 16.0, 47: 68.0, 13: 47.0}       # X-13 is low-confidence (0.4)
CONTROLS = [3, 10, 14, 20, 21, 35, 43, 45]                            # must stay silent
EXTRA_CLEAN = [5, 8, 19, 23, 25, 27, 39]


def fmt_runs(mask, t):
    return " ".join(f"{t[a]:.1f}-{t[b - 1]:.1f}" for a, b in runs(mask)) or "-"


def run_video(n: int, path: Path, model, show: bool) -> tuple[list, float]:
    fs = extract_frames(path)
    tl = build_timeline(fs, cached_detections(fs))
    S = Series(tl)
    t0 = time.perf_counter()
    W = wrists_series(fs, tl, model=model)
    t_pose = time.perf_counter() - t0
    events = rule_t7_pose(S, W)
    feats = pose_features(S, W)
    behind = view_from_behind(S)
    lap = out_of_sight(S, feats, behind)
    seen = [w["box"] is not None for w in W]
    confs = [c for c, s in zip(feats["conf"], seen) if s]
    hidden = [0.0 < c < 0.6 for c in feats["conf"]]
    print(f"=== X-{n}: {len(fs)} frames ({len(fs) / fs.sample_fps:.0f} s), view={'behind' if behind else 'front' if behind is False else '?'}, "
          f"pose {t_pose:.2f}s ({1000 * t_pose / max(1, len(fs)):.1f} ms/frame)")
    print(f"    cashier box in {sum(seen)}/{len(fs)} frames, mean wrist conf {sum(confs) / max(1, len(confs)):.2f}, "
          f"wrists hidden (<0.6): {fmt_runs(hidden, S.t)}")
    print(f"    out of sight (rule): {fmt_runs(lap, S.t)} | reach: {fmt_runs(feats['reach'], S.t)}")
    print(f"    cash_in_hand: {fmt_runs(S.cash_in_hand, S.t)} | cash_desk: {fmt_runs(S.cash_desk, S.t)} | counting: {fmt_runs(S.counting, S.t)}")
    for ev in events:
        print(f"    T7 EVENT t={ev.t_start:.1f}-{ev.t_end:.1f} score={ev.score} {ev.details}")
    if not events:
        print("    no T7 events")
    if show:
        print("     t   cih cd cnt | lw(y,c)     rw(y,c)    | elb_rel wr_rel | lap reach")
        for i, t in enumerate(S.t):
            w = W[i]
            print(f"    {t:5.1f}  {int(S.cash_in_hand[i])}   {int(S.cash_desk[i])}  {int(S.counting[i])}  | "
                  f"{w['lw'][1]:5.0f},{w['lw'][2]:.2f} {w['rw'][1]:5.0f},{w['rw'][2]:.2f} | "
                  f"{feats['elb_rel'][i]:6.2f} {feats['wr_rel'][i]:6.2f} | {int(lap[i])}   {int(feats['reach'][i])}")
    return events, t_pose


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show" in sys.argv
    vids = [int(a) for a in args] if args else list(T7_TRUTH) + CONTROLS + EXTRA_CLEAN
    idx = index_videos()
    model = load_pose(threads=32)
    results: dict[int, list] = {}
    for n in vids:
        results[n], _ = run_video(n, idx[n], model, show)
    print("\n=== summary")
    bad = 0
    for n, evs in results.items():
        ts = [e.t_start for e in evs]
        if n in T7_TRUTH:
            truth = T7_TRUTH[n]
            d = min((abs(t - truth) for t in ts), default=None)
            print(f"  X-{n:<3} positive (label {truth:>5.1f}s): {'HIT' if d is not None and d <= 5 else 'MISS'}  events={[round(t, 1) for t in ts]}"
                  + (f" |dt|={d:.1f}s" if d is not None else ""))
        else:
            print(f"  X-{n:<3} clean: {'FP ' + str([round(t, 1) for t in ts]) if ts else 'silent'}")
            bad += bool(ts)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
