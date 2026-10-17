"""Print the scene timeline of one video (layout + per-frame state) for rule design.

    .venv/bin/python scripts/debug_timeline.py 20 [--from 0 --to 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.labels import index_videos, load_labels  # noqa: E402
from src.pipeline.detect_cache import cached_detections  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402
from src.pipeline.state import build_timeline  # noqa: E402


def fmt_box(b):
    return "none" if b is None else "(" + ",".join(str(int(v)) for v in b) + ")"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--from", dest="t0", type=float, default=0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    args = ap.parse_args()
    videos = index_videos()
    label = next((l for l in load_labels() if l.n == args.n), None)
    frames = extract_frames(videos[args.n])
    dets = cached_detections(frames)
    tl = build_timeline(frames, dets)
    lay = tl.layout
    print(f"X-{args.n}: {frames.meta.duration_sec:.1f}s, {len(frames)} frames, label={label.violation if label else '?'} "
          f"t={label.t_sec if label else '?'} type={label.type_id if label else '?'}")
    print(f"layout: desk={fmt_box(lay.desk)} laptop={fmt_box(lay.laptop)} counter={fmt_box(lay.counter_machine)} "
          f"safe={fmt_box(lay.safe)} door={fmt_box(lay.door)} seat={fmt_box(lay.cashier_seat)}")
    print(f"zones:  cashier={fmt_box(lay.cashier_zone)} near_desk={fmt_box(lay.near_desk_zone)} "
          f"safe={fmt_box(lay.safe_zone)} door={fmt_box(lay.door_zone)} presence={ {k: round(v, 2) for k, v in lay.presence.items()} }")
    print(f"cashier track id: {tl.cashier_track_id()}")
    print(f"{'t':>6} {'br':>4} {'scr':>4} {'safeO':>5} {'hnd':>3}  persons[id:flags(anchor,gap)]  | objects on desk (label score) | off-desk")
    print("flags: C=cashier s=at seat d=near desk D=at door S=at safe")
    for st in tl.frames:
        if not (args.t0 <= st.t <= args.t1):
            continue
        ps = " ".join(f"{p.track_id}:{'C' if p.is_cashier else ''}{'s' if p.at_seat else ''}{'d' if p.near_desk else ''}"
                      f"{'D' if p.at_door else ''}{'S' if p.at_safe else ''}({p.anchor_overlap:.1f},{p.desk_gap:.1f})" for p in st.persons)
        on = " ".join(f"{o.label}:{o.score:.2f}" for o in st.objects if o.on_desk and o.score >= 0.3)
        off = " ".join(f"{o.label}:{o.score:.2f}" for o in st.objects if not o.on_desk and o.score >= 0.4)
        scr = ("ON" if st.screen_on else "off") + ("*" if st.lights_off else "")
        print(f"{st.t:6.1f} {st.brightness:4.0f} {scr:>4} {st.safe_door_open:5.2f} {st.hands_below_desk:3d}  {ps:44s} | {on} | {off}")


if __name__ == "__main__":
    main()
