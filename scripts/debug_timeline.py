"""Print the v2 scene timeline of one video for rule design.

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


def fb(b):
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
    tl = build_timeline(frames, cached_detections(frames))
    lay = tl.layout
    print(f"X-{args.n}: {frames.meta.duration_sec:.1f}s, {len(frames)} frames, label={label.violation if label else '?'} "
          f"t={label.t_sec if label else '?'} type={label.type_id if label else '?'}")
    print(f"layout: desk={fb(lay.desk)} laptop={fb(lay.laptop)} counter={fb(lay.counter_machine)} safe={fb(lay.safe)} "
          f"door={fb(lay.door)} seat={fb(lay.cashier_seat)} presence={ {k: round(v, 2) for k, v in lay.presence.items()} }")
    print(f"cashier track={tl.cashier_track_id} exit_t={tl.exit_t} entered={tl.entered} left={tl.left} "
          f"lights_off={tl.lights_off_events} lock={tl.lock_events} empty_start={tl.room_empty_start}")
    print(f"{'t':>6} {'br':>4} {'scr':>7} {'C':>3} {'vac':>3} {'att':>3} {'door':>4} {'cnt':>3} {'hnd':>3} {'cih':>3} {'room':>4}  "
          f"persons[id:flags(gap,seatIoU)] | desk objects (label score age flags) | in-hand")
    print("flags: C=cashier R=in_room d=at_desk F=far_side D=at_door | obj flags: B=background M=masked s=static")
    for st in tl.frames:
        if not (args.t0 <= st.t <= args.t1):
            continue
        ps = " ".join(f"{p.track_id}:{'C' if p.is_cashier else ''}{'R' if p.in_room else ''}{'d' if p.at_desk else ''}"
                      f"{'F' if p.far_side else ''}{'D' if p.at_door else ''}({p.desk_gap:.1f},{p.seat_iou:.2f})" for p in st.persons)
        on = " ".join(f"{o.label}:{o.score:.2f}/{o.age}{'B' if o.background else ''}{'M' if o.masked else ''}{'s' if o.static else ''}"
                      for o in st.objects if o.on_desk and o.score >= 0.3)
        hand = " ".join(f"{o.label}:{o.score:.2f}" for o in st.objects if o.in_cashier and not o.on_desk and o.score >= 0.3)
        print(f"{st.t:6.1f} {st.brightness:4.0f} {st.screen:>7}{'*' if st.lights_off else ' '}{'L' if st.lid_closed else ' '} "
              f"{'Y' if st.cashier_present else '-':>2} {'Y' if st.seat_vacated else '-':>3} {'Y' if st.attending_safe else '-':>3} "
              f"{st.safe_score:4.2f}{'O' if st.door_open else ' '} {'Y' if st.hand_at_counter else '-':>3} {st.hands_visible:3d} "
              f"{'Y' if st.cash_in_hand else '-':>3} {st.n_in_room:4d}  {ps:46s} | {on} | {hand}")


if __name__ == "__main__":
    main()
