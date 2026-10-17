"""Self-test of src/pipeline/screen.py (P6 screen state, P7 lights-off, lock events) on cached data.

    .venv/bin/python scripts/selftest_screen.py            # the spec's check videos, asserts the expectations
    .venv/bin/python scripts/selftest_screen.py --all      # plus a one-line summary for every cached video

Offline only: cached frames + cached detections (never runs the detector). Other state modules may still be
stubs, so stand-ins are used: persons = person boxes with score >= 0.35 and height >= 0.12 H; cashier present
= a person box overlapping layout.anchor by >= 15% of the anchor area.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.labels import index_videos  # noqa: E402
from src.pipeline import screen  # noqa: E402
from src.pipeline.detect_cache import load_any_cached  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402
from src.pipeline.track import track_boxes  # noqa: E402
from src.pipeline.zones import area, build_layout, inter_area  # noqa: E402


class CashierStandIn:
    """Stand-in for people.CashierModel: only `.present[i]` is needed by screen.lock_events."""

    def __init__(self, persons, layout):
        anchor = layout.anchor
        self.present = [bool(anchor) and any(inter_area(d.box, anchor) / area(anchor) >= 0.15 for d in ps)
                        for ps in persons]


def analyse(n: int):
    frames = extract_frames(index_videos()[n])
    dets = load_any_cached(frames)
    if dets is None:
        raise SystemExit(f"X-{n}: no cached detections")
    import cv2

    h, w = cv2.imread(str(frames.frames[0].path)).shape[:2]
    persons = [[d for d in ds if d.label == "person" and d.score >= 0.35 and d.box[3] - d.box[1] >= 0.12 * h]
               for ds in dets]
    _, tracks = track_boxes(persons, w, h)
    layout = build_layout(dets, w, h, tracks)
    cm = CashierStandIn(persons, layout)
    states = screen.screen_states(frames, layout, persons, dets)
    fps = frames.sample_fps
    lights = [i / fps for i in screen.lights_off_events(states)]
    locks = [i / fps for i in screen.lock_events(states, cm, fps)]
    return frames, layout, states, lights, locks, cm


def fmt(st, t):
    return (f"  {t:5.1f} br={st.brightness:5.1f} scr={st.screen:7s} lid={'C' if st.lid_closed else 'o'} "
            f"L={'*' if st.lights_off else ' '} mean={st.laptop_mean:6.1f} white={st.white_frac:.2f} "
            f"wfull={st.white_full:.2f} occ={'Y' if st.occluded else 'n'}")


def show(n, frames, layout, states, lights, locks, t0, t1, step=1):
    lp = layout.laptop
    print(f"== X-{n}: {len(frames)} frames, laptop={None if lp is None else tuple(int(v) for v in lp)} "
          f"presence={layout.presence.get('laptop', 0):.2f} lights_off={lights} lock={locks}")
    for st, f in zip(states, frames.frames):
        if t0 <= f.timestamp_sec <= t1 and int(round(f.timestamp_sec * frames.sample_fps)) % step == 0:
            print(fmt(st, f.timestamp_sec))


results: list[tuple[str, bool, bool]] = []   # (name, passed, xfail)


def check(name: str, cond: bool, detail: str = "", xfail: bool = False) -> None:
    tag = "PASS" if cond else ("XFAIL (by design)" if xfail else "FAIL")
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    results.append((name, cond, xfail))


def frac(states, frames, t0, t1, pred) -> float:
    sel = [pred(s) for s, f in zip(states, frames.frames) if t0 <= f.timestamp_sec <= t1]
    return sum(sel) / max(1, len(sel))


def rng(states, frames, t0, t1, attr) -> tuple[float, float]:
    vals = [getattr(s, attr) for s, f in zip(states, frames.frames) if t0 <= f.timestamp_sec <= t1]
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return (min(vals), max(vals)) if vals else (math.nan, math.nan)


def main() -> None:
    V = {}
    for n in (11, 12, 28, 22, 23, 29, 27, 19, 45, 34, 38, 16, 30):
        V[n] = analyse(n)

    # ---- X-11: lights off at 16.0, screen active (white in the whole box 0.15-0.17) after it -> T2 cue
    fr, lay, st, li, lo, _ = V[11]
    show(11, fr, lay, st, li, lo, 14.0, 25.0)
    check("X-11 lights_off event at 16.0 s", li == [16.0], f"events={li}")
    b = [s.brightness for s in st]
    check("X-11 brightness 139 -> 45 then ~92", 135 <= b[31] <= 145 and b[32] <= 60 and 85 <= b[40] <= 100,
          f"15.5s={b[31]:.0f} 16.0s={b[32]:.0f} 20.0s={b[40]:.0f}")
    check("X-11 lights_off flag from 16.0 s on", all(s.lights_off for s in st[32:]) and not any(s.lights_off for s in st[:32]))
    a = frac(st, fr, 17.0, 25.5, lambda s: s.screen == "active")
    check("X-11 screen 'active' after lights off (t >= 17 s)", a >= 0.95, f"active frac={a:.2f}")
    wf = rng(st, fr, 17.0, 25.5, "white_full")
    check("X-11 white_full 0.13-0.19 after lights off", 0.13 <= wf[0] and wf[1] <= 0.19, f"white_full={wf[0]:.2f}-{wf[1]:.2f}")
    check("X-11 no lock events (the lights-off drop is not a lock)", lo == [], f"lock={lo}")

    # ---- X-12: exposure spike 154 at 32.0 is not an event; lights off at 32.5; screen off; lid closed from ~11 s
    fr, lay, st, li, lo, _ = V[12]
    show(12, fr, lay, st, li, lo, 9.0, 13.0)
    show(12, fr, lay, st, li, lo, 31.0, 36.0)
    check("X-12 lights_off event at 32.5 s only (not at the 32.0 s spike)", li == [32.5], f"events={li}")
    b = [s.brightness for s in st]
    check("X-12 brightness 139 -> 154 spike -> 75", 150 <= b[64] <= 158 and 70 <= b[65] <= 80, f"32.0s={b[64]:.0f} 32.5s={b[65]:.0f}")
    o = frac(st, fr, 33.5, 42.5, lambda s: s.screen == "off")
    check("X-12 screen 'off' after lights off (t >= 33.5 s)", o >= 0.95, f"off frac={o:.2f}")
    wf = rng(st, fr, 33.5, 42.5, "white_full")
    check("X-12 white_full <= 0.03 after lights off", wf[1] <= 0.03, f"white_full={wf[0]:.2f}-{wf[1]:.2f}")
    c1 = frac(st, fr, 12.0, 42.5, lambda s: s.lid_closed)
    c0 = frac(st, fr, 0.0, 10.5, lambda s: s.lid_closed)
    check("X-12 lid_closed True from 12 s on, False before 11 s", c1 >= 0.98 and c0 == 0.0, f"after={c1:.2f} before={c0:.2f}")
    o2 = frac(st, fr, 12.0, 31.5, lambda s: s.screen == "off")
    check("X-12 screen 'off' while the lid is closed (12-31.5 s, lights on)", o2 >= 0.95, f"off frac={o2:.2f}")
    check("X-12 no lock events", lo == [], f"lock={lo}")

    # ---- X-28: bluish-white active screen during 7-10 s
    fr, lay, st, li, lo, _ = V[28]
    show(28, fr, lay, st, li, lo, 6.0, 10.5)
    a = frac(st, fr, 7.0, 10.0, lambda s: s.screen == "active")
    m = rng(st, fr, 7.0, 10.0, "laptop_mean")
    check("X-28 screen 'active' during 7-10 s", a == 1.0, f"active frac={a:.2f} mean={m[0]:.0f}-{m[1]:.0f}")
    check("X-28 no lights_off / lock events", li == [] and lo == [], f"lights={li} lock={lo}")

    # ---- X-22: dim or active (mean 115-127) after 9 s, lid open; lock-like drop at 5.0 (144->96)
    fr, lay, st, li, lo, _ = V[22]
    show(22, fr, lay, st, li, lo, 4.0, 21.0)
    a = frac(st, fr, 9.0, 21.0, lambda s: s.screen in ("dim", "active"))
    m = rng(st, fr, 9.0, 21.0, "laptop_mean")
    check("X-22 screen 'dim'/'active' after 9 s", a >= 0.95, f"lit frac={a:.2f}")
    check("X-22 region mean 115-127 after 9 s", 115 <= m[0] and m[1] <= 127, f"mean={m[0]:.1f}-{m[1]:.1f}")
    check("X-22 lid open throughout", frac(st, fr, 0, 21, lambda s: s.lid_closed) == 0.0)
    check("X-22 lock event at 5.0 s (144->96)", any(4.5 <= t <= 5.5 for t in lo), f"lock={lo}")

    # ---- X-23: lock at ~4.5 s (145->106); afterwards the lock screen 105-115
    fr, lay, st, li, lo, _ = V[23]
    show(23, fr, lay, st, li, lo, 3.0, 8.0)
    check("X-23 lock event at ~4.5 s", any(4.0 <= t <= 5.0 for t in lo), f"lock={lo}")
    m = rng(st, fr, 6.0, 18.5, "laptop_mean")
    check("X-23 region mean 100-115 after the lock", 100 <= m[0] and m[1] <= 115, f"mean={m[0]:.1f}-{m[1]:.1f}")
    d = frac(st, fr, 6.0, 18.5, lambda s: s.screen == "dim")
    o = frac(st, fr, 6.0, 18.5, lambda s: s.screen == "off")
    check("X-23 screen 'dim' after the lock (spec P6 literal)", d >= 0.95, f"dim frac={d:.2f} off frac={o:.2f}", xfail=True)
    check("X-23 screen 'off' after the lock (lock screen, lit threshold 115 of spec T2)", o >= 0.95, f"off frac={o:.2f}")

    # ---- X-29: lock at 5.5-6.0 (154->82) then off
    fr, lay, st, li, lo, _ = V[29]
    show(29, fr, lay, st, li, lo, 4.5, 9.0)
    check("X-29 lock event at 5.5-6.0 s", any(5.5 <= t <= 6.0 for t in lo), f"lock={lo}")
    o = frac(st, fr, 7.0, 17.5, lambda s: s.screen == "off")
    m = rng(st, fr, 7.0, 17.5, "laptop_mean")
    check("X-29 screen 'off' after the lock", o >= 0.95, f"off frac={o:.2f} mean={m[0]:.0f}-{m[1]:.0f}")

    # ---- X-27, X-34, X-38: off; X-45: unknown; X-19: active most of 26-44 s despite flicker
    for n, t1 in ((27, 26.5), (34, 17.5), (38, 59.0)):
        fr, lay, st, li, lo, _ = V[n]
        show(n, fr, lay, st, li, lo, 0.0, t1, step=8)
        o = frac(st, fr, 0.0, t1, lambda s: s.screen == "off")
        m = rng(st, fr, 0.0, t1, "laptop_mean")
        check(f"X-{n} screen 'off' throughout", o >= 0.95, f"off frac={o:.2f} mean={m[0]:.0f}-{m[1]:.0f}")
        check(f"X-{n} no lights_off / lock events", li == [] and lo == [], f"lights={li} lock={lo}")
    fr, lay, st, li, lo, _ = V[45]
    show(45, fr, lay, st, li, lo, 0.0, 33.5, step=12)
    u = frac(st, fr, 0.0, 33.5, lambda s: s.screen == "unknown")
    check("X-45 screen 'unknown' (laptop presence < 0.7)", u == 1.0, f"unknown frac={u:.2f} presence={lay.presence.get('laptop'):.2f}")
    fr, lay, st, li, lo, _ = V[19]
    show(19, fr, lay, st, li, lo, 34.0, 44.5)
    a = frac(st, fr, 26.0, 44.0, lambda s: s.screen == "active")
    a2 = frac(st, fr, 34.5, 44.0, lambda s: s.screen == "active")
    check("X-19 screen 'active' most of 26-44 s", a >= 0.9, f"active frac 26-44={a:.2f}, 34.5-44={a2:.2f}")
    check("X-19 no lights_off events (exposure 157-160)", li == [], f"lights={li}")
    check("X-19 no lock events (flicker)", lo == [], f"lock={lo}")
    for n in (16, 30):
        fr, lay, st, li, lo, _ = V[n]
        b = rng(st, fr, 0, 1e9, "brightness")
        check(f"X-{n} no lights_off events (exposure spike)", li == [], f"brightness {b[0]:.0f}-{b[1]:.0f} events={li}")

    if "--all" in sys.argv:
        print("\n== all cached videos: class distribution / events")
        for n in sorted(index_videos()):
            try:
                fr, lay, st, li, lo, _ = analyse(n)
            except SystemExit as e:
                print(f"  X-{n}: {e}")
                continue
            cnt = {c: sum(1 for s in st if s.screen == c) for c in ("active", "dim", "off", "unknown")}
            m = rng(st, fr, 0, 1e9, "laptop_mean")
            print(f"  X-{n:<3d} n={len(st):3d} presence={lay.presence.get('laptop', 0):.2f} {cnt} mean={m[0]:.0f}-{m[1]:.0f} "
                  f"lid_closed={sum(s.lid_closed for s in st):3d} lights_off={li} lock={lo}")

    failed = [r for r in results if not r[1] and not r[2]]
    xfailed = [r for r in results if not r[1] and r[2]]
    print(f"\n{len(results) - len(failed) - len(xfailed)} passed, {len(failed)} failed, {len(xfailed)} xfail")
    for name, _, _ in failed:
        print("  FAILED:", name)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
