"""Independent verification of src/pipeline/screen.py (P6/P7, lock events) on cached data only.

    .venv/bin/python scripts/verify_screen.py [--montage DIR]

Adversarial checks on videos the module's own self-test does not cover (spec sections 1/5: X-1, X-3, X-48,
X-19 before the lid opens, X-24, X-25, X-13, X-20, X-46, X-21, X-10, X-26, X-15, X-14, X-35/37, no-laptop videos),
plus robustness (empty FrameSet, missing laptop/desk/safe, unreadable frame, short persons list, hflip
invariance), signatures used by state.py and runtime. Never runs the detector.
"""
from __future__ import annotations

import dataclasses
import inspect
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.labels import index_videos  # noqa: E402
from src.pipeline import screen, state  # noqa: E402
from src.pipeline.detect import Detection  # noqa: E402
from src.pipeline.detect_cache import load_any_cached  # noqa: E402
from src.pipeline.frames import extract_frames  # noqa: E402
from src.pipeline.screen_util import measure_frame  # noqa: E402
from src.pipeline.track import track_boxes  # noqa: E402
from src.pipeline.zones import SceneLayout, area, build_layout, inter_area  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    results.append((name, bool(cond), detail))


class CashierStandIn:
    def __init__(self, persons, layout):
        anchor = layout.anchor
        self.present = [bool(anchor) and any(inter_area(d.box, anchor) / area(anchor) >= 0.15 for d in ps)
                        for ps in persons]


def load(n: int):
    frames = extract_frames(index_videos()[n])
    dets = load_any_cached(frames)
    if dets is None:
        raise SystemExit(f"X-{n}: no cached detections")
    h, w = cv2.imread(str(frames.frames[0].path)).shape[:2]
    persons = [[d for d in ds if d.label == "person" and d.score >= 0.35 and d.box[3] - d.box[1] >= 0.12 * h]
               for ds in dets]
    _, tracks = track_boxes(persons, w, h)
    layout = build_layout(dets, w, h, tracks)
    return frames, dets, persons, layout, w, h


def analyse(n: int):
    frames, dets, persons, layout, w, h = load(n)
    cm = CashierStandIn(persons, layout)
    t0 = time.perf_counter()
    states = screen.screen_states(frames, layout, persons, dets)
    dt = time.perf_counter() - t0
    fps = frames.sample_fps
    lights = [i / fps for i in screen.lights_off_events(states)]
    locks = [i / fps for i in screen.lock_events(states, cm, fps)]
    return dict(frames=frames, dets=dets, persons=persons, layout=layout, states=states, lights=lights,
                locks=locks, cm=cm, dt=dt, w=w, h=h)


def frac(v, t0, t1, pred) -> float:
    sel = [pred(s) for s, f in zip(v["states"], v["frames"].frames) if t0 <= f.timestamp_sec <= t1]
    return sum(sel) / max(1, len(sel))


def rng(v, t0, t1, attr):
    vals = [getattr(s, attr) for s, f in zip(v["states"], v["frames"].frames) if t0 <= f.timestamp_sec <= t1]
    vals = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    return (min(vals), max(vals)) if vals else (math.nan, math.nan)


def dist(v, t0=0.0, t1=1e9) -> dict:
    return {c: sum(1 for s, f in zip(v["states"], v["frames"].frames) if t0 <= f.timestamp_sec <= t1 and s.screen == c)
            for c in ("active", "dim", "off", "unknown")}


def line(v, t) -> str:
    i = int(round(t * v["frames"].sample_fps))
    s = v["states"][i]
    return (f"    {t:5.1f} br={s.brightness:5.1f} scr={s.screen:7s} lid={'C' if s.lid_closed else 'o'} "
            f"L={'*' if s.lights_off else ' '} mean={s.laptop_mean:6.1f} white={s.white_frac:.2f} "
            f"wfull={s.white_full:.2f} occ={'Y' if s.occluded else 'n'}")


def header(n, v):
    lp = v["layout"].laptop
    print(f"== X-{n}: {len(v['frames'])} frames ({v['frames'].frames[-1].timestamp_sec:.1f} s), "
          f"laptop={None if lp is None else tuple(int(x) for x in lp)} presence={v['layout'].presence.get('laptop', 0):.2f} "
          f"dist={dist(v)} lid_closed={sum(s.lid_closed for s in v['states'])} lights_off={v['lights']} "
          f"lock={v['locks']} screen_states {v['dt'] * 1000:.0f} ms")


def montage(n, v, times, out_dir: Path) -> None:
    """Laptop crops (x4) at the given seconds, annotated with the module's class, for eyeballing."""
    lp = v["layout"].laptop
    if lp is None:
        return
    tiles = []
    for t in times:
        i = int(round(t * v["frames"].sample_fps))
        if i >= len(v["states"]):
            continue
        img = cv2.imread(str(v["frames"].frames[i].path))
        x1, y1, x2, y2 = lp
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - 0.4 * bw)), max(0, int(y1 - 0.4 * bh))
        cx2, cy2 = min(img.shape[1], int(x2 + 0.4 * bw)), min(img.shape[0], int(y2 + 0.4 * bh))
        crop = img[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (0, 255, 0), 1)
        crop = cv2.resize(crop, (int(crop.shape[1] * 3), int(crop.shape[0] * 3)), interpolation=cv2.INTER_CUBIC)
        s = v["states"][i]
        cv2.putText(crop, f"{t:.1f}s {s.screen}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(crop, f"m={s.laptop_mean:.0f} w={s.white_frac:.2f}", (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1)
        tiles.append(crop)
    if not tiles:
        return
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
    rows = [np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)]
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for r in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"verify_screen_X-{n}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def partial_occlusion_report(n, v) -> None:
    """Frames where a person covers 30-80% of the screen region are classified from pixels: do they disagree
    with the clean neighbours?"""
    lp = v["layout"].laptop
    if lp is None:
        return
    flips = []
    for i, (s, f) in enumerate(zip(v["states"], v["frames"].frames)):
        m = measure_frame(f.path, lp, v["persons"][i])
        if 0.3 <= m.occ < 0.8:
            nb = [v["states"][j].screen for j in (i - 3, i + 3) if 0 <= j < len(v["states"]) and not v["states"][j].occluded]
            if nb and all(c != s.screen for c in nb):
                flips.append((f.timestamp_sec, round(m.occ, 2), s.screen, nb))
    cnt = sum(1 for i, f in enumerate(v["frames"].frames)
              if 0.3 <= measure_frame(f.path, lp, v["persons"][i]).occ < 0.8)
    print(f"    partial occlusion (0.3<=occ<0.8): {cnt} frames, class differs from both clean neighbours in {len(flips)}: "
          f"{flips[:6]}")


def main() -> None:
    mont_dir = None
    if "--montage" in sys.argv:
        mont_dir = Path(sys.argv[sys.argv.index("--montage") + 1])

    # ---- signatures used by state.py --------------------------------------------------------------------
    sig = inspect.signature(screen.screen_states)
    check("screen_states binds state.py's 3-positional call", _binds(sig, 3), str(sig))
    check("lights_off_events binds state.py's 1-positional call", _binds(inspect.signature(screen.lights_off_events), 1))
    check("lock_events binds state.py's 2-positional call", _binds(inspect.signature(screen.lock_events), 2))
    fields = {f.name for f in dataclasses.fields(screen.ScreenState)}
    check("ScreenState has screen/lid_closed/brightness/lights_off (read by state.py)",
          {"screen", "lid_closed", "brightness", "lights_off"} <= fields, str(sorted(fields)))
    src = (Path("src/pipeline/screen.py").read_text(), Path("src/pipeline/screen_util.py").read_text())
    check("screen.py <= 400 lines", src[0].count("\n") <= 400, f"{src[0].count(chr(10))} lines")
    check("screen_util.py <= 400 lines", src[1].count("\n") <= 400, f"{src[1].count(chr(10))} lines")

    V: dict[int, dict] = {}
    for n in (1, 3, 48, 19, 24, 25, 13, 20, 46, 21, 10, 26, 15, 14, 35, 37, 31, 8, 30, 40, 11, 22, 12):
        V[n] = analyse(n)
        header(n, V[n])

    # ---- camera A, active screens (spec P6: X-1 165/0.55, X-3 171/0.70, X-48 150-164/0.57-0.65) -----------
    for n in (1, 3, 48):
        v = V[n]
        a = frac(v, 0, 1e9, lambda s: s.screen == "active")
        m, wf = rng(v, 0, 1e9, "laptop_mean"), rng(v, 0, 1e9, "white_frac")
        check(f"X-{n} screen 'active' >= 90% of frames (camera A lit screen)", a >= 0.9,
              f"active frac={a:.2f} mean={m[0]:.0f}-{m[1]:.0f} white={wf[0]:.2f}-{wf[1]:.2f}")
        check(f"X-{n} no lights_off / lock events", v["lights"] == [] and v["locks"] == [],
              f"lights={v['lights']} lock={v['locks']}")
        partial_occlusion_report(n, v)
        if mont_dir:
            montage(n, v, [2, 6, 10, 14, 18, 22], mont_dir)

    # ---- X-19: lid closed (spec: 'закрыт 74-99') until 26 s, open & active afterwards ----------------------
    v = V[19]
    for t in (10, 20, 24, 25, 25.5, 26, 26.5, 27, 28, 30):
        print(line(v, t))
    o = frac(v, 0, 25.0, lambda s: s.screen == "off")
    m = rng(v, 0, 25.0, "laptop_mean")
    check("X-19 screen 'off' (closed lid) >= 90% of 0-25 s", o >= 0.9, f"off frac={o:.2f} mean={m[0]:.0f}-{m[1]:.0f}")
    check("X-19 no 'active'/'dim' before 24 s", frac(v, 0, 24.0, lambda s: s.screen in ("active", "dim")) == 0.0)
    a = frac(v, 27.0, 57.0, lambda s: s.screen == "active")
    check("X-19 screen 'active' >= 90% of 27-57 s", a >= 0.9, f"active frac={a:.2f}")
    check("X-19 lid opening at ~26 s is not a lock event", v["locks"] == [], f"lock={v['locks']}")
    partial_occlusion_report(19, v)
    if mont_dir:
        montage(19, v, [5, 15, 24, 25.5, 26.5, 28, 35, 50], mont_dir)

    # ---- dark / black / edge-on screens with an open lid: 'off' and no events (spec T2 exceptions) -----------
    for n, t0, t1, why in ((24, 0, 14.0, "open lid, dark screen (spec: экран тёмный)"),
                           (25, 0, 31.0, "dark screen before the exit (spec 'экран тёмный')"),
                           (13, 55.0, 65.0, "exit 58-62 s with a dark screen, lights on"),
                           (20, 0, 11.0, "black screen 103-107 (spec)"),
                           (46, 0, 18.5, "76-81 (spec)"),
                           (21, 0, 19.0, "cashier away at the start, dark laptop"),
                           (10, 0, 15.0, "lid seen edge-on (grey cabinet behind)"),
                           (26, 0, 19.5, "T1-A exit, screen dark"),
                           (15, 0, 20.5, "T1-A exit, screen dark")):
        v = V[n]
        o = frac(v, t0, t1, lambda s: s.screen == "off")
        m = rng(v, t0, t1, "laptop_mean")
        check(f"X-{n} screen 'off' >= 90% of {t0}-{t1} s ({why})", o >= 0.9, f"off frac={o:.2f} mean={m[0]:.0f}-{m[1]:.0f}")
        check(f"X-{n} no lights_off / lock events", v["lights"] == [] and v["locks"] == [],
              f"lights={v['lights']} lock={v['locks']}")
        if mont_dir:
            dur = v["frames"].frames[-1].timestamp_sec
            montage(n, v, [round(k * dur / 5, 1) for k in range(6)], mont_dir)
    check("X-24 lid open (spec h/w 0.92 >= 0.9)", not any(s.lid_closed for s in V[24]["states"]))
    check("X-13 no lights_off event although the room empties at 58-62 s", V[13]["lights"] == [])

    # ---- X-14: builder's known issue (30 'dim' frames at 117-142 s on a dark screen) -------------------------
    v = V[14]
    d = dist(v, 117, 142)
    m = rng(v, 117, 142, "laptop_mean")
    print(f"    X-14 117-142 s dist={d} mean={m[0]:.0f}-{m[1]:.0f}; whole video dim frames at "
          f"{[f.timestamp_sec for s, f in zip(v['states'], v['frames'].frames) if s.screen == 'dim'][:40]}")
    check("X-14 (clean, dark screen) 'dim'/'active' in < 15% of frames", frac(v, 0, 1e9, lambda s: s.screen != "off") < 0.15,
          f"non-off frac={frac(v, 0, 1e9, lambda s: s.screen != 'off'):.2f}")
    if mont_dir:
        montage(14, v, [60, 100, 118, 125, 132, 140], mont_dir)

    # ---- X-35 / X-37: spec says 'unknown' (lid seen from behind); module gives 'off' (conservative) -----------
    for n in (35, 37):
        v = V[n]
        print(f"    X-{n} dist={dist(v)} mean={rng(v, 0, 1e9, 'laptop_mean')} (spec: unknown; off is equally silent for T2)")
        check(f"X-{n} never 'active'/'dim' (lid seen from behind)", frac(v, 0, 1e9, lambda s: s.screen in ("active", "dim")) == 0.0)
        if mont_dir:
            montage(n, v, [2, 10, 20, 30], mont_dir)

    # ---- no laptop / no desk / no safe: 'unknown', no exceptions ---------------------------------------------
    for n in (31, 8, 30, 40):
        v = V[n]
        check(f"X-{n} 'unknown' throughout (laptop presence {v['layout'].presence.get('laptop', 0):.2f})",
              frac(v, 0, 1e9, lambda s: s.screen == "unknown") == 1.0 and v["locks"] == [], f"dist={dist(v)} lock={v['locks']}")

    # ---- robustness --------------------------------------------------------------------------------------------
    v = V[11]
    fr = v["frames"]
    empty = dataclasses.replace(fr, frames=[])
    check("empty FrameSet -> []", screen.screen_states(empty, v["layout"], []) == [])
    check("lights_off_events([]) / lock_events([]) -> []",
          screen.lights_off_events([]) == [] and screen.lock_events([], v["cm"]) == [])
    lay0 = SceneLayout(v["w"], v["h"], len(fr))                      # nothing found at all
    st0 = screen.screen_states(fr, lay0, v["persons"])
    check("layout without desk/laptop/safe -> all 'unknown', lights-off still detected",
          all(s.screen == "unknown" for s in st0) and screen.lights_off_events(st0) == [32.0], f"events={screen.lights_off_events(st0)}")
    lay1 = dataclasses.replace(v["layout"], desk=None, safe=None)    # laptop but no desk
    st1 = screen.screen_states(fr, lay1, v["persons"], v["dets"])
    check("layout with laptop but no desk -> same classes as with the desk", [s.screen for s in st1] == [s.screen for s in v["states"]])
    short = screen.screen_states(fr, v["layout"], v["persons"][:10], v["dets"])
    check("persons_raw shorter than frames -> no exception", len(short) == len(fr))
    bad = dataclasses.replace(fr, frames=[dataclasses.replace(fr.frames[0], path=Path("/nonexistent/f.jpg"))] + fr.frames[1:])
    stb = screen.screen_states(bad, v["layout"], v["persons"], v["dets"])
    check("unreadable frame -> 'unknown' for it, no exception", stb[0].screen == "unknown" and len(stb) == len(fr))
    lay2 = dataclasses.replace(v["layout"], laptop=(0.0, 0.0, 0.5, 0.5))   # degenerate laptop box
    try:
        st2 = screen.screen_states(fr, lay2, v["persons"], None)
        check("degenerate (sub-pixel) laptop box -> no exception", len(st2) == len(fr), f"dist={ {c: sum(1 for s in st2 if s.screen == c) for c in ('unknown', 'off')} }")
    except Exception as e:  # noqa: BLE001
        check("degenerate (sub-pixel) laptop box -> no exception", False, repr(e))

    # ---- hflip invariance (no left/right pixel assumptions) ---------------------------------------------------
    for n in (11, 22):
        v = V[n]
        fl = hflip_set(v, Path("/tmp/claude-1183600229/-shared-home-sagyndyk-sagadinov/91a791ac-1fe1-4943-bf21-20966b33ad42/scratchpad") / f"verify_screen_flip_X-{n}")
        same = sum(1 for a, b in zip(fl["states"], v["states"]) if a.screen == b.screen) / len(v["states"])
        check(f"X-{n} hflip: >= 97% identical classes, same events", same >= 0.97 and fl["lights"] == v["lights"] and fl["locks"] == v["locks"],
              f"same={same:.2f} lights={fl['lights']} lock={fl['locks']}")

    # ---- runtime -----------------------------------------------------------------------------------------------
    v = V[14]
    t0 = time.perf_counter()
    for f in v["frames"].frames:
        img = cv2.imread(str(f.path), cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    io = time.perf_counter() - t0
    check("X-14 (285 frames) screen_states < 2 s excluding JPEG decode", v["dt"] - io < 2.0,
          f"total={v['dt']:.2f} s, decode+gray={io:.2f} s, rest={v['dt'] - io:.2f} s")
    slow = max(V.items(), key=lambda kv: kv[1]["dt"])
    print(f"    slowest video: X-{slow[0]} {slow[1]['dt']:.2f} s for {len(slow[1]['frames'])} frames")

    # ---- fps consistency: lights_off_events() assumes 2 fps unless told, screen_states() uses frames.sample_fps -
    fr3 = dataclasses.replace(V[11]["frames"], sample_fps=1.0)       # pretend 1 fps sampling of the same frames
    st3 = screen.screen_states(fr3, V[11]["layout"], V[11]["persons"], V[11]["dets"])
    e_default, e_fps = screen.lights_off_events(st3), screen.lights_off_events(st3, 1.0)
    print(f"    fps=1.0 FrameSet: per-frame lights_off from {next((i for i, s in enumerate(st3) if s.lights_off), None)}, "
          f"lights_off_events default={e_default} with fps={e_fps}")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print("  FAILED:", name, "--", detail)
    sys.exit(1 if failed else 0)


def _binds(sig: inspect.Signature, n_pos: int) -> bool:
    try:
        sig.bind(*([None] * n_pos))
        return True
    except TypeError:
        return False


def hflip_set(v, out_dir: Path) -> dict:
    """Mirror the frames and every box; rerun the module."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w = v["w"]
    new_frames = []
    for f in v["frames"].frames:
        p = out_dir / f.path.name
        if not p.exists():
            cv2.imwrite(str(p), cv2.flip(cv2.imread(str(f.path)), 1), [cv2.IMWRITE_JPEG_QUALITY, 95])
        new_frames.append(dataclasses.replace(f, path=p))
    fr = dataclasses.replace(v["frames"], frames=new_frames)

    def fb(b):
        return (w - b[2], b[1], w - b[0], b[3])

    def fd(d: Detection) -> Detection:
        return dataclasses.replace(d, box=fb(d.box))

    dets = [[fd(d) for d in ds] for ds in v["dets"]]
    persons = [[fd(d) for d in ds] for ds in v["persons"]]
    lay = v["layout"]
    lay2 = dataclasses.replace(lay, desk=fb(lay.desk) if lay.desk else None, laptop=fb(lay.laptop) if lay.laptop else None,
                               counter_machine=fb(lay.counter_machine) if lay.counter_machine else None,
                               safe=fb(lay.safe) if lay.safe else None, door=fb(lay.door) if lay.door else None,
                               cashier_seat=fb(lay.cashier_seat) if lay.cashier_seat else None)
    cm = CashierStandIn(persons, lay2)
    states = screen.screen_states(fr, lay2, persons, dets)
    fps = fr.sample_fps
    return dict(states=states, lights=[i / fps for i in screen.lights_off_events(states)],
                locks=[i / fps for i in screen.lock_events(states, cm, fps)])


if __name__ == "__main__":
    main()
