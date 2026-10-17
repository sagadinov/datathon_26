"""Laptop screen state and room light (spec: docs/violations_observed.md P6, P7).

Pixels are read from the cached frame JPEGs; everything else is relative to the laptop box.

Region: the upper 58% of layout.laptop (the lid/screen; the keyboard below is dark and spoiled the v1
signal). Per frame the region gives mean gray, the white fraction (> 170), B - R and the person overlap;
the whole frame gives `brightness`.

Classes (lit room):
  lit / off   hysteresis on the region mean: lit from >= 115, off from <= 105, in between keep the previous
              state (first frame: >= 110). 115 is the spec's no-GPU threshold (T2, X-22 115-127 lit vs X-23
              105-115 not lit); the Windows lock screen of X-23 / X-29 (blue wallpaper, 105-112) and the black
              screens of X-27 (100-105), X-20 (100-103), X-34/38 (64-73) are therefore 'off'.
  active      lit and mean >= 130 and (white >= 0.30 or B - R >= 12): X-28 bluish-white 145-151 / 0.38 /
              +16..20, X-22/23/29 before the lock 141-147 / 0.31-0.39, X-19 163-173 / 0.67 / +35.
  dim         lit but not active: X-22 after the lid tilt 115-127 / 0.14, X-11 lit room 105-131 / 0.05. A region
              without bright pixels (white < 0.02) is background behind a lid seen edge-on (X-10, X-15) -> 'off'.
  Lights off (after a P7 event): active = white fraction of the WHOLE laptop box >= 0.10 (X-11 0.15-0.17 vs
  X-12 0.01-0.02); dim when the region is >= 1.25x brighter than the dark frame; else off.
  lid closed (h/w of the per-frame laptop box <= 0.75 and <= 0.8x the video's open ratio, runs >= 2 s) forces
  'off'.
Gates: 'unknown' when laptop presence < 0.7 (X-45: chair back 0.49), when the laptop box is not on/next to the
desk, or (carried from the last clean frame) when a person box covers >= 80% of the region. The seated cashier
of camera B overlaps 40-60% of the region without hiding the screen, hence the high threshold.
Smoothing: majority of 3 in the 4-frame window [i-2, i+1] at 2 fps (X-19 flicker 34.5-44 s).
P7 lights_off: brightness drop >= 35% against the median of the previous 3 s that stays <= 80% of it for the
next 2.5 s (X-11 139->45, X-12 139->75); exposure spikes (X-12 154, X-19 157-160, X-30 149) are rises and the
later returns are < 35%, so they never fire.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.pipeline.detect import Detection
from src.pipeline.frames import FrameSet
from src.pipeline.screen_util import (Measure, laptop_hw_series, laptop_on_desk, laptop_presence, majority,
                                      measure_frame, median3, median_of)
from src.pipeline.zones import SceneLayout

MIN_PRESENCE = 0.7
OCCLUDE_FRAC = 0.8
LIT_ON, LIT_OFF, LIT_INIT = 115.0, 105.0, 110.0
ACTIVE_MEAN, ACTIVE_WHITE, ACTIVE_BLUE = 130.0, 0.30, 12.0
DARK_ACTIVE_WHITE, DARK_DIM_RATIO, DARK_MIN_MEAN = 0.10, 1.25, 60.0
DIM_MIN_WHITE = 0.02      # a lit screen shows at least a few bright pixels (X-11 lit room 0.03-0.09, X-22 0.09-0.15)
LID_CLOSED_HW, LID_CLOSED_REL, LID_MIN_SEC = 0.75, 0.8, 2.0
LIGHTS_DROP, LIGHTS_STAY, LIGHTS_BACK = 0.35, 0.80, 0.85
LOCK_DROP = 40.0
SMOOTH_BEFORE_SEC, SMOOTH_AFTER_SEC = 1.0, 0.5    # 4-frame majority window at 2 fps


@dataclass
class ScreenState:
    screen: str            # 'active' | 'dim' | 'off' | 'unknown'
    lid_closed: bool       # laptop box h/w <= 0.75 (X-12: 1.11-1.26 open -> 0.68-0.73 closed)
    brightness: float      # frame mean gray
    lights_off: bool       # frame brightness below the video's lit level (after a lights_off event)
    laptop_mean: float     # mean L of the upper 55-60% of the laptop box (NaN when unknown)
    white_frac: float      # fraction of pixels > 170 in that region
    occluded: bool         # a person box covers the laptop this frame -> state carried from the last clean frame
    white_full: float = 0.0  # fraction of pixels > 170 in the whole laptop box (lights-off rule, X-11 0.15-0.17)


def screen_states(frames: FrameSet, layout: SceneLayout, persons_raw: list[list[Detection]],
                  dets_per_frame: list[list[Detection]] | None = None) -> list[ScreenState]:
    """P6/P7 per frame (see the module docstring for the thresholds).

    `dets_per_frame` (optional) supplies the per-frame laptop boxes for lid_closed and for a presence measure
    that survives the lid closing (open and closed boxes form two clusters, X-12 0.67 + 0.26). Without it
    the layout box alone is used: lid_closed is constant and presence is layout.presence['laptop']."""
    n = len(frames)
    fps = float(frames.sample_fps or 2.0)
    laptop = layout.laptop
    if laptop is not None and (laptop_presence(layout, dets_per_frame) < MIN_PRESENCE
                               or not laptop_on_desk(laptop, layout.desk)):
        laptop = None
    meas = [measure_frame(f.path, laptop, persons_raw[i] if i < len(persons_raw) else []) for i, f in enumerate(frames)]
    off, _ = lights_off_series([m.brightness for m in meas], fps)
    lid = lid_closed_series(laptop_hw_series(layout, dets_per_frame, n), fps) if laptop else [False] * n

    raw: list[str] = []
    occluded: list[bool] = []
    lit: bool | None = None
    last: tuple[str, bool] | None = None      # (class, lights_off) of the last clean frame
    for i, m in enumerate(meas):
        if laptop is None or not m.ok or math.isnan(m.mean):
            raw.append("unknown")
            occluded.append(False)
            continue
        occ = m.occ >= OCCLUDE_FRAC
        occluded.append(occ)
        if occ:
            raw.append(last[0] if last is not None and last[1] == off[i] else "unknown")
            continue
        cls, lit = _classify(m, off[i], lit)
        if lid[i]:
            cls = "off"
        last = (cls, off[i])
        raw.append(cls)
    final = majority(raw, max(1, int(round(SMOOTH_BEFORE_SEC * fps))), max(0, int(round(SMOOTH_AFTER_SEC * fps))))
    return [ScreenState(final[i], lid[i], m.brightness, off[i], m.mean, m.white, occluded[i], m.white_full)
            for i, m in enumerate(meas)]


def _classify(m: Measure, lights_off: bool, lit: bool | None) -> tuple[str, bool]:
    """Raw class of one clean frame and the updated lit/off hysteresis state."""
    if lights_off:
        if m.white_full >= DARK_ACTIVE_WHITE:
            return "active", True
        if m.mean >= DARK_MIN_MEAN and m.mean >= DARK_DIM_RATIO * m.brightness:
            return "dim", True
        return "off", False
    if lit is None:
        lit = m.mean >= LIT_INIT
    elif lit:
        lit = m.mean > LIT_OFF
    else:
        lit = m.mean >= LIT_ON
    if not lit or m.white < DIM_MIN_WHITE:
        # a mid-grey region without any bright pixel is background behind a lid seen edge-on (X-10 123-126 /
        # 0.00, X-15 100-114 / 0.00), not a lit screen; the hysteresis keeps following the mean
        return "off", lit
    if m.mean >= ACTIVE_MEAN and (m.white >= ACTIVE_WHITE or m.blue >= ACTIVE_BLUE):
        return "active", True
    return "dim", True


def lid_closed_series(hw: list[float], fps: float = 2.0) -> list[bool]:
    """Closed lid: h/w <= 0.75 and <= 0.8x the video's own open ratio (90th percentile of the series, so a wide
    laptop seen frontally is not 'closed' all the time). Runs shorter than 2 s are box jitter (X-14 2-3 frames
    at 0.69-0.73, X-17 2 frames) unless they reach the end of the clip (spec D3: >= 2 frames there)."""
    vals = [v for v in hw if not math.isnan(v)]
    if not vals:
        return [False] * len(hw)
    ref_open = sorted(vals)[int(0.9 * (len(vals) - 1))] if len(vals) >= 5 else None
    raw = median3([(not math.isnan(v)) and v <= LID_CLOSED_HW and (ref_open is None or v <= LID_CLOSED_REL * ref_open)
                   for v in hw])
    min_run, n = max(2, int(round(LID_MIN_SEC * fps))), len(raw)
    out, i = [False] * n, 0
    while i < n:
        j = i
        while j < n and raw[j]:
            j += 1
        if j > i and (j - i >= min_run or (j == n and j - i >= 2)):
            out[i:j] = [True] * (j - i)
        i = max(j, i + 1)
    return out


def lights_off_series(brightness: list[float], fps: float) -> tuple[list[bool], list[int]]:
    """P7. Returns (lights_off per frame, event frame indices). An event is a frame whose brightness is
    <= 65% of the median of the previous 3 s and whose following 2.5 s stay <= 80% of it; the dark run lasts
    until the brightness recovers to >= 85% of that reference."""
    n = len(brightness)
    pre, post = max(2, int(round(3.0 * fps))), max(1, int(round(2.5 * fps)))
    off, events = [False] * n, []
    i = 1
    while i < n:
        ref = median_of(brightness[max(0, i - pre):i])
        if ref > 0 and brightness[i] <= (1 - LIGHTS_DROP) * ref and median_of(brightness[i:i + post]) <= LIGHTS_STAY * ref:
            events.append(i)
            j = i
            while j < n and brightness[j] < LIGHTS_BACK * ref:
                off[j] = True
                j += 1
            i = max(j, i + 1)
        else:
            i += 1
    return off, events


def lights_off_events(states: list[ScreenState], fps: float = 2.0) -> list[float]:
    """Frame INDICES (as floats) of lights-off events; the caller converts to seconds. The states carry no
    frame rate, so the pipeline's 2 fps sampling is assumed unless `fps` is given."""
    _, events = lights_off_series([s.brightness for s in states], fps)
    return [float(i) for i in events]


def lock_events(states: list[ScreenState], cm, fps: float = 2.0) -> list[float]:
    """Frame indices where the laptop mean drops >= 40 within <= 1 s while the cashier is present
    (X-29 144->82 at 5.5-6.0, X-23 145->106, X-22 144->96). Both frames must be clean (no occlusion), under
    the same lighting, with an open lid, and the earlier one lit; `cm.present[i]` (per-frame bool) is checked
    at either end of the drop. One event per drop (no second event within the window)."""
    present = getattr(cm, "present", None)

    def is_present(i: int) -> bool:
        return True if present is None or i >= len(present) else bool(present[i])

    win = max(1, int(round(1.0 * fps)))
    events: list[float] = []
    last_event = -10 * win
    for i in range(1, len(states)):
        s = states[i]
        if s.occluded or s.lid_closed or math.isnan(s.laptop_mean) or i - last_event <= win:
            continue
        for j in range(max(0, i - win), i):
            sj = states[j]
            if sj.occluded or sj.lights_off != s.lights_off or sj.screen not in ("active", "dim"):
                continue
            if sj.laptop_mean - s.laptop_mean >= LOCK_DROP and (is_present(i) or is_present(j)):
                events.append(float(i))
                last_event = i
                break
    return events
