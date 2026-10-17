"""Laptop screen state and room light (spec: docs/violations_observed.md P6, P7).

Pixels are read from the cached frame JPEGs; everything else is relative to the laptop box.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.detect import Detection
from src.pipeline.frames import FrameSet
from src.pipeline.zones import SceneLayout


@dataclass
class ScreenState:
    screen: str            # 'active' | 'dim' | 'off' | 'unknown'
    lid_closed: bool       # laptop box h/w <= 0.75 (X-12: 1.11-1.26 open -> 0.68-0.73 closed)
    brightness: float      # frame mean gray
    lights_off: bool       # frame brightness below the video's lit level (after a lights_off event)
    laptop_mean: float     # mean L of the upper 55-60% of the laptop box (NaN/0 when unknown)
    white_frac: float      # fraction of pixels > 170 in that region
    occluded: bool         # a person box covers the laptop this frame -> state carried from the last clean frame


def screen_states(frames: FrameSet, layout: SceneLayout, persons_raw: list[list[Detection]]) -> list[ScreenState]:
    """P6/P7 per frame.
    Region: upper 55-60% of layout.laptop; 'unknown' when presence(laptop) < 0.7 (layout.presence['laptop']) or
    the laptop box is not within/adjacent to the desk, or when a person box overlaps the region (carry last).
    active: mean >= 130 and white >= 0.45, or (B - R >= 25 and mean >= 120, camera A bluish screen; X-28
    135-145 at 33-39% white). dim: mean 100-130 (X-22 115-127 tilted lid, X-23 105-115 wallpaper).
    off/closed: mean <= 99 (X-27 72-80, X-19 closed 74-99, X-34/38 74-79, X-46 76-81, X-29 lock 77-90).
    Hysteresis on >= 120 / off <= 105 and majority 3 of 4 frames (X-19 flicker). When lights are off
    (lights_off), 'active' = white >= 0.10 in the laptop box (X-11: 0.15-0.17 vs X-12: 0.01-0.02).
    lights_off: brightness drop >= 35% in one frame (X-11 139->45, X-12 139->75) and stays low; exposure
    spikes +15-20 (X-12 154, X-19 157-160, X-30 149) are not events."""
    raise NotImplementedError


def lights_off_events(states: list[ScreenState]) -> list[float]:
    """Timestamps (frame index / fps handled by the caller? no: return frame INDICES as floats) of lights-off events."""
    raise NotImplementedError


def lock_events(states: list[ScreenState], cm) -> list[float]:
    """Frame indices where the laptop mean drops >= 40 within <= 1 s while the cashier is present
    (X-29 144->82 at 5.5-6.0, X-23 145->106, X-22 144->96)."""
    raise NotImplementedError
