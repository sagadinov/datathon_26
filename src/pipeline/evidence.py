"""Evidence frame selection and JPEG/base64 encoding (q=80, long side <= 640) for events."""
from __future__ import annotations

import base64

import cv2

from src.pipeline.frames import FrameSet
from src.pipeline.rules import Event
from src.pipeline.state import Timeline


def encode_jpeg_b64(img, quality: int = 80, long_side: int = 640) -> str:
    h, w = img.shape[:2]
    if max(h, w) > long_side:
        s = long_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _annotate(img, st, event: Event, tl: Timeline | None = None):
    """Draw the objects relevant to the event type (people, valuables, cup, safe) on a copy."""
    img = img.copy()
    want = {"T1": {"cash", "money_bag", "bag", "book"}, "T5": {"bag", "money_bag", "jacket", "book"}, "T6": {"cup", "book"},
            "T7": {"cash", "money_bag"}, "T8": {"cash"}, "T9": {"money_bag", "bag"}, "T2": {"cup", "book"}}.get(event.type_id, set())
    for o in st.objects:
        if o.label in want and (o.on_desk or o.in_cashier) and o.score >= 0.3 and o.is_event_object:
            x1, y1, x2, y2 = (int(v) for v in o.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    blobs = getattr(tl, "blob_boxes", None) if tl is not None else None
    if event.type_id == "T1" and blobs:                       # new static desk object found by blob.py
        k = next((j for j, f in enumerate(tl.frames) if f is st), -1)
        for b in (blobs[k] if 0 <= k < len(blobs) else []):
            x1, y1, x2, y2 = (int(v) for v in b)
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 2)
    for p in st.persons:
        x1, y1, x2, y2 = (int(v) for v in p.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0) if p.is_cashier else (255, 160, 0), 2)
    return img


def evidence_frames(frames: FrameSet, tl: Timeline, event: Event, n: int = 3, annotate: bool = True) -> list[dict]:
    """1-3 frames: event start, ~1 s in, and ~2 s in (deduplicated)."""
    idxs = []
    for i in getattr(event, "evidence_idx", []) or []:
        i = max(0, min(len(tl.frames) - 1, int(i)))
        if i not in idxs:
            idxs.append(i)
    if not idxs:
        for off in (0.0, 1.0, 2.5):
            t = min(event.t_start + off, event.t_end)
            i = min(range(len(tl.frames)), key=lambda k: abs(tl.frames[k].t - t))
            if i not in idxs:
                idxs.append(i)
    out = []
    for i in idxs[:n]:
        st = tl.frames[i]
        img = frames.frames[st.index].load()
        if annotate:
            img = _annotate(img, st, event, tl)
        out.append({"timestamp_sec": round(st.t, 1), "image_base64": encode_jpeg_b64(img)})
    return out
