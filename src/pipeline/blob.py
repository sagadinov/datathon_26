"""Desk blobs (spec P5 `desk_blob`): small new objects lying on the desk, found by pixel comparison with a
background model of the desk - no object detector involved (banknote bundles of 8x7 .. 25x12 px at 640 px are
invisible to it, see docs/limitations.md item 2).

Model (thresholds relative to the desk box and the measured desk noise, no pixel constants):
  * ROI = layout.desk. Static masks: laptop, counter_machine, background phone clusters (slightly expanded).
    Per-frame masks: every person box (slightly expanded).
  * Background (Lab, per pixel) = median over unmasked frames of the first EARLY_S seconds; where a person covered
    the pixel for that whole window, the median over the first EARLY_S seconds ignoring person boxes (robust to
    moving arms); where still unknown, the median over all unmasked frames.
  * Difference = Lab distance of (frame - background) after removing its local mean over a window of ~1/4 of the
    desk box (high-pass: the cashier's shadow and auto-exposure shifts of 20-30 L levels vanish, compact objects
    stay). Threshold per frame = max(MIN_THR, median + K_MAD * MAD of the difference over unmasked pixels).
  * Candidate = connected component of (difference > thr), area MIN_AREA..MAX_AREA px^2 (scaled from 640 px
    frames), compact, centre in the upper MAX_REL_Y of the desk box (desk top, not the front panel / chair), not
    touching the ROI border or a person box, isolated (the ring around it is unchanged), contrasting with that ring
    in the same frame, and NOT already there in the first START_S seconds (static objects, e.g. the desk phone).
  * Stable = matched (IoU >= IOU_MIN, gaps <= MAX_GAP frames) over >= MIN_FRAMES frames; the whole chain counts.

API: desk_blobs(frames, tl) -> per-frame list of stable blob boxes (frame coordinates, for evidence frames);
     blob_series(frames, tl) -> per-frame bool. Never runs a model; ~1 s for a 70 s video on CPU.
Known gaps: an object placed where the clean desk was never visible unmasked before it arrived (X-15 bag placed
at 6 s under the seated cashier's box) has no reference and is missed; banknotes inside the counting machine's
hopper become a blob when the layout's counter_machine box is wrong (X-4); the bundle is suppressed while it
touches a person box (X-4 44-53 s, client's hands next to it).
"""
from __future__ import annotations

import numpy as np

from src.pipeline.frames import FrameSet
from src.pipeline.zones import area, cluster_boxes, inter_area

Box = tuple[float, float, float, float]

PARAMS = {
    "early_s": 10.0,      # background window
    "start_s": 2.0,       # objects already there in this window are static, never a blob
    "min_samples": 3,     # unmasked frames needed for a per-pixel median
    "k_mad": 4.0,         # threshold = noise median + k_mad * MAD ...
    "min_thr": 10.0,      # ... but never below this Lab distance
    "min_area": 40.0,     # px^2 at 640 px long side (scaled with the frame)
    "max_area": 600.0,
    "min_side": 4.0,      # px, bbox width and height
    "min_fill": 0.35,     # component area / bbox area
    "ring_changed_max": 0.30,   # fraction of ring pixels that differ from the background
    "ring_contrast": 0.8,       # blob-vs-ring Lab distance >= ring_contrast * thr
    "static_frac": 1.0,         # blob-vs-first-2s distance < static_frac * thr -> static object (bundles: 20-65)
    "max_rel_y": 0.6,           # blob centre must lie above desk.y0 + max_rel_y * desk_h (desk top, not the front panel/chair)
    "person_pad": 0.03,         # person box expansion (fraction of its size) for masks / touching
    "static_pad_px": 2,
    "min_frames": 3,
    "iou_min": 0.5,
    "max_gap": 1,
}


def _clip(b: Box, w: int, h: int) -> tuple[int, int, int, int]:
    return (max(0, int(np.floor(b[0]))), max(0, int(np.floor(b[1]))),
            min(w, int(np.ceil(b[2]))), min(h, int(np.ceil(b[3]))))


def _expand(b: Box, frac: float = 0.0, px: float = 0.0) -> Box:
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - w * frac - px, b[1] - h * frac - px, b[2] + w * frac + px, b[3] + h * frac + px)


def _paint(mask: np.ndarray, boxes, roi, value=False) -> None:
    x1, y1, x2, y2 = roi
    for b in boxes:
        bx1, by1, bx2, by2 = _clip(b, x2, y2)
        bx1, by1 = max(bx1, x1), max(by1, y1)
        if bx2 > bx1 and by2 > by1:
            mask[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = value


def _box_iou(a: Box, b: Box) -> float:
    i = inter_area(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def _masked_median(stack: np.ndarray, valid: np.ndarray, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median of stack (n,H,W,3) over frames where valid (n,H,W); known where count >= min_samples.
    Sort-based (invalid samples pushed to the end), ~10x faster than np.nanmedian."""
    n = stack.shape[0]
    count = valid.sum(axis=0)
    data = stack.astype(np.uint16)
    data[~valid] = 999
    data.sort(axis=0)
    idx = np.clip((count - 1) // 2, 0, n - 1)[None, ..., None]
    med = np.take_along_axis(data, np.broadcast_to(idx, (1,) + data.shape[1:]), axis=0)[0].astype(np.float32)
    known = count >= min_samples
    med[~known] = 0.0
    return med, known


def _masked_mean(img: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """Local mean of img (H,W,3) over valid pixels in a k x k window (normalised box filter)."""
    import cv2

    w = valid.astype(np.float32)
    num = cv2.boxFilter(img * w[..., None], -1, (k, k), normalize=False, borderType=cv2.BORDER_REFLECT)
    den = cv2.boxFilter(w, -1, (k, k), normalize=False, borderType=cv2.BORDER_REFLECT)
    return num / np.maximum(den, 1.0)[..., None]


class DeskBlobModel:
    """Background model of the desk + per-frame blob extraction (see module doc)."""

    def __init__(self, frames: FrameSet, tl, params: dict | None = None):
        import cv2

        self.cv2, self.p = cv2, {**PARAMS, **(params or {})}
        p = self.p
        self.n = len(frames)
        self.t = [f.timestamp_sec for f in frames.frames]
        self.boxes: list[list[Box]] = [[] for _ in range(self.n)]
        self.ages: list[list[int]] = [[] for _ in range(self.n)]
        self.stats: dict = {}
        if tl.layout.desk is None or self.n == 0:
            return
        img0 = frames.frames[0].load()
        H, W = img0.shape[:2]
        self.scale = (max(H, W) / 640.0) ** 2
        self.roi = roi = _clip(tl.layout.desk, W, H)
        x1, y1, x2, y2 = roi
        rh, rw = y2 - y1, x2 - x1
        if rh < 8 or rw < 8:
            return
        # static masks: laptop, counter machine, background phone clusters
        static_ok = np.ones((rh, rw), dtype=bool)
        anchors = [b for b in (tl.layout.laptop, tl.layout.counter_machine) if b]
        phones = [o.box for s in tl.frames for o in s.objects if o.label == "phone" and o.background]
        anchors += [b for b, k in cluster_boxes(phones, 0.5) if k >= 3]
        _paint(static_ok, [_expand(b, px=p["static_pad_px"]) for b in anchors], roi)
        self.static_ok = static_ok
        # per-frame person masks (expanded) and raw person boxes
        self.persons = [[q.box for q in s.persons] for s in tl.frames]
        n_frames = self.n
        stack = np.empty((n_frames, rh, rw, 3), dtype=np.uint8)
        valid = np.empty((n_frames, rh, rw), dtype=bool)
        for i, f in enumerate(frames.frames):
            img = img0 if i == 0 else f.load()
            stack[i] = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)
            m = static_ok.copy()
            _paint(m, [_expand(b, p["person_pad"], 1) for b in self.persons[i]], roi)
            valid[i] = m
        self.stack, self.valid = stack, valid
        early = np.array([t < p["early_s"] for t in self.t])
        if early.sum() < p["min_samples"]:
            early[:] = True
        # background: masked early median -> raw early median -> masked global median
        bg, known = _masked_median(stack[early], valid[early], p["min_samples"])
        raw_bg, raw_known = _masked_median(stack[early], np.broadcast_to(static_ok, (int(early.sum()), rh, rw)),
                                           p["min_samples"])
        fill = ~known & raw_known
        bg[fill], known[fill] = raw_bg[fill], True
        if not known.all():                                  # global median, only where still unknown
            g_bg, g_known = _masked_median(stack[:, ~known], valid[:, ~known], p["min_samples"])
            bg[~known], known[~known] = g_bg, g_known
        self.bg, self.known = bg, known
        # first START_S seconds reference (static objects), raw
        first = np.array([t < p["start_s"] for t in self.t])
        if first.sum() == 0:
            first[0] = True
        self.first_ref, self.first_known = _masked_median(
            stack[first], np.broadcast_to(static_ok, (int(first.sum()), rh, rw)), 1)
        # high-pass window ~ a quarter of the desk box, odd; global floor of the threshold from early frames
        self.hp_k = max(15, int(0.25 * min(rh, rw)) | 1)
        self.thr = p["min_thr"]
        thrs = [self._dist(i)[1] for i in np.flatnonzero(early)[:12]]
        self.thr = max(p["min_thr"], float(np.median(thrs))) if thrs else p["min_thr"]
        self.stats = {"roi": roi, "thr": round(self.thr, 2), "hp_k": self.hp_k,
                      "known_frac": round(float(known.mean()), 3), "n_anchors": len(anchors)}
        self._run()

    # ------------------------------------------------------------------------------------------ per frame
    def _dist(self, i: int) -> tuple[np.ndarray, float]:
        """High-passed Lab distance to the background (local mean over valid pixels removed, so shadows and
        exposure changes vanish) and this frame's robust threshold."""
        p = self.p
        v = self.valid[i] & self.known
        d = self.stack[i].astype(np.float32) - self.bg
        d -= _masked_mean(d, v, self.hp_k)
        dist = np.sqrt((d * d).sum(axis=2))
        vals = dist[v]
        if vals.size < 50:
            return dist, self.thr
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        return dist, max(self.thr, med + p["k_mad"] * 1.4826 * mad)

    def candidates(self, i: int) -> list[tuple[Box, dict]]:
        """Blob candidates of frame i (before temporal stability), as (box, info)."""
        cv2, p = self.cv2, self.p
        x1, y1, x2, y2 = self.roi
        rh, rw = y2 - y1, x2 - x1
        dist, thr = self._dist(i)
        dist = cv2.medianBlur(dist.astype(np.float32), 3)
        changed = (dist > thr) & self.known
        fg = (changed & self.valid[i]).astype(np.uint8)
        n_cc, labels, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        out = []
        persons = [_expand(b, p["person_pad"], 1) for b in self.persons[i]]
        lab = self.stack[i].astype(np.float32)
        for k in range(1, n_cc):
            bx, by, bw, bh, a = st[k]
            a = float(a) / self.scale
            if not (p["min_area"] <= a <= p["max_area"]) or min(bw, bh) < p["min_side"] or a * self.scale < p["min_fill"] * bw * bh:
                continue
            if bx <= 0 or by <= 0 or bx + bw >= rw or by + bh >= rh:
                continue                                    # touches the ROI border (chair, legs, desk edge)
            box = (float(x1 + bx), float(y1 + by), float(x1 + bx + bw), float(y1 + by + bh))
            if by + bh / 2 > p["max_rel_y"] * rh:
                continue                                    # on the front panel / chair, not the desk top
            if any(inter_area(box, pb) > 0 for pb in persons):
                continue
            comp = labels == k
            ring = cv2.dilate(comp.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            ring &= ~cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ring &= self.valid[i] & self.known
            if ring.sum() < 8:
                continue
            ring_changed = float(changed[ring].mean())
            if ring_changed > p["ring_changed_max"]:
                continue                                    # part of a larger change (shadow, clothing, chair)
            c_med, r_med = np.median(lab[comp], axis=0), np.median(lab[ring], axis=0)
            contrast = float(np.linalg.norm(c_med - r_med))
            if contrast < p["ring_contrast"] * thr:
                continue
            fk = comp & self.first_known
            static_d = float(np.median(np.linalg.norm(lab[fk] - self.first_ref[fk], axis=1))) if fk.any() else 1e9
            if static_d < p["static_frac"] * thr:
                continue                                    # was there in the first seconds already
            out.append((box, {"area": round(a, 1), "contrast": round(contrast, 1), "ring_changed": round(ring_changed, 2),
                              "static_d": round(static_d, 1), "mean_dist": round(float(dist[comp].mean()), 1), "thr": round(thr, 1)}))
        return out

    def _run(self) -> None:
        p = self.p
        cands = [self.candidates(i) for i in range(self.n)]
        # chains: match to the previous frames (gap <= max_gap) by IoU
        chain_id = [[-1] * len(c) for c in cands]
        chains: dict[int, list[tuple[int, int]]] = {}
        next_id = 0
        for i in range(self.n):
            for k, (box, _) in enumerate(cands[i]):
                best, best_iou = None, p["iou_min"]
                for j in range(max(0, i - 1 - p["max_gap"]), i):
                    for kk, (pb, _) in enumerate(cands[j]):
                        v = _box_iou(box, pb)
                        if v >= best_iou and chain_id[j][kk] >= 0:
                            best, best_iou = chain_id[j][kk], v
                if best is None:
                    best = next_id
                    next_id += 1
                chain_id[i][k] = best
                chains.setdefault(best, []).append((i, k))
        self.cands, self.chains = cands, chains
        for cid, members in chains.items():
            if len(members) >= p["min_frames"]:
                for i, k in members:
                    self.boxes[i].append(cands[i][k][0])
                    self.ages[i].append(len(members))


def desk_blobs(frames: FrameSet, tl, params: dict | None = None) -> list[list[Box]]:
    """Per frame: boxes (frame coordinates) of stable new blobs on the desk."""
    return DeskBlobModel(frames, tl, params).boxes


def blob_series(frames: FrameSet, tl, params: dict | None = None) -> list[bool]:
    """Per frame: a stable new blob exists on the desk."""
    return [bool(b) for b in desk_blobs(frames, tl, params)]
