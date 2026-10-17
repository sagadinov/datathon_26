"""Cashier wrist positions from a CPU pose model (P10 of docs/violations_observed.md).

Model: RTMPose (OpenMMLab, Apache-2.0) run through rtmlib (Apache-2.0) on ONNXRuntime, CPU only.
Weights live in models/pose/ (downloaded once at setup time); nothing is fetched at run time. Setup:

    cd models/pose && for n in rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504 \
                               rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504; do
      curl -sSL -o $n.zip https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/$n.zip
      unzip -o -q $n.zip -d $n && cp $n/*/rtmpose_onnx/$n/end2end.onnx ${n%%_simcc*}_body7_256x192.onnx
      rm -rf $n $n.zip; done
    # -> rtmpose-m_body7_256x192.onnx (54 MB, sha1 8f0c4b41...) and rtmpose-s_body7_256x192.onnx (22 MB)
The model is a top-down estimator: it is run on the cashier box of every sampled frame (the box is padded and
warped to 192x256 by rtmlib), never on the whole frame, so a 2 fps video costs ~15-30 ms per frame on a few
CPU threads.

    wrists_series(frames, tl) -> [{"lw": (x, y, conf), "rw": (x, y, conf), ...}, ...]   one dict per frame

COCO-17 keypoint order; wrists are indices 9 (left) and 10 (right). Frames without a cashier box, or whose
image cannot be loaded, get conf = 0.0 for both wrists (coordinates nan).
"""
from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import time
from pathlib import Path

import numpy as np

from src.common import MODELS_DIR

log = logging.getLogger("pipeline.pose")

POSE_DIR = MODELS_DIR / "pose"
POSE_MODELS = {   # name -> (file, (w, h) input size)
    "rtmpose-m": ("rtmpose-m_body7_256x192.onnx", (192, 256)),
    "rtmpose-s": ("rtmpose-s_body7_256x192.onnx", (192, 256)),
}
DEFAULT_MODEL = os.environ.get("POSE_MODEL", "rtmpose-m")
# COCO indices used by the rules / self-test
KP = {"nose": 0, "l_sho": 5, "r_sho": 6, "l_elb": 7, "r_elb": 8, "lw": 9, "rw": 10, "l_hip": 11, "r_hip": 12}
EMPTY = (math.nan, math.nan, 0.0)

_MODEL_CACHE: dict = {}


def pose_model_path(name: str = DEFAULT_MODEL) -> Path:
    return POSE_DIR / POSE_MODELS[name][0]


def load_pose(name: str = DEFAULT_MODEL, threads: int | None = None):
    """Load (once per process) the RTMPose ONNX model with an explicit CPU thread budget."""
    key = (name, threads)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import onnxruntime as ort
    from rtmlib import RTMPose

    path = pose_model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"pose weights missing: {path} (run scripts/download_models.sh)")
    threads = threads or int(os.environ.get("POSE_THREADS", "0")) or min(8, os.cpu_count() or 1)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):     # rtmlib prints "load ... backend"
        model = RTMPose(str(path), model_input_size=POSE_MODELS[name][1], backend="onnxruntime", device="cpu")
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    model.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    log.info("pose model %s loaded in %.2fs (%d threads)", path.name, time.perf_counter() - t0, threads)
    _MODEL_CACHE[key] = model
    return model


def _kp(kps: np.ndarray, scores: np.ndarray, j: int) -> tuple[float, float, float]:
    return float(kps[j, 0]), float(kps[j, 1]), float(scores[j])


def _pad_box(box, w: int, h: int, pad: float = 0.15):
    """Widen the cashier box a little so that wrists at the box border still fall inside the crop."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return [max(0.0, x0 - pad * bw), max(0.0, y0 - pad * bh), min(float(w), x1 + pad * bw), min(float(h), y1 + pad * bh)]


def wrists_series(frames, tl, model=None, name: str = DEFAULT_MODEL, threads: int | None = None) -> list[dict]:
    """Per-frame wrist keypoints of the cashier (tl.frames[i].cashier_box), see module docstring.

    Each dict holds "lw"/"rw" = (x, y, conf) in frame pixels, "box" = the cashier box used (or None), and
    "kps" = {name: (x, y, conf)} for the keypoints in KP (shoulders, elbows, hips, nose) for diagnostics.
    """
    n = len(tl.frames)
    out = [{"lw": EMPTY, "rw": EMPTY, "box": None, "kps": {}} for _ in range(n)]
    boxes = [s.cashier_box for s in tl.frames]
    if not any(b is not None for b in boxes):
        return out
    model = model or load_pose(name, threads)
    t0, done = time.perf_counter(), 0
    for i, box in enumerate(boxes):
        if box is None or i >= len(frames.frames):
            continue
        try:
            img = frames.frames[i].load()
        except Exception as e:                      # a missing cached frame must not kill the rule stage
            log.warning("pose: frame %d unreadable (%s)", i, e)
            continue
        h, w = img.shape[:2]
        bb = _pad_box(box, w, h)
        kps, scores = model(img, [bb])             # (1, 17, 2), (1, 17)
        k, s = kps[0], scores[0]
        out[i] = {"lw": _kp(k, s, KP["lw"]), "rw": _kp(k, s, KP["rw"]), "box": tuple(float(v) for v in box),
                  "kps": {nm: _kp(k, s, j) for nm, j in KP.items()}}
        done += 1
    dt = time.perf_counter() - t0
    log.info("pose: %d/%d frames in %.2fs (%.1f ms/frame)", done, n, dt, 1000 * dt / max(1, done))
    return out


def wrist_conf(wr: dict) -> float:
    """Max confidence of the two wrists (0 when the cashier was not seen)."""
    return max(wr["lw"][2], wr["rw"][2])
