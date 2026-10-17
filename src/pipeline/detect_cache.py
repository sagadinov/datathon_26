"""Disk cache of per-frame detections, keyed by the frame-set key and the detector config.

    python -m src.pipeline.detect_cache --all --procs 4 --threads 32   # warm the cache for data/open

run.py and eval read through `cached_detections()`, so rule iterations never re-run the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from src.common import CACHE_DIR, setup_logging
from src.pipeline.detect import Detection, Detector
from src.pipeline.frames import FrameSet, extract_frames

log = logging.getLogger("pipeline.detect_cache")
DET_KW = dict(box_threshold=0.2, shortest_edge=384, longest_edge=640, nms_iou=0.6)


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def cache_path(frames: FrameSet, cfg: dict, cache_dir: Path | None = None) -> Path:
    key = f"{frames.frames[0].path.parent.name}_{config_hash(cfg)}.json"
    return (cache_dir or CACHE_DIR / "detections") / key


def load_cached(frames: FrameSet, cfg: dict, cache_dir: Path | None = None) -> list[list[Detection]] | None:
    p = cache_path(frames, cfg, cache_dir)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if len(data["frames"]) != len(frames):
        return None
    return [[Detection.from_list(r) for r in fr["dets"]] for fr in data["frames"]]


def cached_detections(frames: FrameSet, detector: Detector | None = None, batch_size: int = 4,
                      cache_dir: Path | None = None, frame_indices: list[int] | None = None,
                      det_kw: dict | None = None, deadline: float | None = None) -> list:
    """Detections for every frame (or only `frame_indices`). Reads/writes the cache.

    With `deadline` (absolute time.perf_counter() value) the loop measures its own speed and
    skips frames (growing stride) so that it finishes in time; skipped frames are returned as
    None and nothing is cached in that case.
    """
    det_kw = det_kw or DET_KW
    cfg = detector.config if detector else {**_cfg_without_model(det_kw)}
    hit = load_cached(frames, cfg, cache_dir)
    if hit is not None and frame_indices is None:
        log.info("detections cache hit: %s", cache_path(frames, cfg, cache_dir).name)
        return hit
    if detector is None:
        detector = Detector(**det_kw)
        cfg = detector.config
        hit = load_cached(frames, cfg, cache_dir)
        if hit is not None and frame_indices is None:
            return hit
    idx = list(range(len(frames))) if frame_indices is None else sorted(frame_indices)
    results: list = [None for _ in frames] if deadline else [[] for _ in frames]
    t0, done, skipped = time.perf_counter(), 0, False
    pos = 0
    while pos < len(idx):
        chunk = idx[pos:pos + batch_size]
        outs = detector.detect_batch([frames.frames[j].load() for j in chunk])
        for j, o in zip(chunk, outs):
            results[j] = o
        done += len(chunk)
        pos += len(chunk)
        per_frame = (time.perf_counter() - t0) / done
        if deadline and pos < len(idx):
            remaining_t = deadline - time.perf_counter()
            need = (len(idx) - pos) * per_frame
            if need > remaining_t:
                k = int(need / max(remaining_t, 1e-6)) + 1
                rest = idx[pos::k]
                log.warning("time budget: %.0fs needed, %.0fs left -> stride %d for the remaining %d frames",
                            need, remaining_t, k, len(rest))
                idx, pos, skipped = idx[:pos] + rest, pos, True
        if (done // batch_size) % 10 == 0:
            log.info("detect %d frames, %.2fs/frame", done, per_frame)
    log.info("detected %d frames in %.1fs", done, time.perf_counter() - t0)
    if frame_indices is None and not skipped:
        p = cache_path(frames, cfg, cache_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cfg": cfg, "frames": [
            {"t": f.timestamp_sec, "dets": [d.to_list() for d in r]} for f, r in zip(frames, results)]}))
        tmp.replace(p)
    return results


def _cfg_without_model(det_kw: dict) -> dict:
    """Build the config dict the Detector would produce, without loading weights (cache lookup)."""
    from src.pipeline.detect import MODEL_ID, VOCAB

    return {"model": MODEL_ID, "shortest_edge": det_kw["shortest_edge"], "longest_edge": det_kw["longest_edge"],
            "box_threshold": det_kw["box_threshold"], "nms_iou": det_kw["nms_iou"],
            "phrases": [p.lower() for ps in VOCAB.values() for p in ps]}


_worker_det: Detector | None = None


def _init_worker(threads: int) -> None:
    global _worker_det
    setup_logging("detect_cache")
    _worker_det = Detector(threads=threads, **DET_KW)


def _process(video: str) -> tuple[str, float, int]:
    t0 = time.perf_counter()
    frames = extract_frames(video)
    cached_detections(frames, _worker_det)
    return video, round(time.perf_counter() - t0, 1), len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--all", action="store_true", help="all videos in data/open/videos")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    setup_logging("detect_cache")
    videos = list(args.videos)
    if args.all:
        from eval.labels import index_videos

        videos += [str(p) for p in index_videos().values()]
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    if args.procs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.procs, initializer=_init_worker, initargs=(args.threads,)) as pool:
            for video, dt, n in pool.imap_unordered(_process, videos):
                log.info("done %s: %d frames in %.1fs", Path(video).name, n, dt)
    else:
        _init_worker(args.threads)
        for v in videos:
            video, dt, n = _process(v)
            log.info("done %s: %d frames in %.1fs", Path(video).name, n, dt)


if __name__ == "__main__":
    main()
