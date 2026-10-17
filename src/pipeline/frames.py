"""Frame extraction: ffmpeg -> sampled frames (default 2 fps, long side 640) with timestamps.

Frames are cached on disk under cache/frames/<sha1-of-file>_<fps>_<size>/ so repeated
runs (eval, API retries) cost nothing. ffmpeg is resolved from $FFMPEG_BIN, then PATH,
then the binary bundled with imageio-ffmpeg, so the code works on login/compute nodes
and inside Docker without `module load`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.common import CACHE_DIR

log = logging.getLogger("pipeline.frames")


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_sec: float


@dataclass
class Frame:
    index: int
    timestamp_sec: float
    path: Path

    def load(self) -> np.ndarray:
        """Return the frame as a BGR uint8 array."""
        img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read frame {self.path}")
        return img


@dataclass
class FrameSet:
    meta: VideoMeta
    sample_fps: float
    long_side: int
    frames: list[Frame] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def nearest(self, t_sec: float) -> Frame:
        """Frame whose timestamp is closest to `t_sec`."""
        return min(self.frames, key=lambda f: abs(f.timestamp_sec - t_sec))


def ffmpeg_bin() -> str:
    """Locate an ffmpeg executable (env -> PATH -> imageio-ffmpeg bundle)."""
    cand = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if cand:
        return cand
    import imageio_ffmpeg  # bundled static binary, BSD-2 licence

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: str | Path) -> VideoMeta:
    """Read basic stream metadata with OpenCV (no ffprobe dependency)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    duration = n_frames / fps if n_frames > 0 else 0.0
    return VideoMeta(str(path), width, height, fps, n_frames, round(duration, 3))


def file_sha1(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _scale_expr(long_side: int) -> str:
    # Scale so that max(w, h) == long_side, keep aspect, even dimensions.
    return (
        f"scale='if(gte(iw,ih),{long_side},-2)':'if(gte(iw,ih),-2,{long_side})'"
    )


def extract_frames(
    video_path: str | Path,
    sample_fps: float = 2.0,
    long_side: int = 640,
    cache_dir: Path | None = None,
    jpeg_quality: int = 3,
    threads: int = 4,
    force: bool = False,
) -> FrameSet:
    """Sample `video_path` at `sample_fps`, resize to `long_side`, cache as JPEGs.

    Output frame n of the ffmpeg `fps` filter represents time n / sample_fps, so the
    timestamp error is bounded by half a source-frame interval (~17 ms at 30 fps).
    """
    video_path = Path(video_path)
    cache_dir = Path(cache_dir or CACHE_DIR / "frames")
    key = f"{file_sha1(video_path)[:16]}_fps{sample_fps:g}_s{long_side}"
    out_dir = cache_dir / key
    meta_file = out_dir / "meta.json"

    if meta_file.exists() and not force:
        data = json.loads(meta_file.read_text())
        meta = VideoMeta(**data["meta"])
        frames = [Frame(i, t, out_dir / name) for i, (t, name) in enumerate(data["frames"])]
        if all(f.path.exists() for f in frames):
            log.info("frames cache hit: %s (%d frames)", key, len(frames))
            return FrameSet(meta, sample_fps, long_side, frames)

    meta = probe_video(video_path)
    tmp_dir = out_dir.with_suffix(".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-threads", str(threads), "-i", str(video_path),
        "-vf", f"fps={sample_fps:g},{_scale_expr(long_side)}",
        "-q:v", str(jpeg_quality), "-start_number", "0",
        str(tmp_dir / "f_%05d.jpg"),
    ]
    log.info("ffmpeg extract %s -> %s", video_path.name, key)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[:500]}")

    names = sorted(p.name for p in tmp_dir.glob("f_*.jpg"))
    frames_meta = [(round(i / sample_fps, 3), name) for i, name in enumerate(names)]
    (tmp_dir / "meta.json").write_text(
        json.dumps({"meta": meta.__dict__, "frames": frames_meta}, ensure_ascii=False)
    )
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    frames = [Frame(i, t, out_dir / name) for i, (t, name) in enumerate(frames_meta)]
    log.info("extracted %d frames (%.1fs video)", len(frames), meta.duration_sec)
    return FrameSet(meta, sample_fps, long_side, frames)


if __name__ == "__main__":  # quick manual check
    import sys
    import time

    from src.common import setup_logging

    setup_logging("pipeline")
    t0 = time.perf_counter()
    fs = extract_frames(sys.argv[1])
    print(f"{fs.meta} -> {len(fs)} frames in {time.perf_counter() - t0:.2f}s")
    for f in list(fs)[:3] + list(fs)[-2:]:
        img = f.load()
        print(f"  #{f.index} t={f.timestamp_sec}s {img.shape[1]}x{img.shape[0]} {f.path.name}")
