"""Build robustness variants of the open dataset with ffmpeg (CLAUDE.md section 7, last paragraph).

    .venv/bin/python scripts/make_augmented.py [--only 4,20] [--variants hflip,small] [--procs 4]

For every data/open/videos/*.mov the script writes data/aug/<variant>/videos/<same file name>:

    hflip    mirrored image                                        (hflip)
    rot_p10  rotated +10 deg (clockwise), frame size kept, black corners (rotate)
    rot_m10  rotated -10 deg                                        (rotate)
    small    long side 480 px, aspect kept, even dimensions          (scale)

All outputs are constant 30 fps, H.264 (libx264, crf 23, preset veryfast, yuv420p), no audio.
The run is idempotent: an existing output is only re-probed and skipped when its duration still
matches the source (+-0.2 s); otherwise it is re-encoded. Encoding goes to a `.part` temp file
that is renamed on success, so an interrupted run never leaves a truncated final file.

Every output is probed (ffprobe when available, otherwise the container summary of `ffmpeg -i`),
compared with its source and reported in a table; data/aug/manifest.json keeps the numbers.
ffmpeg is resolved from $FFMPEG_BIN, PATH or the imageio-ffmpeg bundle (same as the pipeline);
ffprobe from $FFPROBE_BIN, PATH, next to ffmpeg, or the cluster `module load FFmpeg` directory.
Exit code is 1 when any file failed to encode or failed the duration check.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabulate import tabulate  # noqa: E402

from eval.labels import VIDEOS_DIR, video_number  # noqa: E402
from src.common import ROOT, setup_logging  # noqa: E402

log = setup_logging("make_augmented")

AUG_DIR = ROOT / "data" / "aug"
# Where EasyBuild installs the cluster module FFmpeg/<version>; only ffprobe is taken from there.
MODULE_FFPROBE_GLOB = "/shared/opt/easybuild/software/FFmpeg/*/bin/ffprobe"
OUT_FPS = 30
ENCODE = ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p"]


def scale_long_side(long_side: int) -> str:
    """max(w, h) == long_side, aspect kept, even dims (same expression as src/pipeline/frames.py)."""
    return f"scale='if(gte(iw,ih),{long_side},-2)':'if(gte(iw,ih),-2,{long_side})'"


def variant_filters(small_long_side: int) -> dict[str, str]:
    # `rotate` keeps the input frame size by default; uncovered corners are filled with black.
    return {
        "hflip": "hflip",
        "rot_p10": "rotate=10*PI/180:fillcolor=black",
        "rot_m10": "rotate=-10*PI/180:fillcolor=black",
        "small": scale_long_side(small_long_side),
    }


VARIANT_NAMES = tuple(variant_filters(480))


# --------------------------------------------------------------------------- binaries, probing
def resolve_ffmpeg() -> str:
    cand = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if cand:
        return cand
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def resolve_ffprobe(ffmpeg: str) -> str | None:
    cands = [os.environ.get("FFPROBE_BIN"), shutil.which("ffprobe"), str(Path(ffmpeg).with_name("ffprobe"))]
    cands += sorted(glob.glob(MODULE_FFPROBE_GLOB), reverse=True)
    for c in cands:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


@dataclass
class Probe:
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    size_mb: float


def probe(path: Path, ffmpeg: str, ffprobe: str | None) -> Probe:
    """Duration / size / fps of a video via ffprobe, falling back to the `ffmpeg -i` summary."""
    size_mb = round(path.stat().st_size / 2**20, 2)
    if ffprobe:
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            info = json.loads(proc.stdout)
            st = (info.get("streams") or [{}])[0]
            num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
            fps = round(float(num) / float(den), 3) if float(den or 0) else None
            dur = info.get("format", {}).get("duration")
            return Probe(round(float(dur), 3) if dur else None, st.get("width"), st.get("height"), fps, size_mb)
    text = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    dur = round(int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]), 3) if m else None
    wh = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})[, ]", text)
    fr = re.search(r"([\d.]+) fps", text)
    return Probe(dur, int(wh[1]) if wh else None, int(wh[2]) if wh else None, float(fr[1]) if fr else None, size_mb)


# --------------------------------------------------------------------------- one job
@dataclass
class Ctx:
    ffmpeg: str
    ffprobe: str | None
    out_dir: Path
    filters: dict[str, str]
    threads: int
    tol: float
    force: bool
    dry_run: bool
    src_probe: dict[Path, Probe]


@dataclass
class Row:
    n: int
    video: str
    variant: str
    status: str  # ok | skip | redo | fail | dry
    src_s: float | None = None
    out_s: float | None = None
    d_s: float | None = None
    size_mb: float | None = None
    wh: str = ""
    fps: float | None = None
    enc_s: float | None = None
    note: str = ""


def build_cmd(ctx: Ctx, src: Path, tmp: Path, variant: str) -> list[str]:
    return [ctx.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-filter_threads", str(ctx.threads), "-threads", str(ctx.threads), "-i", str(src),
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"fps={OUT_FPS},{ctx.filters[variant]}",
            "-threads", str(ctx.threads), *ENCODE, "-f", "mov", str(tmp)]


def _fill(row: Row, src_p: Probe, out_p: Probe, tol: float) -> Row:
    row.src_s, row.out_s, row.size_mb, row.fps = src_p.duration, out_p.duration, out_p.size_mb, out_p.fps
    row.wh = f"{out_p.width}x{out_p.height}" if out_p.width else ""
    if src_p.duration is not None and out_p.duration is not None:
        row.d_s = round(out_p.duration - src_p.duration, 2)
    if row.d_s is None or abs(row.d_s) > tol:
        row.status, row.note = "fail", f"duration mismatch (tol {tol:g} s)"
    return row


def process_one(ctx: Ctx, src: Path, variant: str) -> Row:
    n = video_number(src.name) or -1
    row = Row(n=n, video=src.name, variant=variant, status="ok")
    src_p = ctx.src_probe[src]
    dst = ctx.out_dir / variant / "videos" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.unlink(missing_ok=True)

    if dst.exists() and not ctx.force:
        done = _fill(Row(n, src.name, variant, "skip"), src_p, probe(dst, ctx.ffmpeg, ctx.ffprobe), ctx.tol)
        if done.status == "skip":
            return done
        row.status, row.note = "redo", "existing file failed the check, re-encoded"
        log.warning("%s/%s: %s", variant, src.name, row.note)

    cmd = build_cmd(ctx, src, tmp, variant)
    if ctx.dry_run:
        print(" ".join(cmd))
        row.status, row.src_s = "dry", src_p.duration
        return row

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    row.enc_s = round(time.perf_counter() - t0, 1)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = (proc.stderr.strip().splitlines() or ["no stderr"])[-1]
        row.status, row.note, row.src_s = "fail", f"ffmpeg rc={proc.returncode}: {tail[:120]}", src_p.duration
        log.error("%s/%s: %s", variant, src.name, row.note)
        return row
    tmp.replace(dst)
    row = _fill(row, src_p, probe(dst, ctx.ffmpeg, ctx.ffprobe), ctx.tol)
    log.info("%s/%s: %s %.1fs out=%ss d=%ss %sMB", variant, src.name, row.status, row.enc_s,
             row.out_s, row.d_s, row.size_mb)
    return row


# --------------------------------------------------------------------------- reporting
def print_report(rows: list[Row], wall: float, ctx: Ctx) -> None:
    cols = ["n", "variant", "status", "src_s", "out_s", "d_s", "size_mb", "wh", "fps", "enc_s", "note"]
    rows = sorted(rows, key=lambda r: (r.n, r.variant))
    print(tabulate([[getattr(r, c) for c in cols] for r in rows], headers=cols, tablefmt="simple"))

    summary = []
    for variant in ctx.filters:
        vr = [r for r in rows if r.variant == variant]
        if not vr:
            continue
        deltas = [abs(r.d_s) for r in vr if r.d_s is not None]
        enc = [r.enc_s for r in vr if r.enc_s is not None]
        summary.append([variant, len(vr), sum(r.status in ("ok", "redo") for r in vr),
                        sum(r.status == "skip" for r in vr), sum(r.status == "fail" for r in vr),
                        round(sum(r.size_mb or 0 for r in vr), 1), round(max(deltas), 2) if deltas else None,
                        round(sum(enc) / len(enc), 1) if enc else None])
    print("\nSUMMARY")
    print(tabulate(summary, headers=["variant", "files", "encoded", "skipped", "failed", "total_mb",
                                     "max_abs_d_s", "mean_enc_s"], tablefmt="simple"))
    total_mb = round(sum(r.size_mb or 0 for r in rows), 1)
    print(f"\ntotal {len(rows)} files, {total_mb} MB, wall {wall:.1f}s, {ctx.threads} threads per ffmpeg\n"
          f"ffmpeg={ctx.ffmpeg}\nffprobe={ctx.ffprobe or 'none (parsed ffmpeg -i output)'}")


def write_manifest(rows: list[Row], ctx: Ctx) -> Path:
    path = ctx.out_dir / "manifest.json"
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        data = {}
    files = data.setdefault("files", {})
    for r in rows:
        if r.status in ("ok", "skip", "redo"):
            files.setdefault(r.variant, {})[r.video] = {k: v for k, v in asdict(r).items()
                                                        if k in ("n", "src_s", "out_s", "d_s", "size_mb", "wh", "fps")}
    data.update({"generated_at": datetime.now().isoformat(timespec="seconds"), "ffmpeg": ctx.ffmpeg,
                 "ffprobe": ctx.ffprobe, "encode": ENCODE + ["fps", str(OUT_FPS), "-an"],
                 "filters": ctx.filters, "source_dir": str(VIDEOS_DIR), "tolerance_sec": ctx.tol})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return path


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Create hflip / +-10 deg / downscaled copies of the dataset videos.")
    ap.add_argument("--src", default=str(VIDEOS_DIR), help="source folder with X-N.mov files")
    ap.add_argument("--out", default=str(AUG_DIR), help="output root: <out>/<variant>/videos/<name>")
    ap.add_argument("--variants", default=",".join(VARIANT_NAMES), help=f"comma-separated subset of {VARIANT_NAMES}")
    ap.add_argument("--only", default=None, help="comma-separated video numbers, e.g. 4,20")
    ap.add_argument("--procs", type=int, default=4, help="parallel ffmpeg processes")
    ap.add_argument("--threads", type=int, default=2, help="decoder/filter/encoder threads per ffmpeg")
    ap.add_argument("--small-long-side", type=int, default=480, help="long side of the `small` variant")
    ap.add_argument("--tol", type=float, default=0.2, help="allowed |duration difference| in seconds")
    ap.add_argument("--force", action="store_true", help="re-encode outputs that already exist")
    ap.add_argument("--dry-run", action="store_true", help="print the ffmpeg commands and exit")
    args = ap.parse_args()

    filters = variant_filters(args.small_long_side)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in filters]
    if unknown or not variants:
        sys.exit(f"unknown variants {unknown}; choose from {VARIANT_NAMES}")
    filters = {v: filters[v] for v in variants}

    videos = sorted((p for p in Path(args.src).glob("*.mov") if video_number(p.name) is not None),
                    key=lambda p: video_number(p.name))
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        videos = [p for p in videos if video_number(p.name) in wanted]
    if not videos:
        sys.exit(f"no videos matched in {args.src}")

    ffmpeg = resolve_ffmpeg()
    ffprobe = resolve_ffprobe(ffmpeg)
    log.info("%d videos x %d variants -> %s (procs=%d threads=%d) ffmpeg=%s ffprobe=%s",
             len(videos), len(filters), args.out, args.procs, args.threads, ffmpeg, ffprobe)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.procs)) as pool:
        src_probe = dict(zip(videos, pool.map(lambda p: probe(p, ffmpeg, ffprobe), videos)))
        ctx = Ctx(ffmpeg, ffprobe, Path(args.out), filters, args.threads, args.tol, args.force, args.dry_run, src_probe)
        jobs = [(src, variant) for src in videos for variant in filters]
        rows = list(pool.map(lambda j: process_one(ctx, *j), jobs))
    wall = time.perf_counter() - t0

    print_report(rows, wall, ctx)
    if not args.dry_run:
        print(f"manifest: {write_manifest(rows, ctx).relative_to(ROOT)}")
    failed = [r for r in rows if r.status == "fail"]
    log.info("done: %d files, %d failed, %.1fs", len(rows), len(failed), wall)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
