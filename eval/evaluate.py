"""Dataset evaluation over data/open: verdict, timecode, type metrics and per-video timings.

    python -m eval.evaluate [--device cpu] [--no-vlm] [--jobs 8] [--only 4,20,18] [--limit N]
                            [--videos-dir data/aug/hflip/videos]   # robustness variants, same labels

Writes eval/results/<timestamp>_<git-sha>.json and prints a per-video table + summary.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from tabulate import tabulate

from eval.labels import VIDEOS_DIR, Label, load_labels
from src.common import ROOT
from src.rules_catalog import type_id

RESULTS_DIR = ROOT / "eval" / "results"
TOL = (3.0, 5.0)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "nogit"


def _run_one(args: tuple) -> dict:
    """Worker: analyse one video, return a compact row (no base64 images)."""
    video, device, use_vlm = args
    from src.pipeline.run import analyze

    t0 = time.perf_counter()
    try:
        res = analyze(video, device=device, use_vlm=use_vlm)
        err = None
    except Exception:  # keep the eval going; a crash counts as no_violation
        res, err = {"verdict": "no_violation", "violations": []}, traceback.format_exc(limit=3)
    events = [
        {"type": v.get("type"), "type_id": type_id(v.get("type", "")), "t": v.get("timecode_sec"),
         "score": v.get("score"), "n_frames": len(v.get("evidence_frames", []))}
        for v in res.get("violations", [])
    ]
    return {"verdict": res.get("verdict"), "events": events, "seconds": round(time.perf_counter() - t0, 2),
            "timings": res.get("_debug", {}).get("timings", {}), "error": err}


def score_row(label: Label, pred: dict) -> dict:
    pv = pred["verdict"] == "violation"
    row = {"n": label.n, "gt": "V" if label.violation else "-", "pred": "V" if pv else "-",
           "gt_t": label.t_sec, "gt_type": label.type_id, "pred_t": None, "pred_type": None,
           "dt": None, "hit3": None, "hit5": None, "type_ok": None, "n_events": len(pred["events"]),
           "sec": pred["seconds"], "err": bool(pred["error"])}
    row["outcome"] = {(True, True): "TP", (True, False): "FN", (False, True): "FP", (False, False): "TN"}[
        (label.violation, pv)]
    if label.violation and pv and pred["events"] and label.t_sec is not None:
        ev = min(pred["events"], key=lambda e: abs((e["t"] or 0) - label.t_sec))
        row["pred_t"], row["pred_type"] = ev["t"], ev["type_id"]
        row["dt"] = round(ev["t"] - label.t_sec, 1)
        row["hit3"], row["hit5"] = abs(row["dt"]) <= TOL[0], abs(row["dt"]) <= TOL[1]
        row["type_ok"] = ev["type_id"] == label.type_id
    return row


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tp = sum(r["outcome"] == "TP" for r in rows)
    fn = sum(r["outcome"] == "FN" for r in rows)
    fp = sum(r["outcome"] == "FP" for r in rows)
    tn = sum(r["outcome"] == "TN" for r in rows)
    scored = [r for r in rows if r["dt"] is not None]
    secs = [r["sec"] for r in rows]
    return {
        "videos": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": round((tp + tn) / n, 3) if n else None,
        "recall_violation": round(tp / (tp + fn), 3) if tp + fn else None,
        "precision_violation": round(tp / (tp + fp), 3) if tp + fp else None,
        "fp_on_clean": fp,
        "fn_videos": [r["n"] for r in rows if r["outcome"] == "FN"],
        "fp_videos": [r["n"] for r in rows if r["outcome"] == "FP"],
        "timecode_scored": len(scored),
        "timecode_mae": round(statistics.mean(abs(r["dt"]) for r in scored), 2) if scored else None,
        "timecode_median_abs": round(statistics.median(abs(r["dt"]) for r in scored), 2) if scored else None,
        "hit_3s": round(sum(r["hit3"] for r in scored) / len(scored), 3) if scored else None,
        "hit_5s": round(sum(r["hit5"] for r in scored) / len(scored), 3) if scored else None,
        "type_acc_on_tp": round(sum(bool(r["type_ok"]) for r in scored) / len(scored), 3) if scored else None,
        "sec_mean": round(statistics.mean(secs), 1) if secs else None,
        "sec_max": round(max(secs), 1) if secs else None,
        "errors": sum(r["err"] for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--only", default=None, help="comma-separated video numbers, e.g. 4,20,18")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="free-text tag stored in the result file")
    ap.add_argument("--videos-dir", default=None,
                    help="folder with X-N.mov files instead of data/open/videos (e.g. data/aug/hflip/videos); "
                         "labels still come from open_labels.xlsx, matched by number N")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir) if args.videos_dir else VIDEOS_DIR
    if not videos_dir.is_absolute() and not videos_dir.exists():
        videos_dir = ROOT / videos_dir
    if not videos_dir.is_dir():
        sys.exit(f"videos dir not found: {videos_dir}")
    labels = [l for l in load_labels(videos_dir=videos_dir) if l.video is not None]
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        labels = [l for l in labels if l.n in wanted]
    if args.limit:
        labels = labels[: args.limit]
    if not labels:
        sys.exit("no videos matched")
    print(f"evaluating {len(labels)} videos from {videos_dir}, device={args.device}, vlm={not args.no_vlm}, "
          f"jobs={args.jobs}")

    tasks = [(str(l.video), args.device, not args.no_vlm) for l in labels]
    t0 = time.perf_counter()
    if args.jobs > 1:
        os.environ.setdefault("TORCH_THREADS", str(max(1, (os.cpu_count() or 8) // args.jobs // 2)))
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            preds = list(ex.map(_run_one, tasks))
    else:
        preds = [_run_one(t) for t in tasks]
    wall = round(time.perf_counter() - t0, 1)

    rows = [score_row(l, p) for l, p in zip(labels, preds)]
    cols = ["n", "gt", "pred", "outcome", "gt_t", "pred_t", "dt", "hit3", "gt_type", "pred_type", "type_ok",
            "n_events", "sec", "err"]
    print(tabulate([[r[c] for c in cols] for r in rows], headers=cols, tablefmt="simple"))
    summary = summarize(rows)
    summary["wall_sec"] = wall
    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{git_sha()}.json"
    out.write_text(json.dumps({
        "config": {"device": args.device, "use_vlm": not args.no_vlm, "jobs": args.jobs, "tag": args.tag,
                   "git_sha": git_sha(), "tolerances_sec": TOL, "videos_dir": str(videos_dir)},
        "summary": summary, "rows": rows,
        "predictions": {l.n: p for l, p in zip(labels, preds)},
    }, ensure_ascii=False, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
