"""Job persistence: jobs/<job_id>/{input.mov, status.json, result.json, pipeline.log}.

status.json is the source of truth for the API (status, timings, error, which result file
to serve). result.json is written by the pipeline subprocess; result.json.partial.json is
its best-so-far snapshot while it is still running. Every write here is atomic
(tmp file + os.replace) so a crash never leaves a half-written status behind.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

JOB_ID_RE = re.compile(r"^[0-9a-f]{8}$")
VERDICTS = ("violation", "no_violation")
FALLBACK_NOTE = "pipeline error, see log"


def atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    """Parse a JSON file; None when it is missing, unreadable or not (yet) valid JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def valid_result(data: Any) -> bool:
    """True when `data` has the mandatory pipeline/API result shape."""
    return (
        isinstance(data, dict)
        and data.get("verdict") in VERDICTS
        and isinstance(data.get("violations"), list)
    )


def fallback_result(note: str = FALLBACK_NOTE) -> dict:
    """Result served when the pipeline produced nothing usable (crash/timeout)."""
    return {"verdict": "no_violation", "violations": [], "note": note}


class JobStore:
    """Filesystem-backed job registry under `jobs_dir`."""

    def __init__(self, jobs_dir: Path | str):
        self.root = Path(jobs_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ----- paths -------------------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def input_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "input.mov"

    def result_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "result.json"

    def partial_path(self, job_id: str) -> Path:
        # The pipeline writes <out>.partial.json next to --out while it is still working.
        return self.job_dir(job_id) / "result.json.partial.json"

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "pipeline.log"

    def status_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "status.json"

    # ----- lifecycle ---------------------------------------------------------------------
    def new_job(self) -> str:
        """Allocate a fresh 8-hex-char job id and create its directory."""
        for _ in range(100):
            job_id = secrets.token_hex(4)
            try:
                self.job_dir(job_id).mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return job_id
        raise RuntimeError("cannot allocate a unique job id")

    def write_status(self, job_id: str, status: str, **fields: Any) -> dict:
        data = {"job_id": job_id, "status": status, **fields, "updated_at": time.time()}
        atomic_write_json(self.status_path(job_id), data)
        return data

    def update_status(self, job_id: str, **fields: Any) -> dict:
        """Merge `fields` into the stored status (status itself may be among them)."""
        data = self.read_status(job_id) or {"job_id": job_id, "status": "processing"}
        data.update(fields)
        data["updated_at"] = time.time()
        atomic_write_json(self.status_path(job_id), data)
        return data

    def read_status(self, job_id: str) -> dict | None:
        """Stored status dict, or None for unknown / malformed ids (never touches the FS then)."""
        if not job_id or not JOB_ID_RE.match(job_id):
            return None
        data = read_json(self.status_path(job_id))
        return data if isinstance(data, dict) and data.get("status") else None

    def list_processing(self) -> list[str]:
        """Ids of jobs still marked `processing` (to re-queue after a restart), oldest first."""
        found: list[tuple[float, str]] = []
        for d in self.root.iterdir():
            if not (d.is_dir() and JOB_ID_RE.match(d.name)):
                continue
            st = self.read_status(d.name)
            if st and st.get("status") == "processing":
                found.append((float(st.get("created_at") or 0.0), d.name))
        return [job_id for _, job_id in sorted(found)]

    # ----- results -----------------------------------------------------------------------
    def load_result(self, job_id: str, status: dict) -> dict:
        """The result dict for a `done` job: from its result file, else inline, else fallback."""
        name = status.get("result_file")
        if name:
            data = read_json(self.job_dir(job_id) / name)
            if valid_result(data):
                return data
        inline = status.get("result")
        if valid_result(inline):
            return inline
        return fallback_result("result file missing, see log")

    def api_response(self, job_id: str, status: dict) -> dict:
        """GET /api/v1/jobs/{id} body; key order is job_id, status, verdict, violations."""
        body: dict[str, Any] = {"job_id": job_id, "status": status["status"]}
        if status["status"] == "done":
            result = self.load_result(job_id, status)
            body["verdict"] = result["verdict"]
            body["violations"] = result["violations"]
            if result.get("note"):
                body["note"] = result["note"]
        elif status["status"] == "failed":
            body["error"] = status.get("error") or "unknown error"
        return body
