"""API tests: auth, upload validation and the job lifecycle through the real pipeline subprocess.

Run:  .venv/bin/python -m pytest tests/test_api.py -q
A tiny synthetic video (ffmpeg lavfi testsrc) is used, so no dataset is needed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import Settings, create_app
from src.api.worker import ffmpeg_bin

KEY = "test-key-123"
H = {"X-API-Key": KEY}


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("video") / "tiny.mp4"
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
           "-i", "testsrc=size=320x240:rate=10:duration=2", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True, timeout=60)
    return out


def make_settings(tmp_path: Path, **over) -> Settings:
    base = dict(api_key=KEY, jobs_dir=tmp_path / "jobs", workers=1, torch_threads=2,
                time_budget_sec=60.0, max_upload_mb=1, static_dir=tmp_path / "no-static")
    base.update(over)
    return Settings(**base)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


def wait_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        body = client.get(f"/api/v1/jobs/{job_id}", headers=H).json()
        if body["status"] != "processing":
            return body
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} still processing after {timeout}s")


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["workers"] == 1


def test_auth_required(client, tiny_video):
    assert client.get("/api/v1/jobs/deadbeef").status_code == 401
    r = client.get("/api/v1/jobs/deadbeef", headers={"X-API-Key": "nope"})
    assert r.status_code == 401 and r.json() == {"detail": "invalid api key"}
    with open(tiny_video, "rb") as fh:
        assert client.post("/api/v1/analyze", files={"file": fh}).status_code == 401


def test_unknown_and_malformed_job(client):
    assert client.get("/api/v1/jobs/deadbeef", headers=H).status_code == 404
    assert client.get("/api/v1/jobs/not-a-job-id", headers=H).status_code == 404


def test_missing_file_field_is_400(client, tiny_video):
    assert client.post("/api/v1/analyze", headers=H).status_code == 400
    with open(tiny_video, "rb") as fh:
        assert client.post("/api/v1/analyze", headers=H, files={"video": fh}).status_code == 400
    assert client.post("/api/v1/analyze", headers=H, data={"file": "not a file"}).status_code == 400


def test_oversize_is_413(client, tmp_path):
    big = b"\0" * int(1.5 * 2**20)  # max_upload_mb=1 in make_settings
    r = client.post("/api/v1/analyze", headers=H, files={"file": ("big.mov", big, "video/quicktime")})
    assert r.status_code == 413
    assert list((tmp_path / "jobs").iterdir()) == []  # no job directory left behind


def test_junk_file_fails(client):
    r = client.post("/api/v1/analyze", headers=H, files={"file": ("bad.mov", b"junk\n", "video/quicktime")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert len(job_id) == 8 and int(job_id, 16) >= 0
    body = wait_job(client, job_id)
    assert body["status"] == "failed" and body["error"]
    assert list(body) == ["job_id", "status", "error"]


def test_video_is_done(client, tiny_video, tmp_path):
    with open(tiny_video, "rb") as fh:
        r = client.post("/api/v1/analyze", headers=H,
                        files={"file": ("tiny.mov", fh, "video/quicktime")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    body = wait_job(client, job_id)
    assert body["status"] == "done", body
    assert list(body)[:4] == ["job_id", "status", "verdict", "violations"]
    assert body["verdict"] in ("violation", "no_violation") and isinstance(body["violations"], list)
    assert "note" not in body  # a real pipeline result, not the crash fallback
    jd = tmp_path / "jobs" / job_id
    for name in ("input.mov", "result.json", "pipeline.log", "status.json"):
        assert (jd / name).is_file(), name
    status = json.loads((jd / "status.json").read_text())
    assert status["status"] == "done" and status["result_file"] == "result.json"


def test_pipeline_crash_gives_fallback_done(tmp_path, tiny_video):
    # An invalid --device makes the pipeline CLI exit immediately without writing a result.
    with TestClient(create_app(make_settings(tmp_path, pipeline_device="bogus"))) as c:
        with open(tiny_video, "rb") as fh:
            job_id = c.post("/api/v1/analyze", headers=H, files={"file": fh}).json()["job_id"]
        body = wait_job(c, job_id)
    assert body["status"] == "done" and body["verdict"] == "no_violation" and body["violations"] == []
    assert body["note"] == "pipeline error, see log"


def test_requeue_after_restart(tmp_path, tiny_video):
    # A job left in `processing` by a previous process is picked up at startup.
    settings = make_settings(tmp_path)
    jd = settings.jobs_dir / "0badf00d"
    jd.mkdir(parents=True)
    shutil.copy(tiny_video, jd / "input.mov")
    (jd / "status.json").write_text(json.dumps(
        {"job_id": "0badf00d", "status": "processing", "created_at": time.time()}))
    with TestClient(create_app(settings)) as c:
        body = wait_job(c, "0badf00d")
    assert body["status"] == "done"
