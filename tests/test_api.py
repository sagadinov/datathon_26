"""API tests: auth, upload validation and the job lifecycle through the real pipeline subprocess.

Run:  .venv/bin/python -m pytest tests/test_api.py -q       (~2-3 min: several real pipeline runs)
A tiny synthetic video (ffmpeg lavfi testsrc) is used, so no dataset is needed. CACHE_DIR and
LOGS_DIR are redirected to a temporary directory unless already set, so nothing is left in the repo.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="datathon-api-tests-")  # must exist before src.common is imported
os.environ.setdefault("CACHE_DIR", os.path.join(_TMP, "cache"))
os.environ.setdefault("LOGS_DIR", os.path.join(_TMP, "logs"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api import worker as W  # noqa: E402
from src.api.jobs import JobStore  # noqa: E402
from src.api.main import Settings, create_app  # noqa: E402
from src.common import ROOT  # noqa: E402

KEY = "test-key-123"
H = {"X-API-Key": KEY}


def _lavfi(out: Path, *extra: str) -> Path:
    cmd = [W.ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
           "-i", "testsrc=size=320x240:rate=10:duration=2", "-pix_fmt", "yuv420p", *extra, str(out)]
    subprocess.run(cmd, check=True, timeout=60)
    return out


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory) -> Path:
    return _lavfi(tmp_path_factory.mktemp("video") / "tiny.mp4")


@pytest.fixture(scope="session")
def image_as_video(tmp_path_factory) -> Path:
    """A single-frame .mov (a picture, really): ffmpeg opens it but 2 fps sampling yields 0 frames."""
    return _lavfi(tmp_path_factory.mktemp("video") / "image_as.mov", "-frames:v", "1")


def make_settings(tmp_path: Path, **over) -> Settings:
    base = dict(api_key=KEY, jobs_dir=tmp_path / "jobs", workers=1, torch_threads=2,
                time_budget_sec=60.0, max_upload_mb=1, static_dir=tmp_path / "no-static")
    base.update(over)
    return Settings(**base)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


def upload(client: TestClient, video: Path, name: str = "tiny.mov") -> str:
    with open(video, "rb") as fh:
        r = client.post("/api/v1/analyze", headers=H, files={"file": (name, fh, "video/quicktime")})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert len(job_id) == 8 and int(job_id, 16) >= 0
    return job_id


def wait_job(client: TestClient, job_id: str, timeout: float = 150.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        body = client.get(f"/api/v1/jobs/{job_id}", headers=H).json()
        if body["status"] != "processing":
            return body
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} still processing after {timeout}s")


def read_status(settings: Settings, job_id: str) -> dict:
    return json.loads((settings.jobs_dir / job_id / "status.json").read_text(encoding="utf-8"))


def wait_pipeline_pid(settings: Settings, job_id: str, timeout: float = 90.0) -> int:
    """Block until the worker has launched the pipeline subprocess for `job_id`; return its pid."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = read_status(settings, job_id)
        if st["status"] != "processing":
            raise AssertionError(f"job finished before the pipeline was launched: {st}")
        if st.get("pipeline_pid"):
            return int(st["pipeline_pid"])
        time.sleep(0.2)
    raise AssertionError("pipeline subprocess was not launched in time")


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "queue": 0, "workers": 1, "running": 0}


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


def test_probe_and_pid_helpers(tmp_path, tiny_video, image_as_video):
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"junk\n")
    err, detail = W.probe_video(junk)
    assert err == W.ERR_UNDECODABLE and "moov atom not found" in detail  # raw detail is for the log only
    assert W.probe_video(image_as_video)[0] == W.ERR_NO_FRAMES
    assert W.probe_video(tiny_video) == (None, "")
    assert W.pid_alive(os.getpid()) and not W.is_our_pipeline(os.getpid(), "x")
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    assert not W.pid_alive(proc.pid)


def test_junk_file_fails_with_clean_error(client):
    r = client.post("/api/v1/analyze", headers=H, files={"file": ("bad.mov", b"junk\n", "video/quicktime")})
    assert r.status_code == 200
    body = wait_job(client, r.json()["job_id"])
    assert body["status"] == "failed" and body["error"] == W.ERR_UNDECODABLE
    assert "/" not in body["error"]  # no server paths in the public message
    assert list(body) == ["job_id", "status", "error"]


def test_image_as_video_fails(client, image_as_video):
    body = wait_job(client, upload(client, image_as_video, "photo.mov"))
    assert body["status"] == "failed" and body["error"] == W.ERR_NO_FRAMES


def test_video_is_done(client, tiny_video, tmp_path):
    job_id = upload(client, tiny_video)
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
    assert status["attempt"] == 1 and status["pipeline_pid"] is None and status["adopted"] is False


def test_pipeline_crash_gives_fallback_done(tmp_path, tiny_video):
    # An invalid --device makes the pipeline CLI exit immediately without writing a result.
    with TestClient(create_app(make_settings(tmp_path, pipeline_device="bogus"))) as c:
        body = wait_job(c, upload(c, tiny_video))
    assert body["status"] == "done" and body["verdict"] == "no_violation" and body["violations"] == []
    assert body["note"] == "pipeline error, see log"


def test_graceful_stop_requeues_job_in_flight(tmp_path, tiny_video):
    # Lifespan shutdown == SIGTERM / Ctrl-C / docker stop while a pipeline is running.
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        job_id = upload(c, tiny_video)
        pid = wait_pipeline_pid(settings, job_id)
    assert not W.pid_alive(pid)  # the pipeline was killed with the server ...
    st = read_status(settings, job_id)
    assert st["status"] == "processing" and st["attempt"] == 1, st  # ... but the job was not finalised
    with TestClient(create_app(settings)) as c:  # restart: picked up again, finishes normally
        body = wait_job(c, job_id)
    assert body["status"] == "done" and "note" not in body, body
    assert read_status(settings, job_id)["attempt"] == 2


def test_orphan_pipeline_is_adopted_after_hard_restart(tmp_path, tiny_video):
    # SIGKILL/OOM of the API leaves the pipeline running with nobody waiting for it: the restarted
    # worker must wait for that process instead of launching a second one for the same job.
    settings = make_settings(tmp_path)
    store = JobStore(settings.jobs_dir)
    job_id = store.new_job()
    shutil.copy(tiny_video, store.input_path(job_id))
    cmd = [sys.executable, "-m", "src.pipeline.run", str(store.input_path(job_id)), "--out",
           str(store.result_path(job_id)), "--device", "cpu", "--time-budget", "60", "--no-vlm"]
    with open(store.log_path(job_id), "a") as logf:
        orphan = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                  env={**os.environ, "TORCH_THREADS": "2"}, start_new_session=True)
    store.write_status(job_id, "processing", created_at=time.time(), attempt=1,
                       pipeline_pid=orphan.pid, pipeline_started_at=time.time())
    try:
        with TestClient(create_app(settings)) as c:
            body = wait_job(c, job_id)
    finally:
        orphan.wait(timeout=120)
    assert body["status"] == "done" and "note" not in body, body
    st = read_status(settings, job_id)
    assert st["adopted"] is True and st["attempt"] == 2 and st["pipeline_pid"] is None
    log = store.log_path(job_id).read_text(encoding="utf-8")
    assert "src.pipeline.run" not in log  # the worker never launched its own pipeline
    assert log.count("wrote ") == 1


def test_worker_survives_vanished_job_dir(tmp_path, tiny_video):
    # Someone deletes jobs/<id> on a live server: the thread must log and carry on, not die.
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        first = upload(c, tiny_video)
        pid = wait_pipeline_pid(settings, first)
        shutil.rmtree(settings.jobs_dir / first)
        t0 = time.time()
        while W.pid_alive(pid) and time.time() - t0 < 120:
            time.sleep(0.5)
        assert c.get(f"/api/v1/jobs/{first}", headers=H).status_code == 404
        body = wait_job(c, upload(c, tiny_video))  # needs the single worker thread to be alive
        assert body["status"] == "done", body
        assert c.get("/health").json() == {"status": "ok", "queue": 0, "workers": 1, "running": 0}
