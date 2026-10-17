"""Background workers: WORKERS threads consume a queue of job ids and run the pipeline CLI.

Per job: ffmpeg probe (two frames at 2 fps) -> `python -m src.pipeline.run` subprocess (cwd =
repo root, env TORCH_THREADS, own process group, output -> jobs/<id>/pipeline.log, hard timeout
TIME_BUDGET_SEC + 60 s) -> result.json: done | meaningful partial: done | undecodable or no
frames: failed + short Russian error | crash or timeout: done with the empty fallback result.
Restarts: stop() kills running pipelines and leaves their jobs `processing` for the next start
to re-queue; after a hard kill the orphaned pipeline (pid in status.json) is adopted, not re-run.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from src.api.jobs import JobStore, fallback_result, read_json, valid_result

GRACE_SEC = 60.0  # hard kill this long after the pipeline's own time budget
MAX_ATTEMPTS = 3  # runs of one job across restarts before giving up with the fallback result
PROBE_TIMEOUT_SEC = 30.0
# Public error texts for `failed` (UI language is Russian); ffmpeg's raw stderr, which contains
# server paths, goes to jobs/<id>/pipeline.log only.
ERR_UNDECODABLE = "Файл повреждён или не является видео: не удалось декодировать ни одного кадра"
ERR_NO_FRAMES = "Файл не содержит видеопотока"
ERR_PROBE_TIMEOUT = "Не удалось проверить видеофайл: превышено время декодирования"
ERR_NO_INPUT = "Загруженный файл не найден на сервере"
DECODE_ERROR_MARKERS = ("cannot open video", "ffmpeg failed", "Invalid data found when processing input",
                        "moov atom not found")
NO_FRAMES_MARKERS = ("extracted 0 frames", "(0 frames)")
FRAMES_ONLY_NOTE = "partial: frames only"  # the pipeline's first snapshot: nothing analysed yet


@dataclass(frozen=True)
class Outcome:
    exit_code: int | None = None
    timed_out: bool = False
    interrupted: bool = False  # killed by stop(): leave the job in `processing`
    adopted: bool = False  # finished by a pipeline started before a hard restart


def ffmpeg_bin() -> str:
    """Locate ffmpeg like src.pipeline.frames does (env -> PATH -> imageio-ffmpeg bundle)."""
    cand = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if cand:
        return cand
    import imageio_ffmpeg  # bundled static binary; avoids importing cv2 into the API process

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: Path, timeout: float = PROBE_TIMEOUT_SEC) -> tuple[str | None, str]:
    """Decode the first two frames at 2 fps (what the pipeline samples).

    Returns (error, detail): error is None for a usable video, else a short public message;
    detail is ffmpeg's raw stderr for the job log (may contain server paths).
    """
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
           "-vf", "fps=2", "-frames:v", "2", "-f", "null", "-", "-progress", "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ERR_PROBE_TIMEOUT, f"ffmpeg probe timed out after {timeout:.0f}s"
    except OSError as exc:  # no ffmpeg at all: let the pipeline decide instead of blocking
        return None, f"probe skipped: {exc}"
    detail = proc.stderr.strip()
    if proc.returncode != 0:
        return ERR_UNDECODABLE, detail or f"ffmpeg exit {proc.returncode}"
    counts = [int(m) for m in re.findall(r"^frame=(\d+)", proc.stdout, re.M)]
    if not counts or counts[-1] == 0:
        return ERR_NO_FRAMES, detail or "ffmpeg decoded 0 frames at 2 fps (image or audio-only file?)"
    return None, ""


def pid_alive(pid: int) -> bool:
    """True while `pid` exists and is not a zombie (Linux /proc; kill(0) elsewhere)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False
    except (OSError, IndexError):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def is_our_pipeline(pid: int, input_path: str) -> bool:
    """True when `pid` is alive and its command line is the pipeline CLI for `input_path`."""
    if pid <= 0 or not pid_alive(pid):
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "src.pipeline.run" in cmd and input_path in cmd


def kill_group(pgid: int, proc: subprocess.Popen | None = None) -> None:
    """SIGTERM the pipeline's process group (python + ffmpeg children), SIGKILL after 5 s.

    `proc` (when the group leader is our child) is polled so that the zombie gets reaped.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            dead = proc.poll() is not None if proc is not None else not pid_alive(pgid)
            if dead:
                return
            time.sleep(0.1)


class PipelineWorker:
    """Thread pool that runs queued jobs through the pipeline subprocess."""

    def __init__(
        self,
        store: JobStore,
        *,
        n_workers: int,
        time_budget_sec: float,
        torch_threads: int,
        repo_root: Path,
        device: str = "cpu",
        use_vlm: bool = False,
        python_exe: str | None = None,
        log: logging.Logger | None = None,
    ):
        self.store = store
        self.n_workers = max(1, int(n_workers))
        self.time_budget = float(time_budget_sec)
        self.torch_threads = int(torch_threads)
        self.repo_root = Path(repo_root)
        self.device = device
        self.use_vlm = use_vlm
        self.python = python_exe or sys.executable
        self.log = log or logging.getLogger("api")
        self._queue: queue.Queue[str] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._running: dict[str, int] = {}  # job_id -> process group id of its pipeline
        self._lock = threading.Lock()

    # ----- public ------------------------------------------------------------------------
    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def running(self) -> int:
        with self._lock:
            return len(self._running)

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)
        self.ensure_threads()

    def requeue_unfinished(self) -> int:
        """Put jobs left in `processing` by a previous process back on the queue."""
        ids = self.store.list_processing()
        for job_id in ids:
            self.log.info("job %s: left in processing by the previous run, re-queued", job_id)
            self.enqueue(job_id)
        return len(ids)

    def start(self) -> None:
        self._stop.clear()
        self._threads = [self._spawn(i) for i in range(self.n_workers)]

    def ensure_threads(self) -> int:
        """Respawn worker threads that died (defence in depth); returns the number alive."""
        with self._lock:
            if self._stop.is_set():
                return 0
            for i, t in enumerate(self._threads):
                if not t.is_alive():
                    self.log.error("worker thread %s died, respawning", t.name)
                    self._threads[i] = self._spawn(i)
            return len(self._threads)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the threads and kill running pipelines; their jobs stay `processing` and are
        re-queued by the next start (see _map_outcome)."""
        self._stop.set()
        with self._lock:
            running = dict(self._running)
        for job_id, pgid in running.items():
            self.log.warning("job %s: shutdown, killing pipeline pgid %d (re-queued at next start)", job_id, pgid)
            kill_group(pgid)
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    # ----- internals ---------------------------------------------------------------------
    def _spawn(self, i: int) -> threading.Thread:
        t = threading.Thread(target=self._loop, name=f"worker-{i}", daemon=True)
        t.start()
        return t

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run_job(job_id)
            except Exception:  # never let a worker thread die
                self.log.exception("job %s: unexpected worker error", job_id)
                self._finish_fallback(job_id)
            finally:
                self._queue.task_done()

    def _finish_fallback(self, job_id: str) -> None:
        try:
            self.store.update_status(job_id, status="done", result=fallback_result(), pipeline_pid=None,
                                     pipeline_error="worker exception, see logs/api.log")
        except Exception as exc:  # job directory vanished, disk full, ...: drop the job, keep the thread
            self.log.error("job %s: cannot write status.json either (%s); job dropped", job_id, exc)

    def _run_job(self, job_id: str) -> None:
        t0 = time.perf_counter()
        status = self.store.read_status(job_id) or {}
        attempt = int(status.get("attempt") or 0) + 1
        waited = time.time() - float(status.get("created_at") or time.time())
        self.log.info("job %s start (attempt %d, queue wait %.1fs, queue=%d)", job_id, attempt, waited,
                      self.queue_size)
        inp, result = self.store.input_path(job_id), self.store.result_path(job_id)

        if valid_result(read_json(result)):  # finished right before a restart
            self._finish(job_id, "done", t0, result_file=result.name)
            return
        if attempt > MAX_ATTEMPTS:
            self.log.error("job %s: giving up after %d attempts", job_id, attempt - 1)
            self._finish(job_id, "done", t0, result=fallback_result(),
                         pipeline_error=f"gave up after {attempt - 1} attempts")
            return
        self.store.update_status(job_id, started_at=time.time(), attempt=attempt)
        if not inp.is_file():
            self._finish(job_id, "failed", t0, error=ERR_NO_INPUT)
            return
        if attempt > 1 and self._adopt_orphan(job_id, status, t0):
            return
        error, detail = probe_video(inp)
        if error:
            self._append_log(job_id, f"ffmpeg probe failed: {detail}")
            self._finish(job_id, "failed", t0, error=error, error_detail=detail[:500])
            return
        self._map_outcome(job_id, t0, self._launch(job_id, inp, result))

    def _launch(self, job_id: str, inp: Path, result: Path) -> Outcome:
        cmd = [self.python, "-m", "src.pipeline.run", str(inp), "--out", str(result),
               "--device", self.device, "--time-budget", f"{self.time_budget:g}"]
        if not self.use_vlm:
            cmd.append("--no-vlm")
        env = {**os.environ, "TORCH_THREADS": str(self.torch_threads)}
        hard_timeout = self.time_budget + GRACE_SEC
        timed_out = False
        with open(self.store.log_path(job_id), "a", encoding="utf-8") as logf:
            logf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            logf.flush()
            with self._lock:  # stop() sets the flag before reading _running: no launch slips past it
                if self._stop.is_set():
                    return Outcome(interrupted=True)
                proc = subprocess.Popen(cmd, cwd=self.repo_root, env=env, stdout=logf,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
                self._running[job_id] = proc.pid
            try:  # pid on disk lets the next start adopt the pipeline after a hard kill of the API
                self.store.update_status(job_id, pipeline_pid=proc.pid, pipeline_started_at=time.time())
            except OSError as exc:
                self.log.warning("job %s: cannot record pipeline pid: %s", job_id, exc)
            try:
                proc.wait(timeout=hard_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self.log.error("job %s: pipeline exceeded %.0fs, killing", job_id, hard_timeout)
                kill_group(proc.pid, proc)
            finally:
                with self._lock:
                    self._running.pop(job_id, None)
        return Outcome(proc.returncode, timed_out, interrupted=self._stop.is_set())

    def _adopt_orphan(self, job_id: str, status: dict, t0: float) -> bool:
        """Wait for a pipeline started before a hard restart (pid in status.json) instead of
        running the job a second time. True when the job was finalised (or interrupted) here."""
        pid = int(status.get("pipeline_pid") or 0)
        if not is_our_pipeline(pid, str(self.store.input_path(job_id))):
            return False
        started = float(status.get("pipeline_started_at") or time.time())
        deadline = started + self.time_budget + GRACE_SEC
        self.log.warning("job %s: adopting orphan pipeline pid %d (running for %.0fs)", job_id, pid,
                         time.time() - started)
        with self._lock:
            if self._stop.is_set():
                return True
            self._running[job_id] = pid
        timed_out = False
        try:
            while pid_alive(pid):
                if self._stop.is_set():
                    kill_group(pid)
                    self.log.warning("job %s: interrupted by shutdown, left in `processing` for re-queue", job_id)
                    return True
                if time.time() > deadline:
                    timed_out = True
                    self.log.error("job %s: orphan pipeline exceeded its deadline, killing", job_id)
                    kill_group(pid)
                    break
                time.sleep(0.5)
        finally:
            with self._lock:
                self._running.pop(job_id, None)
        if timed_out or valid_result(read_json(self.store.result_path(job_id))) or self._meaningful_partial(job_id):
            self._map_outcome(job_id, t0, Outcome(None, timed_out, adopted=True))
            return True
        self.log.warning("job %s: orphan pipeline ended without a result, running it again", job_id)
        return False

    def _meaningful_partial(self, job_id: str) -> dict | None:
        """The best-so-far snapshot, unless it is the empty 'frames only' one."""
        part = read_json(self.store.partial_path(job_id))
        return part if valid_result(part) and part.get("note") != FRAMES_ONLY_NOTE else None

    def _map_outcome(self, job_id: str, t0: float, oc: Outcome) -> None:
        result, partial = self.store.result_path(job_id), self.store.partial_path(job_id)
        extra = {"exit_code": oc.exit_code, "timed_out": oc.timed_out, "adopted": oc.adopted}
        if valid_result(read_json(result)):
            self._finish(job_id, "done", t0, result_file=result.name, **extra)
            return
        if oc.interrupted:
            self.log.warning("job %s: interrupted by shutdown, left in `processing` for re-queue", job_id)
            return
        if self._meaningful_partial(job_id):
            self.log.warning("job %s: serving partial result (exit=%s timeout=%s)", job_id, oc.exit_code, oc.timed_out)
            self._finish(job_id, "done", t0, result_file=partial.name, partial=True, **extra)
            return
        tail = self._log_tail(job_id)
        if not oc.timed_out and oc.exit_code != 0:
            if any(m in tail for m in NO_FRAMES_MARKERS):
                self._finish(job_id, "failed", t0, error=ERR_NO_FRAMES, **extra)
                return
            if any(m in tail for m in DECODE_ERROR_MARKERS):
                self._finish(job_id, "failed", t0, error=ERR_UNDECODABLE, **extra)
                return
        reason = (f"timeout after {self.time_budget + GRACE_SEC:.0f}s" if oc.timed_out
                  else f"exit code {oc.exit_code}" if oc.exit_code is not None else "ended without result")
        self.log.error("job %s: pipeline %s without result, see %s", job_id, reason, self.store.log_path(job_id))
        self._finish(job_id, "done", t0, result=fallback_result(), pipeline_error=reason, **extra)

    def _finish(self, job_id: str, status: str, t0: float, **fields) -> None:
        elapsed = time.perf_counter() - t0
        data = self.store.update_status(job_id, status=status, finished_at=time.time(),
                                        elapsed_sec=round(elapsed, 2), pipeline_pid=None, **fields)
        total = time.time() - float(data.get("created_at") or time.time())
        self.log.info("job %s %s in %.1fs (since upload %.1fs)%s", job_id, status, elapsed, total,
                      f" error={fields['error']}" if fields.get("error") else "")

    def _append_log(self, job_id: str, text: str) -> None:
        try:
            with open(self.store.log_path(job_id), "a", encoding="utf-8") as fh:
                fh.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
        except OSError:
            pass

    def _log_tail(self, job_id: str, n_bytes: int = 4000) -> str:
        try:
            with open(self.store.log_path(job_id), "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - n_bytes))
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return ""
