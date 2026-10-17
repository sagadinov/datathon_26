"""Background workers: WORKERS threads consume a queue of job ids and run the pipeline CLI.

Per job: ffmpeg decodability probe (one frame) -> `python -m src.pipeline.run` as a
subprocess (cwd = repo root, env TORCH_THREADS, stdout/stderr -> jobs/<id>/pipeline.log)
with a hard timeout of TIME_BUDGET_SEC + 60 s -> outcome mapping:
  result.json              -> done
  result.json.partial.json -> done (best-so-far)
  undecodable input        -> failed + error text
  crash / timeout          -> done with an empty no_violation result + note (never failed)
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.api.jobs import JobStore, fallback_result, read_json, valid_result

GRACE_SEC = 60.0  # hard kill this long after the pipeline's own time budget
DECODE_ERROR_MARKERS = (
    "cannot open video",
    "ffmpeg failed",
    "Invalid data found when processing input",
    "moov atom not found",
)


def ffmpeg_bin() -> str:
    """Locate ffmpeg like src.pipeline.frames does (env -> PATH -> imageio-ffmpeg bundle)."""
    cand = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if cand:
        return cand
    import imageio_ffmpeg  # bundled static binary; avoids importing cv2 into the API process

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: Path, timeout: float = 30.0) -> tuple[bool, str]:
    """Decode a single frame with ffmpeg. Returns (ok, error_detail)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", str(path), "-frames:v", "1", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg probe timed out"
    except OSError as exc:  # no ffmpeg at all: let the pipeline decide instead of blocking
        return True, f"probe skipped: {exc}"
    if proc.returncode == 0:
        return True, ""
    lines = [ln.strip() for ln in proc.stderr.strip().splitlines() if ln.strip()]
    return False, " | ".join(lines[-3:])[:300] or f"ffmpeg exit {proc.returncode}"


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
        self._running: dict[str, subprocess.Popen] = {}
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

    def requeue_unfinished(self) -> int:
        """Put jobs left in `processing` by a previous process back on the queue."""
        ids = self.store.list_processing()
        for job_id in ids:
            self.enqueue(job_id)
        return len(ids)

    def start(self) -> None:
        self._stop.clear()
        for i in range(self.n_workers):
            t = threading.Thread(target=self._loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the threads; kill running pipelines (their jobs are re-queued at next start)."""
        self._stop.set()
        with self._lock:
            procs = list(self._running.values())
        for proc in procs:
            self._kill(proc)
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    # ----- internals ---------------------------------------------------------------------
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
                self.store.update_status(job_id, status="done", result=fallback_result(),
                                         pipeline_error="worker exception, see logs/api.log")
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        t0 = time.perf_counter()
        status = self.store.read_status(job_id) or {}
        waited = time.time() - float(status.get("created_at") or time.time())
        self.log.info("job %s start (queue wait %.1fs, queue=%d)", job_id, waited, self.queue_size)
        self.store.update_status(job_id, started_at=time.time())
        inp, result = self.store.input_path(job_id), self.store.result_path(job_id)

        if valid_result(read_json(result)):  # finished right before a restart
            self._finish(job_id, "done", t0, result_file=result.name)
            return
        if not inp.is_file():
            self._finish(job_id, "failed", t0, error="Входной файл не найден (input.mov)")
            return
        ok, detail = probe_video(inp)
        if not ok:
            self._finish(job_id, "failed", t0,
                         error=f"Не удалось декодировать видео: {detail or 'файл не является видео'}")
            return
        exit_code, timed_out = self._launch(job_id, inp, result)
        self._map_outcome(job_id, t0, exit_code, timed_out)

    def _launch(self, job_id: str, inp: Path, result: Path) -> tuple[int | None, bool]:
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
            proc = subprocess.Popen(cmd, cwd=self.repo_root, env=env, stdout=logf,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            with self._lock:
                self._running[job_id] = proc
            try:
                proc.wait(timeout=hard_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self.log.error("job %s: pipeline exceeded %.0fs, killing", job_id, hard_timeout)
                self._kill(proc)
            finally:
                with self._lock:
                    self._running.pop(job_id, None)
        return proc.returncode, timed_out

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        """SIGTERM the pipeline's process group (python + ffmpeg children), SIGKILL after 5 s."""
        for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def _map_outcome(self, job_id: str, t0: float, exit_code: int | None, timed_out: bool) -> None:
        result, partial = self.store.result_path(job_id), self.store.partial_path(job_id)
        if valid_result(read_json(result)):
            self._finish(job_id, "done", t0, exit_code=exit_code, result_file=result.name)
            return
        if valid_result(read_json(partial)):
            self.log.warning("job %s: serving partial result (exit=%s timeout=%s)", job_id, exit_code, timed_out)
            self._finish(job_id, "done", t0, exit_code=exit_code, timed_out=timed_out,
                         result_file=partial.name, partial=True)
            return
        tail = self._log_tail(job_id)
        if not timed_out and exit_code != 0 and any(m in tail for m in DECODE_ERROR_MARKERS):
            self._finish(job_id, "failed", t0, exit_code=exit_code,
                         error="Не удалось открыть видео: файл повреждён или не является видео")
            return
        reason = f"timeout after {self.time_budget + GRACE_SEC:.0f}s" if timed_out else f"exit code {exit_code}"
        self.log.error("job %s: pipeline %s without result, see %s", job_id, reason, self.store.log_path(job_id))
        self._finish(job_id, "done", t0, exit_code=exit_code, timed_out=timed_out,
                     result=fallback_result(), pipeline_error=reason)

    def _finish(self, job_id: str, status: str, t0: float, **fields) -> None:
        elapsed = time.perf_counter() - t0
        data = self.store.update_status(job_id, status=status, finished_at=time.time(),
                                        elapsed_sec=round(elapsed, 2), **fields)
        total = time.time() - float(data.get("created_at") or time.time())
        self.log.info("job %s %s in %.1fs (since upload %.1fs)%s", job_id, status, elapsed, total,
                      f" error={fields['error']}" if fields.get("error") else "")

    def _log_tail(self, job_id: str, n_bytes: int = 4000) -> str:
        try:
            with open(self.store.log_path(job_id), "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - n_bytes))
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return ""
