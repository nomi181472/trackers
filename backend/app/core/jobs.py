import json
import logging
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from app import config
from app.config import new_id
from app.logging_config import request_id_ctx

logger = logging.getLogger("tracker_app.jobs")

_lock = threading.Lock()
_JOBS: dict[str, dict] = {}
_executor: ThreadPoolExecutor | None = None
# Map job_id -> Future for queued/running jobs so we can cancel them
_FUTURES: dict[str, Future] = {}
# Set of job_ids that have been requested to cancel
_CANCEL_REQUESTS: set[str] = set()


def get_executor() -> ThreadPoolExecutor:
    """Get active executor or lazily initialize/re-create if shut down."""
    global _executor
    with _lock:
        if _executor is None or getattr(_executor, "_shutdown", False):
            _executor = ThreadPoolExecutor(max_workers=config.SIM_MAX_WORKERS, thread_name_prefix="job")
        return _executor


def shutdown_executor(wait: bool = False, cancel_futures: bool = True) -> None:
    """Shutdown executor cleanly."""
    global _executor
    with _lock:
        if _executor is not None:
            _executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            _executor = None


class QueueFullError(Exception):
    """Raised when concurrent pending submissions exceed the bounded queue depth."""
    pass


class JobCancelledError(Exception):
    """Raised inside a running job when a cancellation has been requested."""
    pass


def get_executor_stats() -> dict:
    """Return current concurrency metrics for diagnostics and UI status."""
    with _lock:
        active_running = sum(1 for j in _JOBS.values() if j.get("status") == "running")
        pending_queued = sum(1 for j in _JOBS.values() if j.get("status") == "queued")
    return {
        "max_workers": config.SIM_MAX_WORKERS,
        "max_queue_size": config.SIM_MAX_QUEUE_SIZE,
        "running_jobs": active_running,
        "queued_jobs": pending_queued,
    }


def _job_file_path(jid: str) -> Path:
    return config.JOBS_DIR / f"{jid}.json"


def _persist_job(job: dict) -> None:
    """Save job state to disk in config.JOBS_DIR so it persists across serverless function instances."""
    try:
        config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        path = _job_file_path(job["id"])
        # Write to temporary file and atomically replace
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(job, f)
        tmp_path.replace(path)
    except Exception as e:
        logger.warning("Could not persist job %s to disk: %s", job.get("id"), e)


def _load_job_from_disk(jid: str) -> dict | None:
    """Read job state from disk if available."""
    path = _job_file_path(jid)
    if path.exists() and path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read job %s from disk: %s", jid, e)
    return None


def recover_interrupted_jobs() -> int:
    """Scan disk for jobs left in 'running' or 'queued' state across process restarts and mark them as errored."""
    recovered = 0
    if not config.JOBS_DIR.exists():
        return recovered
    for p in config.JOBS_DIR.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("status") in ("running", "queued"):
                rec["status"] = "error"
                rec["error"] = "Job interrupted by process restart"
                rec["error_type"] = "ProcessRestartError"
                rec["message"] = "Execution interrupted by server or process restart"
                with _lock:
                    _JOBS[rec["id"]] = rec
                _persist_job(rec)
                recovered += 1
                logger.info("Marked interrupted job %s as failed after restart", rec["id"])
        except Exception as e:
            logger.warning("Could not check/recover job file %s: %s", p, e)
    return recovered


def start_job(kind: str, payload: dict, fn) -> str:
    with _lock:
        queued_count = sum(1 for j in _JOBS.values() if j.get("status") == "queued")
        if queued_count >= config.SIM_MAX_QUEUE_SIZE:
            logger.warning(
                "Job admission rejected: queue depth (%d) reached capacity limit (%d)",
                queued_count, config.SIM_MAX_QUEUE_SIZE,
            )
            raise QueueFullError(
                f"Job queue is full ({queued_count}/{config.SIM_MAX_QUEUE_SIZE} queued). Please wait for ongoing tasks to finish."
            )

        jid = new_id()
        job_record = dict(id=jid, kind=kind, status="queued", progress=0.0,
                          message="Queued", result=None, error=None, traceback=None,
                          error_type=None, payload=payload)
        _JOBS[jid] = job_record

    _persist_job(job_record)
    logger.info("Job queued: %s (kind=%s, workers=%d, queued=%d)", jid, kind, config.SIM_MAX_WORKERS, queued_count + 1)
    future = get_executor().submit(_run, jid, fn)
    with _lock:
        _FUTURES[jid] = future
    return jid


def cancel_job(jid: str) -> bool:
    """Request cancellation of a queued or running job.

    - If the job is still queued (not yet started), the Future is cancelled immediately.
    - If the job is already running, a cancel flag is set; the frame loop in runner.py
      will check this flag and raise JobCancelledError at the next tick.

    Returns True if the cancellation was accepted, False if the job was already finished.
    """
    with _lock:
        job = _JOBS.get(jid)
        if job is None:
            job = _load_job_from_disk(jid)
        if job is None:
            return False
        status = job.get("status")
        if status in ("done", "error", "cancelled"):
            return False

        # Mark cancel intent immediately so the running thread can detect it
        _CANCEL_REQUESTS.add(jid)

        # If still queued, try to cancel the future outright before it starts
        future = _FUTURES.get(jid)
        if future is not None and status == "queued":
            cancelled = future.cancel()
            if cancelled:
                job["status"] = "cancelled"
                job["message"] = "Cancelled before execution started"
                job["error"] = "Cancelled by user"
                job["error_type"] = "JobCancelledError"
                _JOBS[jid] = job
                _persist_job(job)
                logger.info("Job %s cancelled before start (future.cancel succeeded)", jid)
                return True

    logger.info("Cancellation requested for running job %s", jid)
    return True


def is_job_cancelled(jid: str) -> bool:
    """Check whether a cancellation has been requested for the given job."""
    with _lock:
        return jid in _CANCEL_REQUESTS


def _run(jid: str, fn):
    token = request_id_ctx.set(f"job-{jid}")
    try:
        with _lock:
            job = _JOBS.get(jid) or _load_job_from_disk(jid)
        if not job:
            logger.warning("Job %s not found on execution start", jid)
            return

        # Check if already cancelled before we even start
        if is_job_cancelled(jid):
            with _lock:
                job["status"] = "cancelled"
                job["message"] = "Cancelled before execution started"
                job["error"] = "Cancelled by user"
                job["error_type"] = "JobCancelledError"
                _JOBS[jid] = job
            _persist_job(job)
            return

        job["status"] = "running"
        job["message"] = "Starting"
        with _lock:
            _JOBS[jid] = job
        _persist_job(job)
        logger.info("Starting execution of job %s (kind=%s)", jid, job.get("kind"))
        try:
            result = fn(_ProgressFn(jid), job["payload"])
            with _lock:
                cur = _JOBS.get(jid) or job
                cur["status"] = "done"
                cur["progress"] = 1.0
                cur["message"] = "Finished"
                cur["result"] = result
                _JOBS[jid] = cur
            _persist_job(cur)
            logger.info("Job %s completed successfully", jid)
        except JobCancelledError:
            logger.info("Job %s was cancelled during execution", jid)
            with _lock:
                cur = _JOBS.get(jid) or job
                cur["status"] = "cancelled"
                cur["message"] = "Cancelled by user"
                cur["error"] = "Cancelled by user"
                cur["error_type"] = "JobCancelledError"
                _JOBS[jid] = cur
            _persist_job(cur)
        except Exception as e:  # noqa: BLE001
            tb_str = traceback.format_exc()
            err_type = type(e).__name__
            logger.exception("Job %s failed with %s: %s", jid, err_type, e)
            with _lock:
                cur = _JOBS.get(jid) or job
                cur["status"] = "error"
                cur["error"] = str(e)
                cur["error_type"] = err_type
                cur["traceback"] = tb_str
                cur["message"] = f"Failed: {e}"
                _JOBS[jid] = cur
            _persist_job(cur)
    finally:
        # Clean up future reference and cancel token
        with _lock:
            _FUTURES.pop(jid, None)
            _CANCEL_REQUESTS.discard(jid)
        request_id_ctx.reset(token)


class _ProgressFn:
    def __init__(self, jid: str):
        self.jid = jid

    def __call__(self, message: str, frac: float, detail: str = ""):
        # Check cancellation first — raises JobCancelledError if requested
        self.check_cancelled()
        with _lock:
            j = _JOBS.get(self.jid) or _load_job_from_disk(self.jid)
            if j:
                j["message"] = message
                j["progress"] = min(1.0, max(0.0, float(frac)))
                if detail:
                    j["detail"] = detail
                _JOBS[self.jid] = j
                _persist_job(j)

    def check_cancelled(self):
        """Raise JobCancelledError if a cancellation has been requested for this job."""
        if is_job_cancelled(self.jid):
            raise JobCancelledError(f"Job {self.jid} was cancelled by user request")


def get_job(jid: str) -> dict | None:
    with _lock:
        j = _JOBS.get(jid)
    if not j:
        j = _load_job_from_disk(jid)
        if j:
            with _lock:
                _JOBS[jid] = j
    return dict(j) if j else None


def set_result(jid: str, result):
    with _lock:
        j = _JOBS.get(jid) or _load_job_from_disk(jid)
        if j:
            j["result"] = result
            _JOBS[jid] = j
            _persist_job(j)