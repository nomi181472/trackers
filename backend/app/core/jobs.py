import json
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config
from app.config import new_id
from app.logging_config import request_id_ctx

logger = logging.getLogger("tracker_app.jobs")

_lock = threading.Lock()
_JOBS: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=config.SIM_MAX_WORKERS, thread_name_prefix="job")


class QueueFullError(Exception):
    """Raised when concurrent pending submissions exceed the bounded queue depth."""
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
    _executor.submit(_run, jid, fn)
    return jid


def _run(jid: str, fn):
    token = request_id_ctx.set(f"job-{jid}")
    try:
        with _lock:
            job = _JOBS.get(jid) or _load_job_from_disk(jid)
        if not job:
            logger.warning("Job %s not found on execution start", jid)
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
        request_id_ctx.reset(token)


class _ProgressFn:
    def __init__(self, jid: str):
        self.jid = jid

    def __call__(self, message: str, frac: float, detail: str = ""):
        with _lock:
            j = _JOBS.get(self.jid) or _load_job_from_disk(self.jid)
            if j:
                j["message"] = message
                j["progress"] = min(1.0, max(0.0, float(frac)))
                if detail:
                    j["detail"] = detail
                _JOBS[self.jid] = j
                _persist_job(j)


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