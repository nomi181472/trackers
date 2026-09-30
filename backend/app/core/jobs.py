"""In-memory async job runner.

Jobs live in a plain dict; a small thread pool executes them.  The API layer
just polls GET /api/jobs/{id} while the worker updates progress -- simple and
surprisingly effective for a teaching simulator.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from app.config import new_id

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")
_lock = threading.Lock()
_JOBS: dict[str, dict] = {}


def start_job(kind: str, payload: dict, fn) -> str:
    jid = new_id()
    with _lock:
        _JOBS[jid] = dict(id=jid, kind=kind, status="queued", progress=0.0,
                          message="Queued", result=None, error=None, payload=payload)
    _executor.submit(_run, jid, fn)
    return jid


def _run(jid: str, fn):
    with _lock:
        job = _JOBS.get(jid)
    if not job:
        return
    job["status"] = "running"
    job["message"] = "Starting"
    try:
        result = fn(_ProgressFn(jid), job["payload"])
        with _lock:
            if _JOBS.get(jid):
                _JOBS[jid]["status"] = "done"
                _JOBS[jid]["progress"] = 1.0
                _JOBS[jid]["message"] = "Finished"
                _JOBS[jid]["result"] = result
    except Exception as e:  # noqa: BLE001
        with _lock:
            if _JOBS.get(jid):
                _JOBS[jid]["status"] = "error"
                # No traceback: it would ride along to whoever polls this job.
                _JOBS[jid]["error"] = str(e)
                _JOBS[jid]["message"] = f"Failed: {e}"


class _ProgressFn:
    def __init__(self, jid: str):
        self.jid = jid

    def __call__(self, message: str, frac: float, detail: str = ""):
        with _lock:
            j = _JOBS.get(self.jid)
            if j:
                j["message"] = message
                j["progress"] = min(1.0, max(0.0, float(frac)))
                if detail:
                    j["detail"] = detail


def get_job(jid: str) -> dict | None:
    with _lock:
        j = _JOBS.get(jid)
        return dict(j) if j else None


def set_result(jid: str, result):
    with _lock:
        if _JOBS.get(jid):
            _JOBS[jid]["result"] = result