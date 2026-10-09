"""HTTP API for the tracker simulator."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse

from app import config
from app.core import jobs
from app.core.plugins import REGISTRY
from app.core.registry import DETECTION_DEFAULTS, DETECTOR_PARAMS, SCENARIO_DETECTION_PARAMS, get_tracker
from app.core.runner import run_simulation, write_video, video_to_data_url
from app.core.scenario import Scenario

router = APIRouter(prefix="/api")


def _verify_admin_access(x_admin_key: str | None = Header(None, alias="X-Admin-Key")):
    """Verify admin key if SIM_ADMIN_KEY is configured in the environment."""
    required_key = os.environ.get("SIM_ADMIN_KEY")
    if required_key and x_admin_key != required_key:
        raise HTTPException(status_code=403, detail="Unauthorized: invalid or missing X-Admin-Key header")


def _bad_tracker(e: KeyError) -> HTTPException:
    """An unknown tracker id is a client mistake, not a server crash.

    `str(KeyError)` would wrap the message in another pair of quotes, and the
    UI shows this text verbatim, so unwrap it.
    """
    return HTTPException(status_code=400, detail=e.args[0] if e.args else str(e))


@router.get("/health")
def health():
    return {
        "ok": True,
        "workers": jobs.get_executor_stats(),
    }


@router.get("/trackers")
def list_trackers():
    """Everything the simulator knows how to run, with every hyperparameter."""
    trackers = REGISTRY.catalog()
    return {
        "trackers": trackers,
        "detector_params": DETECTOR_PARAMS,
        "scenario_detection_params": SCENARIO_DETECTION_PARAMS,
        "defaults": {t["id"]: REGISTRY.default_params(t["id"]) for t in trackers},
        "worker_concurrency": jobs.get_executor_stats(),
    }


def _parse_int(val, name: str, default: int | None = None) -> int:
    if val is None:
        if default is not None:
            return default
        raise HTTPException(status_code=400, detail=f"{name} must be an integer")
    if isinstance(val, bool):
        raise HTTPException(status_code=400, detail=f"{name} must be an integer")
    try:
        return int(val)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"{name} must be a valid integer") from e


def _parse_float(val, name: str, default: float | None = None) -> float:
    if val is None:
        if default is not None:
            return default
        raise HTTPException(status_code=400, detail=f"{name} must be a number")
    if isinstance(val, bool):
        raise HTTPException(status_code=400, detail=f"{name} must be a number")
    try:
        return float(val)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"{name} must be a valid number") from e


def _validate_scenario(cfg: dict) -> dict:
    """Validate scenario configuration parameters and resource bounds."""
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail="Scenario configuration must be an object")
    params = dict(cfg)

    # 1. Dimensions
    width = _parse_int(params.get("width", 640), "width")
    height = _parse_int(params.get("height", 640), "height")
    if width < 64 or width > 1920:
        raise HTTPException(status_code=400, detail="width must be between 64 and 1920")
    if height < 64 or height > 1920:
        raise HTTPException(status_code=400, detail="height must be between 64 and 1920")
    params["width"] = width
    params["height"] = height

    # 2. FPS & Duration
    fps = _parse_int(params.get("fps", 15), "fps")
    if fps < 1 or fps > 60:
        raise HTTPException(status_code=400, detail="fps must be between 1 and 60")
    params["fps"] = fps

    duration = _parse_float(params.get("duration_seconds", 8.0), "duration_seconds")
    if duration <= 0 or duration > 60:
        raise HTTPException(status_code=400, detail="duration_seconds must be between 0 and 60")
    params["duration_seconds"] = duration

    # 3. Object count
    num_objects = _parse_int(params.get("num_objects", 6), "num_objects")
    if num_objects < 1 or num_objects > 50:
        raise HTTPException(status_code=400, detail="num_objects must be between 1 and 50")
    params["num_objects"] = num_objects

    # 4. Effective frame count: if frames is explicitly passed, validate it;
    # otherwise or additionally ensure the computed duration * fps is bounded.
    if "frames" in params and params["frames"] is not None:
        frames = _parse_int(params["frames"], "frames")
        if frames < 1 or frames > 600:
            raise HTTPException(status_code=400, detail="frames must be between 1 and 600")
        params["frames"] = frames
    else:
        frames = max(2, int(duration * fps))
        if frames < 1 or frames > 600:
            raise HTTPException(status_code=400, detail="computed frames (duration_seconds * fps) must be between 1 and 600")
        params["frames"] = frames

    if "seed" in params and params["seed"] is not None:
        params["seed"] = _parse_int(params["seed"], "seed")

    # 5. Combined resource cost limit
    # (Frames * Width * Height: cap buffer memory, e.g. max 600 frames * 640 * 640 ≈ 2.45e8 pixels; max 300,000,000)
    total_pixels = frames * width * height
    MAX_TOTAL_PIXELS = 300_000_000
    if total_pixels > MAX_TOTAL_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Scenario workload too large ({total_pixels:,} total pixels; limit is {MAX_TOTAL_PIXELS:,})"
        )

    return params


@router.post("/scenarios/preview")
def scenario_preview(payload: dict):
    """Build a scenario and render a preview clip with ground-truth overlay."""
    import gc
    params = _validate_scenario(payload)
    params.setdefault("seed", 7)
    try:
        sc = Scenario(params)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Bad scenario: {e}") from e
    jid = config.new_id()
    path = config.SCENARIOS_DIR / f"{jid}_preview.webm"
    sc.meta["preview_url"] = f"/api/media/{path.name}"
    def _preview_frames():
        for t in range(sc.meta["frames"]):
            yield sc.render_annotated_frame(t, tracks=[], draw_gt=True)
    try:
        sc.meta["preview_codec"] = write_video(_preview_frames(), str(path), sc.fps)
        data_url = video_to_data_url(str(path))
        if data_url:
            sc.meta["preview_data_url"] = data_url
    except Exception as e:  # noqa: BLE001
        sc.meta["preview_warning"] = str(e)
    gc.collect()
    return {"scenario_id": jid, "meta": sc.meta}


@router.post("/simulations")
def start_simulation(payload: dict):
    """payload: {scenario: {...}, detection: {...}, trackers: [{tracker_id, params}]}"""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be an object")
    scenario_cfg = payload.get("scenario")
    if scenario_cfg is None:
        scenario_cfg = {}
    elif not isinstance(scenario_cfg, dict):
        raise HTTPException(status_code=400, detail="scenario must be an object")
    validated_scenario = _validate_scenario(scenario_cfg)
    payload["scenario"] = validated_scenario

    raw_trackers = payload.get("trackers", [])
    if not isinstance(raw_trackers, list) or not raw_trackers:
        raise HTTPException(status_code=400, detail="At least one tracker must be specified")
    if len(raw_trackers) > 18:
        raise HTTPException(status_code=400, detail="Cannot benchmark more than 18 trackers simultaneously")

    # Merge, don't replace: a client that sends only `conf` must still get the
    # rest of the detector knobs rather than silently falling back to whatever
    # `SimDetector` happens to hardcode.
    detection_cfg = payload.get("detection")
    if detection_cfg is not None and not isinstance(detection_cfg, dict):
        raise HTTPException(status_code=400, detail="detection must be an object")
    detection_params = {**DETECTION_DEFAULTS, **(detection_cfg or {})}
    payload["detection"] = detection_params
    specs = []
    for tr in raw_trackers:
        if not isinstance(tr, dict):
            raise HTTPException(status_code=400, detail="Each tracker specification must be an object")
        tid = tr.get("tracker_id")
        try:
            meta = get_tracker(tid)
        except KeyError as e:
            raise _bad_tracker(e) from e
        params = tr.get("params") or {}
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="Tracker params must be an object")
        params = {**params}
        for p in meta["params"]:
            params.setdefault(p["key"], p["default"])
        specs.append({"tracker_id": tid, "params": params})
    payload["trackers"] = specs

    try:
        jid = jobs.start_job("simulation", payload, _run_simulation_job)
    except jobs.QueueFullError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": "2"},
        ) from e
    return {"job_id": jid}


def _run_simulation_job(progress, payload):
    sc = Scenario(payload["scenario"])
    result = run_simulation(sc, payload["trackers"], payload["detection"],
                            progress=progress, out_dir=str(config.JOBS_DIR),
                            job_id=progress.jid)
    return result


@router.get("/jobs/{jid}")
def job_status(jid: str):
    j = jobs.get_job(jid)
    if j is None:
        raise HTTPException(status_code=404, detail="no such job")
    # Allow-list, not blacklist: `payload` and `traceback` are internal and logged to
    # container stdout/files, and must not leak by default to general job pollers.
    return {k: j.get(k) for k in ("id", "kind", "status", "progress",
                                  "message", "detail", "error", "result")}


@router.post("/jobs/{jid}/cancel")
def cancel_job(jid: str):
    """Request cancellation of a queued or running simulation job.

    - Queued jobs are cancelled immediately (future.cancel).
    - Running jobs set a cancellation flag; the simulation frame loop checks it
      at each progress tick and raises JobCancelledError to stop gracefully.
    - Already-finished jobs (done / error / cancelled) return 409.
    """
    j = jobs.get_job(jid)
    if j is None:
        raise HTTPException(status_code=404, detail="no such job")
    current_status = j.get("status")
    if current_status in ("done", "error", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is already in terminal state: {current_status}",
        )
    accepted = jobs.cancel_job(jid)
    if not accepted:
        raise HTTPException(status_code=409, detail="Could not cancel job")
    return {"job_id": jid, "status": "cancelling"}


@router.get("/media/{name}")
def media(name: str):
    candidates = [name]
    stem, ext = os.path.splitext(name)
    if ext == ".mp4":
        candidates.insert(0, f"{stem}.webm")
    elif ext == ".webm":
        candidates.append(f"{stem}.mp4")

    for d in config.MEDIA_DIRS:
        for cand in candidates:
            p = d / cand
            if p.exists():
                media_type = (
                    "video/webm" if p.suffix == ".webm"
                    else "video/mp4" if p.suffix == ".mp4"
                    else "image/png" if p.suffix == ".png"
                    else "image/jpeg" if p.suffix in (".jpg", ".jpeg")
                    else None
                )
                return FileResponse(str(p), media_type=media_type)
    raise HTTPException(status_code=404, detail="media not found")


@router.get("/defaults")
def defaults():
    return {"detection": DETECTION_DEFAULTS}


@router.get("/logs/files")
def list_log_files():
    """List available daily log files sorted from newest to oldest."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(config.LOGS_DIR.glob("app_*.log"), reverse=True)
    res = []
    for f in files:
        try:
            stat = f.stat()
            res.append({
                "filename": f.name,
                "date": f.name.replace("app_", "").replace(".log", ""),
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            })
        except OSError:
            continue
    return {"files": res}


@router.get("/logs")
def get_log_lines(
    file: str | None = None,
    cursor: int | None = None,
    limit: int = 50,
):
    """Retrieve lines with cursor-based pagination.
    
    - `file`: name of the log file (e.g. app_2026-10-07.log). Defaults to latest log file.
    - `cursor`: line offset (0-indexed). If omitted, loads the newest lines from the bottom.
      Returns `next_cursor` pointing to older lines so scrolling down continues backward.
    - `limit`: number of lines per page (default 50, max 200).
    """
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not file:
        available = sorted(config.LOGS_DIR.glob("app_*.log"), reverse=True)
        if not available:
            return {"file": None, "lines": [], "next_cursor": None, "total_lines": 0}
        target_path = available[0]
    else:
        # Prevent directory traversal
        clean_name = os.path.basename(file)
        target_path = config.LOGS_DIR / clean_name

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Log file '{file}' not found")

    limit = min(max(1, limit), 200)

    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed reading log file: {e}")

    total = len(all_lines)
    # Cursor is the upper bound line index for older pagination
    end = cursor if (cursor is not None and 0 <= cursor <= total) else total
    start = max(0, end - limit)
    
    chunk = [line.rstrip("\r\n") for line in all_lines[start:end]]
    next_cursor = start if start > 0 else None

    return {
        "file": target_path.name,
        "lines": chunk,
        "next_cursor": next_cursor,
        "total_lines": total,
        "start_line": start,
        "end_line": end,
    }


@router.post("/cleanup", dependencies=[Depends(_verify_admin_access)])
@router.delete("/cleanup", dependencies=[Depends(_verify_admin_access)])
@router.post("/clear", dependencies=[Depends(_verify_admin_access)])
@router.delete("/clear", dependencies=[Depends(_verify_admin_access)])
def cleanup_data_files(
    include_uploads: bool = False,
    include_jobs: bool = True,
    include_scenarios: bool = True,
):
    """Purge generated job media (*.mp4, *.jpg) and scenario clips on request.
    
    Targets:
    - backend/data/jobs/**/*.mp4
    - backend/data/jobs/**/*.jpg
    - backend/data/scenarios/*.mp4
    - (optional) backend/data/uploads/*
    """
    deleted_count = 0
    freed_bytes = 0
    details = {"jobs_mp4": 0, "jobs_jpg": 0, "jobs_png": 0, "scenarios_mp4": 0, "uploads": 0}

    # 1. Clean jobs directory (webm, mp4, jpg, and png files)
    if include_jobs and config.JOBS_DIR.exists():
        for pattern, key in [
            ("**/*.webm", "jobs_webm"),
            ("**/*.mp4", "jobs_mp4"),
            ("**/*.jpg", "jobs_jpg"),
            ("**/*.png", "jobs_png"),
        ]:
            for p in config.JOBS_DIR.glob(pattern):
                try:
                    if p.is_file():
                        sz = p.stat().st_size
                        p.unlink()
                        deleted_count += 1
                        freed_bytes += sz
                        details[key] = details.get(key, 0) + 1
                except OSError:
                    continue

    # 2. Clean scenarios directory (webm and mp4 files)
    if include_scenarios and config.SCENARIOS_DIR.exists():
        for pattern, key in [("*.webm", "scenarios_webm"), ("*.mp4", "scenarios_mp4")]:
            for p in config.SCENARIOS_DIR.glob(pattern):
                try:
                    if p.is_file():
                        sz = p.stat().st_size
                        p.unlink()
                        deleted_count += 1
                        freed_bytes += sz
                        details[key] = details.get(key, 0) + 1
                except OSError:
                    continue

    # 3. Clean uploads directory if explicitly requested
    if include_uploads and config.UPLOADS_DIR.exists():
        for p in config.UPLOADS_DIR.iterdir():
            try:
                if p.is_file():
                    sz = p.stat().st_size
                    p.unlink()
                    deleted_count += 1
                    freed_bytes += sz
                    details["uploads"] += 1
            except OSError:
                continue

    # Ensure empty target directories exist for future simulations
    for d in (config.SCENARIOS_DIR, config.JOBS_DIR, config.VIDEOS_DIR, config.UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "ok": True,
        "deleted_count": deleted_count,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        "details": details,
    }