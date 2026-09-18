"""HTTP API for the tracker simulator."""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app import config
from app.core import jobs
from app.core.registry import REGISTRY, DETECTOR_PARAMS, SCENARIO_DETECTION_PARAMS, default_params, get_tracker
from app.core.runner import run_simulation, write_video
from app.core.scenario import Scenario

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"ok": True}


@router.get("/trackers")
def list_trackers():
    """Everything the simulator knows how to run, with every hyperparameter."""
    from app.core.trackers import _opencv_probe
    avail = _opencv_probe()
    out = []
    for t in REGISTRY:
        t2 = dict(t)
        t2["available"] = (t["engine"] != "opencv") or (t["id"] in avail)
        out.append(t2)
    return {
        "trackers": out,
        "detector_params": DETECTOR_PARAMS,
        "scenario_detection_params": SCENARIO_DETECTION_PARAMS,
        "defaults": {t["id"]: default_params(t["id"]) for t in REGISTRY},
    }


@router.post("/scenarios/preview")
def scenario_preview(payload: dict):
    """Build a scenario and render a preview clip with ground-truth overlay."""
    params = {**payload}
    params.setdefault("seed", 7)
    try:
        sc = Scenario(params)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Bad scenario: {e}") from e
    jid = config.new_id()
    frames = [sc.render_annotated_frame(t, tracks=[], draw_gt=True) for t in range(sc.meta["frames"])]
    path = config.SCENARIOS_DIR / f"{jid}_preview.mp4"
    sc.meta["preview_url"] = f"/api/media/{path.name}"
    try:
        sc.meta["preview_codec"] = write_video(frames, str(path), sc.fps)
    except Exception as e:  # noqa: BLE001
        sc.meta["preview_warning"] = str(e)
    return {"scenario_id": jid, "meta": sc.meta}


@router.post("/simulations")
def start_simulation(payload: dict):
    """payload: {scenario: {...}, detection: {...}, trackers: [{tracker_id, params}]}"""
    scenario_params = dict(payload.get("scenario", {}))
    detection_params = dict(payload.get("detection", DEFAULT_DETECTION_PROFILE))
    payload["detection"] = detection_params
    raw_trackers = payload.get("trackers", [])
    specs = []
    for tr in raw_trackers:
        tid = tr.get("tracker_id")
        get_tracker(tid)  # raises 404-ish KeyError
        params = {**(tr.get("params") or {})}
        meta = get_tracker(tid)
        for p in meta["params"]:
            params.setdefault(p["key"], p["default"])
        specs.append({"tracker_id": tid, "params": params})
    payload["trackers"] = specs

    try:
        jid = jobs.start_job("simulation", payload, _run_simulation_job)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"job_id": jid}


def _run_simulation_job(progress, payload):
    sc = Scenario(payload["scenario"])
    result = run_simulation(sc, payload["trackers"], payload["detection"],
                            progress=progress, out_dir=str(config.JOBS_DIR),
                            job_id=progress.jid)
    return result


@router.post("/real/upload")
async def real_upload(file: UploadFile):
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    name = f"{config.new_id()}{ext}"
    path = config.UPLOADS_DIR / name
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    path.write_bytes(data)
    return {"upload_id": path.name, "size": len(data)}


@router.post("/real/jobs")
def start_real_job(payload: dict):
    upload_id = payload.get("upload_id")
    if not upload_id:
        raise HTTPException(status_code=400, detail="upload_id required")
    if not (config.UPLOADS_DIR / upload_id).exists():
        raise HTTPException(status_code=400, detail="upload not found; re-upload")
    tid = payload.get("tracker_id")
    get_tracker(tid)  # KeyError -> 400 below
    params = {p["key"]: p["default"] for p in get_tracker(tid)["params"]}
    params.update(payload.get("params") or {})
    det_params = {p["key"]: p["default"] for p in DETECTOR_PARAMS}
    det_params.update(payload.get("det_params") or {})

    def _run(progress, pl):
        import cv2
        from app.core.real import run_real
        from ultralytics import YOLO
        src = config.UPLOADS_DIR / pl["upload_id"]
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError("Could not open the uploaded video")
        progress("Loading detector", 0.05)
        model = YOLO(pl["det_params"]["model"] if "model" in pl["det_params"] else "yolov8n")
        out_dir = str(config.VIDEOS_DIR)
        result = run_real(model, cap, pl["tid"], pl["params"], pl["det_params"],
                          progress=progress, out_dir=out_dir, job_id=progress.jid)
        return {"results": [result]}

    try:
        jid = jobs.start_job("real", dict(tid=tid, upload_id=upload_id, params=params, det_params=det_params),
                             _run)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"job_id": jid}


@router.get("/jobs/{jid}")
def job_status(jid: str):
    j = jobs.get_job(jid)
    if j is None:
        raise HTTPException(status_code=404, detail="no such job")
    return j


@router.get("/media/{name}")
def media(name: str):
    for d in config.MEDIA_DIRS:
        p = d / name
        if p.exists():
            return FileResponse(str(p))
    raise HTTPException(status_code=404, detail="media not found")


@router.get("/defaults")
def defaults():
    return {"detection": DEFAULT_DETECTION_PROFILE}


DEFAULT_DETECTION_PROFILE = {p["key"]: p["default"] for p in DETECTOR_PARAMS}
DEFAULT_DETECTION_PROFILE.update({p["key"]: p["default"] for p in SCENARIO_DETECTION_PARAMS})