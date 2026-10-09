"""HTTP-level contract: the catalog still feeds the UI unchanged, and a bad
tracker id is a client mistake (400), not a server crash (500)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config
from app.core.plugins import REGISTRY
from app.main import app

CATALOG_KEYS = {
    "id", "name", "engine", "mode", "tagline", "description",
    "strengths", "failure_modes", "params", "available",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def uploaded():
    name = f"pytest-{id(object()):x}.mp4"
    path = config.UPLOADS_DIR / name
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42-not-a-real-video")
    try:
        yield name
    finally:
        path.unlink(missing_ok=True)


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert "workers" in body
    assert body["workers"]["max_workers"] >= 1


def test_catalog_shape_is_unchanged(client):
    body = client.get("/api/trackers").json()
    assert {"trackers", "detector_params", "scenario_detection_params", "defaults"} <= set(body)
    assert "worker_concurrency" in body
    assert body["worker_concurrency"]["max_workers"] >= 1
    assert len(body["trackers"]) == len(REGISTRY.ids())
    for entry in body["trackers"]:
        assert set(entry) == CATALOG_KEYS, entry.get("id")
        assert isinstance(entry["available"], bool)
        assert body["defaults"][entry["id"]] == {p["key"]: p["default"] for p in entry["params"]}
    for field in ("detector_params", "scenario_detection_params"):
        assert body[field], field
        for p in body[field]:
            assert {"key", "label", "type", "default"} <= set(p)


def test_catalog_availability_matches_the_plugins(client):
    body = client.get("/api/trackers").json()
    expected = {t["id"]: t["available"] for t in REGISTRY.catalog()}
    assert {t["id"]: t["available"] for t in body["trackers"]} == expected
    for entry in body["trackers"]:
        assert entry["available"] == REGISTRY.get(entry["id"]).is_available()


def test_unknown_tracker_on_simulation_is_a_400(client):
    r = client.post("/api/simulations", json={
        "scenario": {"seed": 1, "frames": 5},
        "trackers": [{"tracker_id": "not_a_tracker"}],
    })
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail.startswith("Unknown tracker 'not_a_tracker'. Known: [")
    assert not detail.startswith('"'), "the detail is shown to the user verbatim"


def test_known_tracker_on_simulation_is_accepted(client):
    r = client.post("/api/simulations", json={
        "scenario": {"seed": 1, "frames": 5},
        "trackers": [{"tracker_id": "greedy_iou"}],
    })
    assert r.status_code == 200, r.text
    assert r.json()["job_id"]


def test_job_status_reports_unknown_jobs(client):
    assert client.get("/api/jobs/definitely-not-a-job").status_code == 404


def test_job_status_does_not_leak_internals(client):
    """An errored job must not hand the caller a traceback or the raw payload."""
    import time
    from app.core import jobs

    def _boom(progress, payload):
        progress("halfway", 0.5)
        raise RuntimeError("engine exploded on purpose")

    jid = jobs.start_job("simulation", {"secret": "do-not-echo-me"}, _boom)
    body = {}
    for _ in range(100):
        body = client.get(f"/api/jobs/{jid}").json()
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.02)

    assert body["status"] == "error"
    assert "engine exploded on purpose" in body["error"]
    assert set(body) == {"id", "kind", "status", "progress",
                         "message", "detail", "error", "result"}
    assert "traceback" not in body
    assert "payload" not in body
    assert "do-not-echo-me" not in client.get(f"/api/jobs/{jid}").text


def test_defaults_endpoint_serves_the_registry_defaults(client):
    from app.core.registry import DETECTION_DEFAULTS

    assert client.get("/api/defaults").json() == {"detection": DETECTION_DEFAULTS}


def test_partial_detection_payload_is_merged_not_replaced(client):
    """A client that sends only `conf` must keep the other knobs."""
    from app.core import jobs
    from app.core.registry import DETECTION_DEFAULTS

    r = client.post("/api/simulations", json={
        "scenario": {"seed": 1, "frames": 4},
        "detection": {"conf": 0.42},
        "trackers": [{"tracker_id": "greedy_iou"}],
    })
    assert r.status_code == 200, r.text
    stored = jobs.get_job(r.json()["job_id"])["payload"]["detection"]
    assert stored["conf"] == 0.42
    assert stored["jitter"] == DETECTION_DEFAULTS["jitter"]
    assert stored["fp_rate"] == DETECTION_DEFAULTS["fp_rate"]
    assert set(stored) == set(DETECTION_DEFAULTS)


def test_cleanup_and_clear_endpoints(client):
    # Create test dummy files (including nested job files)
    nested_dir = config.JOBS_DIR / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    dummy_nested_mp4 = nested_dir / "dummy_nested.mp4"
    dummy_nested_jpg = nested_dir / "dummy_nested.jpg"
    dummy_job_mp4 = config.JOBS_DIR / "dummy_test.mp4"
    dummy_job_jpg = config.JOBS_DIR / "dummy_test.jpg"
    dummy_sc_mp4 = config.SCENARIOS_DIR / "dummy_sc.mp4"

    for p in (dummy_nested_mp4, dummy_nested_jpg, dummy_job_mp4, dummy_job_jpg, dummy_sc_mp4):
        p.write_bytes(b"dummy")

    # Test POST /api/clear
    r = client.post("/api/clear")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["deleted_count"] >= 5
    assert not dummy_job_mp4.exists()
    assert not dummy_job_jpg.exists()
    assert not dummy_nested_mp4.exists()
    assert not dummy_nested_jpg.exists()
    assert not dummy_sc_mp4.exists()

    # Re-create and test DELETE /api/clear
    dummy_job_mp4.write_bytes(b"dummy")
    dummy_sc_mp4.write_bytes(b"dummy")
    r_del = client.delete("/api/clear")
    assert r_del.status_code == 200
    assert r_del.json()["ok"] is True
    assert not dummy_job_mp4.exists()
    assert not dummy_sc_mp4.exists()


def test_cleanup_admin_key_protection(client, monkeypatch):
    """When SIM_ADMIN_KEY is configured in env, calls without X-Admin-Key must be 403."""
    monkeypatch.setenv("SIM_ADMIN_KEY", "super-secret-key")
    r = client.post("/api/clear")
    assert r.status_code == 403
    assert "Unauthorized" in r.json()["detail"]

    r_authorized = client.post("/api/clear", headers={"X-Admin-Key": "super-secret-key"})
    assert r_authorized.status_code == 200
    assert r_authorized.json()["ok"] is True


def test_workload_validation_rejects_excessive_duration_or_fps(client):
    # excessive duration
    r = client.post("/api/simulations", json={
        "scenario": {"seed": 1, "duration_seconds": 120, "fps": 10},
        "trackers": [{"tracker_id": "greedy_iou"}],
    })
    assert r.status_code == 400
    assert "duration_seconds must be between 0 and 60" in r.json()["detail"]

    # excessive fps
    r = client.post("/api/simulations", json={
        "scenario": {"seed": 1, "fps": 100},
        "trackers": [{"tracker_id": "greedy_iou"}],
    })
    assert r.status_code == 400
    assert "fps must be between 1 and 60" in r.json()["detail"]


def test_workload_validation_rejects_computed_frames_exceeding_limit(client):
    # duration 50s * fps 30 = 1500 frames (> 600)
    r = client.post("/api/simulations", json={
        "scenario": {"seed": 1, "duration_seconds": 50, "fps": 30},
        "trackers": [{"tracker_id": "greedy_iou"}],
    })
    assert r.status_code == 400
    assert "computed frames" in r.json()["detail"] or "frames must be between" in r.json()["detail"]


def test_workload_validation_rejects_excessive_dimensions(client):
    r = client.post("/api/simulations", json={
        "scenario": {"seed": 1, "width": 4000, "height": 4000},
        "trackers": [{"tracker_id": "greedy_iou"}],
    })
    assert r.status_code == 400
    assert "width must be between 64 and 1920" in r.json()["detail"]


def test_workload_validation_on_scenario_preview(client):
    r = client.post("/api/scenarios/preview", json={
        "duration_seconds": 100,
        "fps": 30,
    })
    assert r.status_code == 400


@pytest.mark.parametrize("field,bad_val", [
    ("width", "not-a-number"),
    ("height", [640]),
    ("fps", "fifteen"),
    ("duration_seconds", "long"),
    ("num_objects", {}),
    ("frames", "invalid"),
    ("width", True),
])
def test_malformed_numerics_return_400_not_500(client, field, bad_val):
    # Preview endpoint
    r_prev = client.post("/api/scenarios/preview", json={field: bad_val})
    assert r_prev.status_code == 400
    assert "must be" in r_prev.json()["detail"]

    # Simulations endpoint
    r_sim = client.post("/api/simulations", json={
        "scenario": {field: bad_val},
        "trackers": [{"tracker_id": "greedy_iou"}],
    })
    assert r_sim.status_code == 400
    assert "must be" in r_sim.json()["detail"]


def test_queue_bounding_and_admission_limit(client, monkeypatch):
    """When the queue capacity is reached, new submissions must return 429."""
    from app.core import jobs

    # Temporarily set max queue size to 2 for test
    monkeypatch.setattr(config, "SIM_MAX_QUEUE_SIZE", 2)

    # Artificially populate queued jobs
    with jobs._lock:
        old_jobs = dict(jobs._JOBS)
        jobs._JOBS["dummy_q1"] = {"id": "dummy_q1", "status": "queued"}
        jobs._JOBS["dummy_q2"] = {"id": "dummy_q2", "status": "queued"}

    try:
        r = client.post("/api/simulations", json={
            "scenario": {"seed": 1, "frames": 2},
            "trackers": [{"tracker_id": "greedy_iou"}],
        })
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        assert "queue is full" in r.json()["detail"].lower()
    finally:
        with jobs._lock:
            jobs._JOBS.clear()
            jobs._JOBS.update(old_jobs)


def test_process_restart_recovery(client):
    """Jobs left in 'running' or 'queued' on disk must be recovered to 'error'."""
    from app.core import jobs

    jid = f"restart_test_{config.new_id()}"
    job_file = config.JOBS_DIR / f"{jid}.json"
    dummy_job = {
        "id": jid,
        "kind": "simulation",
        "status": "running",
        "progress": 0.5,
        "message": "Executing frame 20",
    }
    jobs._persist_job(dummy_job)

    try:
        recovered = jobs.recover_interrupted_jobs()
        assert recovered >= 1

        rec = jobs.get_job(jid)
        assert rec["status"] == "error"
        assert rec["error_type"] == "ProcessRestartError"
        assert "interrupted" in rec["message"].lower()
    finally:
        job_file.unlink(missing_ok=True)


def test_cancel_nonexistent_job(client):
    r = client.post("/api/jobs/nonexistent-id-xyz/cancel")
    assert r.status_code == 404
    assert "no such job" in r.json()["detail"].lower()


def test_cancel_already_finished_job(client):
    from app.core import jobs
    jid = f"finished_job_{config.new_id()}"
    job_file = config.JOBS_DIR / f"{jid}.json"
    dummy = {"id": jid, "kind": "simulation", "status": "done", "progress": 1.0, "message": "Finished"}
    with jobs._lock:
        jobs._JOBS[jid] = dummy
    jobs._persist_job(dummy)
    try:
        r = client.post(f"/api/jobs/{jid}/cancel")
        assert r.status_code == 409
        assert "already in terminal state" in r.json()["detail"].lower()
    finally:
        with jobs._lock:
            jobs._JOBS.pop(jid, None)
        job_file.unlink(missing_ok=True)


def test_cancel_queued_or_running_job(client):
    from app.core import jobs
    jid = f"queued_job_{config.new_id()}"
    job_file = config.JOBS_DIR / f"{jid}.json"
    dummy = {"id": jid, "kind": "simulation", "status": "running", "progress": 0.2, "message": "Running"}
    with jobs._lock:
        jobs._JOBS[jid] = dummy
    jobs._persist_job(dummy)
    try:
        r = client.post(f"/api/jobs/{jid}/cancel")
        assert r.status_code == 200
        assert r.json()["status"] == "cancelling"
        assert jobs.is_job_cancelled(jid) is True
    finally:
        with jobs._lock:
            jobs._JOBS.pop(jid, None)
            jobs._CANCEL_REQUESTS.discard(jid)
        job_file.unlink(missing_ok=True)




