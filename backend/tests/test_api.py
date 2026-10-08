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
    assert client.get("/api/health").json() == {"ok": True}


def test_catalog_shape_is_unchanged(client):
    body = client.get("/api/trackers").json()
    assert set(body) == {"trackers", "detector_params", "scenario_detection_params", "defaults"}
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
