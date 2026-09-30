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
    assert len(body["trackers"]) == 16
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


def test_unknown_tracker_on_real_job_is_a_400(client, uploaded):
    r = client.post("/api/real/jobs", json={"upload_id": uploaded, "tracker_id": "not_a_tracker"})
    assert r.status_code == 400, r.text
    assert "Unknown tracker" in r.json()["detail"]


def test_single_object_tracker_is_rejected_in_real_mode(client, uploaded):
    """Fix 4: it used to build the engine, never init it, and score a black clip 'A'."""
    r = client.post("/api/real/jobs", json={"upload_id": uploaded, "tracker_id": "mil"})
    assert r.status_code == 400, r.text
    assert "single-object" in r.json()["detail"].lower()


def test_real_job_requires_an_upload(client):
    assert client.post("/api/real/jobs", json={"tracker_id": "bytetrack"}).status_code == 400
    assert client.post("/api/real/jobs",
                       json={"upload_id": "nope.mp4", "tracker_id": "bytetrack"}).status_code == 400


def test_job_status_reports_unknown_jobs(client):
    assert client.get("/api/jobs/definitely-not-a-job").status_code == 404
