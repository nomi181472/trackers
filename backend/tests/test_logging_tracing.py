"""Tests for end-to-end exception logging and tracing."""
import logging
import pytest
from starlette.testclient import TestClient

from app.core import jobs
from app.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_request_id_header_and_tracing(client):
    """Test that incoming requests get X-Request-ID and are traced."""
    # Custom request ID passed by client
    custom_id = "test-custom-trace-id-123"
    resp = client.get("/api/health", headers={"x-request-id": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == custom_id

    # Auto-generated request ID if not provided
    resp_auto = client.get("/api/health")
    assert resp_auto.status_code == 200
    assert resp_auto.headers.get("X-Request-ID") is not None


def test_unhandled_crash_caught_with_diagnostics(client):
    """Test that an unhandled crash in an endpoint returns 500 and logs diagnostics."""
    @app.get("/api/test-crash")
    def trigger_crash():
        raise ZeroDivisionError("division by zero simulation crash")

    resp = client.get("/api/test-crash")
    assert resp.status_code == 500
    data = resp.json()
    assert data["detail"] == "Internal Server Error"
    assert "division by zero simulation crash" in data["error"]
    assert data["error_type"] == "ZeroDivisionError"
    assert "request_id" in data
    assert resp.headers.get("X-Request-ID") == data["request_id"]


def test_job_crash_logs_and_captures_traceback():
    """Test that background thread job crashes are fully recorded with tracebacks."""
    import time

    def _crash_job(progress, payload):
        raise ValueError("synthetic job crash for testing")

    jid = jobs.start_job("simulation", {}, _crash_job)

    # Wait for job completion
    for _ in range(100):
        j = jobs.get_job(jid)
        if j and j["status"] in ("done", "error"):
            break
        time.sleep(0.05)

    job_record = jobs.get_job(jid)
    assert job_record["status"] == "error"
    assert "synthetic job crash for testing" in job_record["error"]
    assert job_record["error_type"] == "ValueError"
    assert job_record["traceback"] is not None
    assert "ValueError: synthetic job crash for testing" in job_record["traceback"]


def test_daily_log_rotation_and_ten_day_retention(tmp_path):
    """Test that DailyRotatingFileHandler keeps at most 10 days of files and deletes older."""
    import datetime
    from app.core.daily_logger import DailyRotatingFileHandler

    handler = DailyRotatingFileHandler(logs_dir=tmp_path, max_days=10)
    logger = logging.getLogger("test_daily_retention")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Seed 15 dummy older log files
    for day in range(1, 16):
        old_file = tmp_path / f"app_2026-09-{day:02d}.log"
        old_file.write_text(f"dummy log from day {day}\n", encoding="utf-8")

    assert len(list(tmp_path.glob("app_*.log"))) == 15

    # Emit a log entry triggering cleanup
    logger.info("New daily log line triggering cleanup")

    # Files must now be trimmed down to at most 10
    remaining = sorted(f.name for f in tmp_path.glob("app_*.log"))
    assert len(remaining) <= 10
    handler.close()


def test_logs_api_endpoints(client):
    """Test GET /api/logs/files and GET /api/logs cursor pagination."""
    files_resp = client.get("/api/logs/files")
    assert files_resp.status_code == 200
    files_data = files_resp.json()
    assert "files" in files_data

    # Test reading logs
    logs_resp = client.get("/api/logs?limit=20")
    assert logs_resp.status_code == 200
    logs_data = logs_resp.json()
    assert "file" in logs_data
    assert "lines" in logs_data
    assert "total_lines" in logs_data

