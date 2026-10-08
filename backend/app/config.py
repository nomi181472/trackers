import os
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/

# Default data directory to /tmp/tracker_data for serverless (Vercel) & container compatibility.
# Can be overridden via DATA_DIR environment variable.
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/tracker_data"))

SCENARIOS_DIR = DATA_DIR / "scenarios"
JOBS_DIR = DATA_DIR / "jobs"
VIDEOS_DIR = DATA_DIR / "videos"
UPLOADS_DIR = DATA_DIR / "uploads"
LOGS_DIR = DATA_DIR / "logs"
MEDIA_DIRS = [JOBS_DIR, VIDEOS_DIR, SCENARIOS_DIR]

for _d in (SCENARIOS_DIR, JOBS_DIR, VIDEOS_DIR, UPLOADS_DIR, LOGS_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        pass


def new_id() -> str:
    return uuid.uuid4().hex[:10]