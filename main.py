"""Root ASGI entrypoint exporting FastAPI app instance for Render / platform deploys."""
import sys
from pathlib import Path

# Add backend directory to sys.path so 'app' can always be imported from repository root
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app

__all__ = ["app"]
