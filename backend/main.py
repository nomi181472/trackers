"""Vercel / standard ASGI entrypoint exporting FastAPI app instance."""
from app.main import app

__all__ = ["app"]
