"""Tracker Failure Simulator — API entrypoint.

Start with:  uvicorn app.main:app --port 8000   (from backend/)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.router.api import router

app = FastAPI(title="Tracker Failure Simulator", version="0.1.0",
              description="See *why* a tracking algorithm fails, with every "
                          "tracker and every hyperparameter exposed and explained.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Tracker Failure Simulator. See /api/trackers and /api/docs."}