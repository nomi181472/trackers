"""Tracker Failure Simulator — API entrypoint with end-to-end logging & tracing.

Start with:  uvicorn app.main:app --port 8000   (from backend/)
"""
from __future__ import annotations

import logging
import time
import traceback
import uuid

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core import jobs
from app.logging_config import request_id_ctx, setup_logging
from app.router.api import router

from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html

# Initialize global structured logger and exception hooks
logger = setup_logging(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On process boot: recover any jobs that were left running or queued before process restart
    recovered = jobs.recover_interrupted_jobs()
    if recovered:
        logger.info("Server startup: recovered %d interrupted job(s) from previous process", recovered)
    yield
    # On process shutdown: cleanly shutdown worker threads
    jobs.shutdown_executor(wait=False, cancel_futures=True)


app = FastAPI(
    title="Tracker Failure Simulator API",
    version="0.1.0",
    description="Interactive visual and programmatic lab to see *why* and *how* multi-object trackers fail. "
                "Inspect hyperparameter knobs, track errors, and retrieve daily application logs.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Assigns unique trace ID to each HTTP request, logs timing, and handles uncaught exceptions."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:10]
        token = request_id_ctx.set(req_id)
        start_time = time.perf_counter()

        client_host = request.client.host if request.client else "unknown"
        logger.info("--> %s %s from %s", request.method, request.url.path, client_host)

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers["X-Request-ID"] = req_id
            logger.info(
                "<-- %s %s responded %s in %.2f ms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
            return response
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_type = type(exc).__name__
            tb_str = traceback.format_exc()
            logger.critical(
                "CRASH on %s %s after %.2f ms [%s]: %s\nTraceback:\n%s",
                request.method,
                request.url.path,
                elapsed_ms,
                err_type,
                exc,
                tb_str,
            )
            return JSONResponse(
                status_code=500,
                headers={"X-Request-ID": req_id},
                content={
                    "detail": "Internal Server Error",
                    "error": str(exc),
                    "error_type": err_type,
                    "request_id": req_id,
                },
            )
        finally:
            request_id_ctx.reset(token)


app.add_middleware(RequestTracingMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Fallback handler to ensure any exception escaping endpoint handlers is logged with traceback."""
    req_id = request_id_ctx.get() or "unknown"
    err_type = type(exc).__name__
    tb_str = traceback.format_exc()
    logger.critical(
        "Unhandled exception caught by global handler on %s %s [%s]: %s\n%s",
        request.method,
        request.url.path,
        err_type,
        exc,
        tb_str,
    )
    return JSONResponse(
        status_code=500,
        headers={"X-Request-ID": req_id},
        content={
            "detail": "Internal Server Error",
            "error": str(exc),
            "error_type": err_type,
            "request_id": req_id,
        },
    )


app.include_router(router)


@app.get("/api/docs", include_in_schema=False)
def api_swagger_ui():
    """Swagger UI accessible at /api/docs."""
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Tracker Failure Simulator API — Swagger UI")


@app.get("/api/redoc", include_in_schema=False)
def api_redoc():
    """ReDoc documentation accessible at /api/redoc."""
    return get_redoc_html(openapi_url="/openapi.json", title="Tracker Failure Simulator API — ReDoc")


@app.get("/")
def root():
    return {
        "message": "Tracker Failure Simulator API",
        "docs": "/docs",
        "api_docs": "/api/docs",
        "redoc": "/redoc",
        "trackers": "/api/trackers",
    }