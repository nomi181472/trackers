"""Logging and exception tracing configuration for Tracker Failure Simulator.

Designed for standard Docker and cloud environments:
- Logs to STDOUT for native container log viewer capture.
- Also logs to a rotating file in data/logs/app.log for container-local diagnostics.
- Integrates contextvars so HTTP request IDs and Job IDs are injected automatically into log lines.
- Registers global uncaught exception handlers for main thread and worker threads.
"""
from __future__ import annotations

import contextvars
import logging
import logging.handlers
import sys
import threading
from typing import Optional

from app import config

# Context variable for tracing request / job lifecycle
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


class TraceContextFilter(logging.Filter):
    """Injects current request_id / trace_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        req_id = request_id_ctx.get()
        record.request_id = req_id if req_id else "-"
        return True


import os

# Determine console log level: in production / serverless (Render, Vercel, or ENV=production),
# default to WARNING so that INFO logs do not clutter the console.
_env_name = os.getenv("ENVIRONMENT") or os.getenv("ENV") or os.getenv("NODE_ENV") or ""
_is_prod = _env_name.lower() in ("production", "prod") or bool(os.getenv("VERCEL") or os.getenv("RENDER"))
_default_console_level = logging.WARNING if _is_prod else logging.INFO

_custom_level_str = os.getenv("LOG_LEVEL", "").upper()
_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
CONSOLE_LOG_LEVEL = _level_map.get(_custom_level_str, _default_console_level)


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configures root and app loggers with structured formatting and tracing."""
    root_logger = logging.getLogger()
    
    # Avoid duplicate handlers if setup_logging is called multiple times (e.g. in tests/reload)
    if getattr(root_logger, "_tracker_logging_initialized", False):
        return logging.getLogger("tracker_app")

    # Set root logger to capture all required levels down to DEBUG/INFO
    root_logger.setLevel(min(log_level, CONSOLE_LOG_LEVEL))

    log_format = (
        "[%(asctime)s] [%(levelname)s] [trace:%(request_id)s] "
        "[%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    trace_filter = TraceContextFilter()

    # 1. Console Handler (Standard Output / Error):
    # Only displays WARNING, ERROR, and CRITICAL in production environments
    console_handler = logging.StreamHandler()
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(trace_filter)
    root_logger.addHandler(console_handler)

    # 2. Daily Rotating File Handler (1 file per day, max 10 days preserved)
    try:
        from app.core.daily_logger import DailyRotatingFileHandler
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = DailyRotatingFileHandler(
            logs_dir=config.LOGS_DIR,
            max_days=10,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(trace_filter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If filesystem permissions or disk quota prevent file logging, continue with console logging
        sys.stderr.write(f"Warning: Could not initialize file logging in {config.LOGS_DIR}: {e}\n")

    # Mute overly verbose third-party loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Hook uncaught exceptions
    def handle_sys_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical(
            "Uncaught process-level exception:",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_sys_exception

    def handle_thread_exception(args: threading.ExceptHookArgs):
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        root_logger.critical(
            f"Uncaught thread exception in thread '{args.thread.name}':",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = handle_thread_exception

    root_logger._tracker_logging_initialized = True
    app_logger = logging.getLogger("tracker_app")
    app_logger.info("Logging initialized. Output routed to STDOUT and %s", config.LOGS_DIR / "app.log")
    return app_logger
