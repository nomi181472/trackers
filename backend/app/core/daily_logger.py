"""Daily rotating file logger with 10-day retention and structured tracing."""
from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Optional


class DailyRotatingFileHandler(logging.Handler):
    """Logs to a daily file (app_YYYY-MM-DD.log).
    
    - Changes files automatically when the date changes.
    - Preserves at most max_days files (oldest are deleted).
    """

    def __init__(self, logs_dir: Path, max_days: int = 10, encoding: str = "utf-8"):
        super().__init__()
        self.logs_dir = Path(logs_dir)
        self.max_days = max_days
        self.encoding = encoding
        self._current_date: Optional[str] = None
        self._stream = None

    def _get_today_str(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d")

    def _cleanup_old_files(self) -> None:
        """Removes log files older than max_days."""
        try:
            log_files = sorted(self.logs_dir.glob("app_*.log"))
            if len(log_files) > self.max_days:
                to_delete = log_files[:-self.max_days]
                for f in to_delete:
                    try:
                        f.unlink(missing_ok=True)
                    except OSError:
                        pass
        except Exception:
            pass

    def _ensure_stream(self) -> None:
        today = self._get_today_str()
        if today != self._current_date or self._stream is None:
            if self._stream:
                try:
                    self._stream.close()
                except Exception:
                    pass
            self._current_date = today
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.logs_dir / f"app_{today}.log"
            self._stream = open(file_path, "a", encoding=self.encoding)
            self._cleanup_old_files()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_stream()
            msg = self.format(record) + "\n"
            self._stream.write(msg)
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        super().close()
