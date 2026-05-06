"""Utilities: structured logging setup."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(
    *,
    log_level: str | None = None,
    log_format: str | None = None,
    stream: object | None = None,
    file_path: str | None = None,
) -> None:
    """Configure root logging with optional JSON output."""
    level_name = (log_level or os.getenv("ORIUS_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = (log_format or os.getenv("ORIUS_LOG_FORMAT", "text")).lower()
    use_json = fmt == "json"

    root = logging.getLogger()
    if getattr(root, "_orius_configured", False):
        return

    root.setLevel(level)
    root.handlers = []

    formatter = (
        JsonFormatter()
        if use_json
        else logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    stream_handler = logging.StreamHandler(stream or sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    root._orius_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger after ensuring global logging is configured."""
    setup_logging()
    return logging.getLogger(name)
