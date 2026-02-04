"""Utilities: logging."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional


class JsonFormatter(logging.Formatter):
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
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    stream: Optional[object] = None,
    file_path: Optional[str] = None,
) -> None:
    level_name = (log_level or os.getenv("GRIDPULSE_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = (log_format or os.getenv("GRIDPULSE_LOG_FORMAT", "text")).lower()
    use_json = fmt == "json"

    root = logging.getLogger()
    if getattr(root, "_gridpulse_configured", False):
        return

    root.setLevel(level)
    root.handlers = []

    formatter = JsonFormatter() if use_json else logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler(stream or sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    root._gridpulse_configured = True


def get_logger(name: str) -> logging.Logger:
    # Key: shared utilities used across the pipeline
    setup_logging()
    return logging.getLogger(name)
