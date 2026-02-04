"""Monitoring alert helpers."""
from __future__ import annotations

import logging
from typing import Mapping

from gridpulse.utils.net import get_session


def send_webhook(url: str, payload: Mapping[str, object], retries: int = 3, backoff: float = 0.5) -> None:
    """Send a JSON payload to a webhook endpoint with retries."""
    log = logging.getLogger(__name__)
    session = get_session(retries=retries, backoff=backoff)
    try:
        resp = session.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        log.info("Alert webhook sent to %s", url)
    except Exception as exc:
        log.error("Alert webhook failed for %s", url, exc_info=exc)
        raise
