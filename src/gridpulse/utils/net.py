"""Utilities: HTTP sessions with retries."""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_session(
    retries: int = 3,
    backoff: float = 0.5,
    status_forcelist: Iterable[int] = (429, 500, 502, 503, 504),
) -> requests.Session:
    retry_cfg = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=tuple(status_forcelist),
        allowed_methods=("GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_cfg)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, object] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int | float = 60,
    log: logging.Logger | None = None,
) -> dict:
    logger = log or logging.getLogger(__name__)
    try:
        resp = session.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("HTTP request failed for %s", url, exc_info=exc)
        raise
    try:
        return resp.json()
    except ValueError as exc:
        logger.error("Invalid JSON response from %s", url, exc_info=exc)
        raise

