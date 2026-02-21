"""Telemetry quality scoring for DC3S reliability weight."""
from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Mapping


_TS_KEYS = ("ts_utc", "utc_timestamp", "timestamp", "ts")
_NON_SIGNAL_KEYS = {"device_id", "zone_id", "target"}


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_ts(event: Mapping[str, Any]) -> datetime | None:
    for key in _TS_KEYS:
        if key in event:
            ts = _parse_ts(event.get(key))
            if ts is not None:
                return ts
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _signal_keys(event: Mapping[str, Any], last_event: Mapping[str, Any] | None) -> list[str]:
    keys: set[str] = set()
    for payload in (event, last_event or {}):
        for key, value in payload.items():
            if key in _TS_KEYS or key in _NON_SIGNAL_KEYS:
                continue
            if value is None or _is_number(value):
                keys.add(key)
    return sorted(keys)


def compute_reliability(
    event: Mapping[str, Any],
    last_event: Mapping[str, Any] | None,
    expected_cadence_s: float,
    reliability_cfg: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compute telemetry reliability weight w_t in [min_w, 1].

    Components:
    - missing fraction across numeric telemetry signals
    - delay penalty for late arrivals
    - out-of-order penalty
    - spike penalty from abrupt relative jumps
    """
    cfg = reliability_cfg or {}
    lambda_delay = float(cfg.get("lambda_delay", 0.002))
    spike_beta = float(cfg.get("spike_beta", 0.25))
    ooo_gamma = float(cfg.get("ooo_gamma", 0.35))
    min_w = float(cfg.get("min_w", 0.05))

    keys = _signal_keys(event, last_event)
    present = 0
    for key in keys:
        value = event.get(key)
        if _is_number(value):
            present += 1
    missing_fraction = 0.0 if not keys else 1.0 - (present / float(len(keys)))

    ts_now = _extract_ts(event)
    ts_prev = _extract_ts(last_event or {})
    delay_seconds = 0.0
    out_of_order = False
    if ts_now is not None and ts_prev is not None:
        delta = (ts_now - ts_prev).total_seconds()
        if delta < 0:
            out_of_order = True
            delay_seconds = float(expected_cadence_s)
        else:
            delay_seconds = max(0.0, float(delta - expected_cadence_s))

    spike_ratio = 0.0
    spike_detected = False
    if last_event:
        for key in keys:
            cur = event.get(key)
            prev = last_event.get(key)
            if not (_is_number(cur) and _is_number(prev)):
                continue
            denom = max(abs(float(prev)), 1e-6)
            ratio = abs(float(cur) - float(prev)) / denom
            if ratio > spike_ratio:
                spike_ratio = ratio
        spike_detected = spike_ratio > spike_beta

    missing_penalty = max(0.0, 1.0 - missing_fraction)
    delay_penalty = math.exp(-lambda_delay * max(0.0, delay_seconds))
    ooo_penalty = (1.0 - ooo_gamma) if out_of_order else 1.0
    spike_penalty = 1.0 if spike_ratio <= spike_beta else 1.0 / (1.0 + (spike_ratio - spike_beta))

    w_raw = missing_penalty * delay_penalty * ooo_penalty * spike_penalty
    w_t = max(min_w, min(1.0, float(w_raw)))

    flags = {
        "missing_fraction": float(missing_fraction),
        "delay_seconds": float(delay_seconds),
        "out_of_order": bool(out_of_order),
        "spike_detected": bool(spike_detected),
        "spike_ratio": float(spike_ratio),
        "components": {
            "missing_penalty": float(missing_penalty),
            "delay_penalty": float(delay_penalty),
            "ooo_penalty": float(ooo_penalty),
            "spike_penalty": float(spike_penalty),
        },
    }
    return w_t, flags
