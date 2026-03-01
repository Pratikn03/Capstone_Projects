"""Telemetry quality scoring for DC3S reliability weight."""
from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Mapping

from .ftit import FTIT_FAULT_KEYS, preview_fault_state


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


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(float(value))
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "on"}:
            return True
        if low in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _ftit_state(adaptive_state: Mapping[str, Any] | None) -> dict[str, Any]:
    root = dict(adaptive_state or {})
    ftit = root.get("ftit")
    if isinstance(ftit, Mapping):
        return dict(ftit)
    return {}


def _ftit_cfg(ftit_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(ftit_cfg or {})
    inner = payload.get("ftit")
    if isinstance(inner, Mapping):
        merged = dict(inner)
        if "law" in payload:
            merged["law"] = payload["law"]
        return merged
    return payload


def _stale_tracker(
    *,
    event: Mapping[str, Any],
    last_event: Mapping[str, Any] | None,
    keys: list[str],
    adaptive_state: Mapping[str, Any] | None,
    stale_tol: float,
    stale_k: int,
) -> tuple[dict[str, Any], bool]:
    ftit_state = _ftit_state(adaptive_state)
    tracker = ftit_state.get("stale_tracker")
    tracker_map = dict(tracker) if isinstance(tracker, Mapping) else {}
    prev_values = tracker_map.get("last_values")
    prev_counts = tracker_map.get("unchanged_counts")
    if not isinstance(prev_values, Mapping):
        prev_values = {}
    if not isinstance(prev_counts, Mapping):
        prev_counts = {}

    next_values: dict[str, float | None] = {}
    next_counts: dict[str, int] = {}
    stale_detected = False
    threshold = max(int(stale_k) - 1, 1)

    for key in keys:
        cur = event.get(key)
        prev_from_tracker = prev_values.get(key)
        prev_from_event = (last_event or {}).get(key)
        prev = prev_from_tracker if _is_number(prev_from_tracker) else prev_from_event
        cur_num = float(cur) if _is_number(cur) else None
        prev_num = float(prev) if _is_number(prev) else None
        prev_count = int(prev_counts.get(key, 0) or 0)

        if cur_num is not None and prev_num is not None and abs(cur_num - prev_num) <= stale_tol:
            count = prev_count + 1
        else:
            count = 0

        next_values[key] = cur_num
        next_counts[key] = int(count)
        if count >= threshold:
            stale_detected = True

    return {
        "last_values": next_values,
        "unchanged_counts": next_counts,
    }, stale_detected


def compute_reliability(
    event: Mapping[str, Any],
    last_event: Mapping[str, Any] | None,
    expected_cadence_s: float,
    reliability_cfg: Mapping[str, Any] | None = None,
    *,
    adaptive_state: Mapping[str, Any] | None = None,
    ftit_cfg: Mapping[str, Any] | None = None,
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
    ftit_cfg_map = _ftit_cfg(ftit_cfg)
    ftit_law = str(ftit_cfg_map.get("law", "linear")).strip().lower()
    stale_tol = float(ftit_cfg_map.get("stale_tol", 1.0e-9))
    stale_k = max(int(ftit_cfg_map.get("stale_k", 3)), 1)

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

    stale_tracker, stale_detected = _stale_tracker(
        event=event,
        last_event=last_event,
        keys=keys,
        adaptive_state=adaptive_state,
        stale_tol=stale_tol,
        stale_k=stale_k,
    )

    explicit_faults: dict[str, bool | None] = {
        "dropout": _as_bool(event.get("dropout")),
        "stale_sensor": _as_bool(event.get("stale_sensor")),
        "delay_jitter": _as_bool(event.get("delay_jitter")),
        "out_of_order": _as_bool(event.get("out_of_order")),
        "spikes": _as_bool(event.get("spikes")),
    }
    fault_flags = {
        "dropout": explicit_faults["dropout"] if explicit_faults["dropout"] is not None else bool(keys) and present < len(keys),
        "stale_sensor": explicit_faults["stale_sensor"] if explicit_faults["stale_sensor"] is not None else bool(stale_detected),
        "delay_jitter": explicit_faults["delay_jitter"] if explicit_faults["delay_jitter"] is not None else bool(delay_seconds > 0.0),
        "out_of_order": explicit_faults["out_of_order"] if explicit_faults["out_of_order"] is not None else bool(out_of_order),
        "spikes": explicit_faults["spikes"] if explicit_faults["spikes"] is not None else bool(spike_detected),
    }

    missing_penalty = max(0.0, 1.0 - missing_fraction)
    delay_penalty = math.exp(-lambda_delay * max(0.0, delay_seconds))
    ooo_penalty = (1.0 - ooo_gamma) if out_of_order else 1.0
    spike_penalty = 1.0 if spike_ratio <= spike_beta else 1.0 / (1.0 + (spike_ratio - spike_beta))

    w_raw = missing_penalty * delay_penalty * ooo_penalty * spike_penalty
    w_linear = max(min_w, min(1.0, float(w_raw)))

    preview = None
    if ftit_cfg_map:
        preview = preview_fault_state(
            adaptive_state=adaptive_state,
            fault_flags=fault_flags,
            cfg=ftit_cfg_map,
        )
    if preview is not None and ftit_law == "ftit_ro":
        w_t = max(min_w, min(1.0, float(preview["w_t"])))
    else:
        w_t = w_linear

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
        "fault_flags": fault_flags,
        "p_drop": float((preview or {}).get("p", {}).get("dropout", 0.0)),
        "p_stale": float((preview or {}).get("p", {}).get("stale_sensor", 0.0)),
        "p_delay": float((preview or {}).get("p", {}).get("delay_jitter", 0.0)),
        "p_ooo": float((preview or {}).get("p", {}).get("out_of_order", 0.0)),
        "p_spike": float((preview or {}).get("p", {}).get("spikes", 0.0)),
        "smooth_rates": {
            key: float((preview or {}).get("p", {}).get(key, 0.0))
            for key in FTIT_FAULT_KEYS
        },
        "stale_tracker": stale_tracker,
        "reliability_rule": "ftit_ro" if preview is not None and ftit_law == "ftit_ro" else "linear",
    }
    return w_t, flags
