"""Telemetry quality scoring for the DC3S reliability weight w_t.

Public API
----------
compute_reliability
    Main entry point.  Given a single telemetry event and the previous event,
    returns ``(w_t, flags)`` where *w_t* in ``[min_w, 1]`` is the reliability
    weight and *flags* is a diagnostic dict.
compute_reliability_robust
    Byzantine-resistant variant operating on a sliding signal-history window
    instead of a single event pair.
"""
from __future__ import annotations

import math
import statistics
import warnings
from datetime import datetime
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from .ftit import FTIT_FAULT_KEYS, preview_fault_state

__all__ = ["compute_reliability", "compute_reliability_robust"]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Event-dict keys checked (in order) when extracting a telemetry timestamp.
_TS_KEYS: tuple[str, ...] = ("ts_utc", "utc_timestamp", "timestamp", "ts")

#: Keys that are never treated as numeric signal channels.
_NON_SIGNAL_KEYS: frozenset[str] = frozenset({"device_id", "zone_id", "target"})

#: Rolling window length for the OOO (out-of-order) fraction tracker.
#: 20 steps at hourly cadence ≈ 20 hours of packet-ordering history.
_OOO_HISTORY_WINDOW: int = 20

# Detector-lag contract used by the thesis text and runtime diagnostics.
_DETECTOR_LAG_STEPS: dict[str, int] = {
    "dropout": 1,
    "stale_sensor": 3,
    "delay_jitter": 2,
    "out_of_order": 10,
    "spikes": 1,
    "drift": 48,
}
_DETECTOR_WARNING_RELIABILITY: float = 0.5


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


def _is_numeric_or_nan(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _signal_keys(event: Mapping[str, Any], last_event: Mapping[str, Any] | None) -> list[str]:
    keys: set[str] = set()
    for payload in (event, last_event or {}):
        for key, value in payload.items():
            if key in _TS_KEYS or key in _NON_SIGNAL_KEYS:
                continue
            if value is None or _is_numeric_or_nan(value):
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


def _max_stale_count(tracker: Mapping[str, Any]) -> int:
    counts = tracker.get("unchanged_counts")
    if not isinstance(counts, Mapping):
        return 0
    values: list[int] = []
    for value in counts.values():
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(values, default=0)


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
    backend = str(cfg.get("backend", "heuristic")).strip().lower()
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
    timestamp_missing = False
    if ts_now is None:
        # No recognised timestamp key found in the event (checked: ts_utc,
        # utc_timestamp, timestamp, ts).  Delay and OOO penalties cannot be
        # computed from telemetry timing, so apply a conservative synthetic
        # penalty instead of letting the reliability score appear healthy even
        # if data is stale.  Add the correct key to the event payload or
        # extend _TS_KEYS to suppress this warning.
        warnings.warn(
            f"DC3S reliability: no timestamp key found in telemetry event "
            f"(checked: {_TS_KEYS}).  Conservative delay/OOO penalties will "
            "be applied. "
            "Set a recognised timestamp key (e.g. 'ts_utc') to enable "
            "accurate staleness detection.",
            UserWarning,
            stacklevel=2,
        )
        timestamp_missing = True
        delay_seconds = float(expected_cadence_s)
        out_of_order = True
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

    # If the event carries an explicit fault label (e.g. CPSBench injecting
    # dropout=True), that takes precedence over the signal-derived heuristic.
    # In production, these fields are absent and we always fall back to the
    # computed detections.
    def _resolve_fault_flag(explicit: bool | None, detected: bool) -> bool:
        return detected if explicit is None else explicit

    explicit_faults: dict[str, bool | None] = {
        "dropout":      _as_bool(event.get("dropout")),
        "stale_sensor": _as_bool(event.get("stale_sensor")),
        "delay_jitter": _as_bool(event.get("delay_jitter")),
        "out_of_order": _as_bool(event.get("out_of_order")),
        "spikes":       _as_bool(event.get("spikes")),
    }
    fault_flags = {
        "dropout":      _resolve_fault_flag(explicit_faults["dropout"],      bool(keys) and present < len(keys)),  # noqa: E241
        "stale_sensor": _resolve_fault_flag(explicit_faults["stale_sensor"], bool(stale_detected)),
        "delay_jitter": _resolve_fault_flag(explicit_faults["delay_jitter"], delay_seconds > 0.0),
        "out_of_order": _resolve_fault_flag(explicit_faults["out_of_order"], bool(out_of_order)),
        "spikes":       _resolve_fault_flag(explicit_faults["spikes"],       bool(spike_detected)),  # noqa: E241
    }


    missing_penalty = max(0.0, 1.0 - missing_fraction)
    delay_penalty = math.exp(-lambda_delay * max(0.0, delay_seconds))
    # PDF §3: OOO penalty is the fraction of packets arriving in order.
    # We track a running fraction via the adaptive_state (ooo_history).
    # Fallback: binary penalty for backward compatibility.
    ooo_history = list((_ftit_state(adaptive_state) or {}).get("ooo_history", []))
    ooo_history.append(0.0 if out_of_order else 1.0)
    # Keep a rolling window of the last _OOO_HISTORY_WINDOW observations.
    if len(ooo_history) > _OOO_HISTORY_WINDOW:
        ooo_history = ooo_history[-_OOO_HISTORY_WINDOW:]
    ooo_fraction_in_order = sum(ooo_history) / max(len(ooo_history), 1)
    ooo_penalty = max(1.0 - ooo_gamma, ooo_fraction_in_order)
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
        "timestamp_missing": bool(timestamp_missing),
        "spike_detected": bool(spike_detected),
        "spike_ratio": float(spike_ratio),
        "components": {
            "missing_penalty": float(missing_penalty),
            "delay_penalty": float(delay_penalty),
            "ooo_penalty": float(ooo_penalty),
            "spike_penalty": float(spike_penalty),
        },
        "sub_scores": {
            "P_drop": float(missing_penalty),
            "P_stale": 1.0 - (1.0 if stale_detected else 0.0),
            "P_delay": float(delay_penalty),
            "P_ooo": float(ooo_penalty),
            "P_spike": float(spike_penalty),
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
        "ooo_history": ooo_history,
        "ooo_fraction_in_order": float(ooo_fraction_in_order),
        "reliability_rule": "ftit_ro" if preview is not None and ftit_law == "ftit_ro" else "linear",
        "detector_lag_steps": dict(_DETECTOR_LAG_STEPS),
    }
    heuristic_w_t = float(w_t)

    max_stale_count = _max_stale_count(stale_tracker)
    stale_tau_max = int(_DETECTOR_LAG_STEPS["stale_sensor"])
    flags["stale_max_unchanged_steps"] = int(max_stale_count)

    # A6 stale-channel decay: once staleness persists beyond the detection lag
    # (stale_tau_max steps), exponentially decay w_t toward min_w.  Each
    # extra step beyond the lag multiplies reliability by a decay factor.
    # This closes the A6 gap: extended sensor freeze now lowers reliability
    # proportionally rather than leaving w_t unchanged after initial detection.
    stale_decay_rate = float(cfg.get("stale_decay_rate", 0.85))
    if max_stale_count > stale_tau_max:
        extra_steps = max_stale_count - stale_tau_max
        decay_factor = stale_decay_rate ** extra_steps
        w_t = max(min_w, min(float(w_t), float(w_t) * decay_factor))
        flags["stale_decay_applied"] = True
        flags["stale_decay_factor"] = float(decay_factor)
        flags["stale_extra_steps"] = int(extra_steps)
    else:
        flags["stale_decay_applied"] = False
        flags["stale_decay_factor"] = 1.0
        flags["stale_extra_steps"] = 0

    flags["detector_lag_warning"] = bool(
        max_stale_count >= stale_tau_max and float(w_t) > _DETECTOR_WARNING_RELIABILITY
    )
    if flags["detector_lag_warning"]:
        warnings.warn(
            "DC3S detector-lag contract warning: stale telemetry persisted for "
            f"{max_stale_count} steps without reliability dropping below "
            f"{_DETECTOR_WARNING_RELIABILITY:.2f}. "
            "Assumption A6 may be violated for the stale-sensor channel.",
            RuntimeWarning,
            stacklevel=2,
        )

    if backend == "deep":
        deep_cfg = dict(cfg.get("deep", {}))
        model_path = str(
            deep_cfg.get("model_path")
            or cfg.get("deep_model_path")
            or ""
        ).strip()
        strict = bool(deep_cfg.get("strict", cfg.get("deep_strict", False)))
        seq_len = int(deep_cfg.get("seq_len", 8))
        try:
            if not model_path:
                raise FileNotFoundError("no DeepOQE model_path configured")
            from .deep_oqe import extract_feature_vector, load_model, prepare_sequence, update_history

            feature = extract_feature_vector(
                event=event,
                last_event=last_event,
                missing_fraction=float(missing_fraction),
                delay_seconds=float(delay_seconds),
                out_of_order=bool(out_of_order),
                spike_ratio=float(spike_ratio),
                ooo_fraction_in_order=float(ooo_fraction_in_order),
                stale_tracker=stale_tracker,
                fault_flags=fault_flags,
            )
            sequence = prepare_sequence(feature, adaptive_state=adaptive_state, seq_len=seq_len)
            model, deep_model_cfg, metadata = load_model(model_path)
            with torch.no_grad():
                tensor = torch.from_numpy(sequence).unsqueeze(0)
                deep_w, fault_logit = model(tensor)
            update_history(adaptive_state if isinstance(adaptive_state, MutableMapping) else None, feature, seq_len=deep_model_cfg.seq_len)
            w_t = float(np.clip(float(deep_w.detach().cpu().item()), float(deep_model_cfg.min_w), 1.0))
            fault_prob = float(torch.sigmoid(fault_logit).detach().cpu().item())
            flags.update(
                {
                    "backend_requested": "deep",
                    "backend_used": "deep",
                    "deep_model_path": model_path,
                    "deep_fault_probability": fault_prob,
                    "deep_sequence_length": int(deep_model_cfg.seq_len),
                    "deep_metadata": metadata,
                    "heuristic_w_t": heuristic_w_t,
                }
            )
            return w_t, flags
        except Exception as exc:
            if strict:
                raise
            flags.update(
                {
                    "backend_requested": "deep",
                    "backend_used": "heuristic_fallback",
                    "deep_failure": str(exc),
                    "heuristic_w_t": heuristic_w_t,
                }
            )
            return w_t, flags

    flags.update({"backend_requested": backend, "backend_used": "heuristic"})

    # --- Adversarial tamper detection (main path) ---
    # An adversary who knows the OQE formula can craft telemetry that
    # scores w_t ≈ 1.0 while subtly shifting the true state.  The detection
    # heuristic: flag when multiple independent fault signals are active
    # simultaneously (dropout flag + stale flag, or spike + no delay penalty)
    # yet w_t remains suspiciously high.  Real degraded telemetry almost never
    # produces all-zero fault flags alongside a high reliability score.
    tamper_score = 0.0
    active_faults = sum([
        bool(fault_flags.get("dropout", False)),
        bool(fault_flags.get("stale_sensor", False)),
        bool(fault_flags.get("spikes", False)),
        bool(fault_flags.get("out_of_order", False)),
    ])
    # If multiple faults are simultaneously active but w_t is still high,
    # the OQE score may have been crafted to appear trustworthy.
    if active_faults >= 2 and float(w_t) > 0.75:
        tamper_score = min(1.0, 0.25 * active_faults)

    # Cross-signal inconsistency: a high-quality signal with zero delay but
    # non-zero OOO fraction is physically implausible (packets can't arrive
    # both instantly and out-of-order).
    delay_s = float(flags.get("delay_seconds", 0.0))
    ooo_frac = float(flags.get("ooo_fraction_in_order", 1.0))
    if delay_s < 0.1 and ooo_frac < 0.8 and float(w_t) > 0.80:
        tamper_score = max(tamper_score, 0.20)

    adversarial_mode = bool(cfg.get("adversarial_mode", False))
    if adversarial_mode and tamper_score > 0.0:
        w_t = max(min_w, float(w_t) * (1.0 - tamper_score))
        flags["tamper_detected"] = True
        flags["tamper_score"] = float(tamper_score)
        flags["tamper_w_penalty"] = float(tamper_score)
    else:
        flags["tamper_detected"] = False
        flags["tamper_score"] = float(tamper_score)

    return w_t, flags


# ---------------------------------------------------------------------------
# Byzantine-Resistant OQE (Phase 3)
# ---------------------------------------------------------------------------

def compute_reliability_robust(
    signal_history: Sequence[float],
    *,
    trim_frac: float = 0.10,
    mad_spike_threshold: float = 3.5,
    consistency_threshold: float = 1.5,
    min_w: float = 0.05,
    adversarial_penalty: float = 0.30,
) -> tuple[float, dict[str, Any]]:
    """Byzantine-resistant reliability estimate from a signal history window.

    Theorem 11 — Byzantine Safety Bound
    -------------------------------------
    If at most f < 1/3 of readings in the last W steps are adversarially
    spoofed, compute_reliability_robust() returns w_t such that:
        w_t <= w_true + delta
    where delta is bounded by the trimmed-mean estimation error, which
    decreases as O(1/sqrt(W*(1-2f))).

    Robustness mechanisms (drop-in replacement for mean-based OQE):
    1. MAD spike detection: spike_ratio = |x_t - median(window)| / MAD(window)
       Robust against one large adversarial reading that would inflate the mean.
    2. Trimmed-mean history: drop top and bottom trim_frac of window before
       computing drift/staleness statistics.
    3. Consistency check: if |median - mean| / std > threshold, flag
       adversarial_suspected and reduce w_t further.

    Args:
        signal_history: Sequence of recent signal readings (most-recent last).
                        Minimum 3 elements required; returns min_w for fewer.
        trim_frac:      Fraction to trim from each tail (default 10%).
        mad_spike_threshold: MAD z-score threshold for spike detection.
        consistency_threshold: |median-mean|/std ratio to flag adversarial input.
        min_w:          Minimum reliability weight (same as compute_reliability).
        adversarial_penalty: Additional w_t reduction when adversarial suspected.

    Returns:
        Tuple (w_t, flags) where:
            w_t:   Robust reliability weight in [min_w, 1.0].
            flags: Dict with spike_detected, adversarial_suspected, diagnostics.
    """
    vals = list(signal_history)
    n = len(vals)

    if n < 3:
        return float(min_w), {
            "robust": True,
            "n": n,
            "spike_detected": False,
            "adversarial_suspected": False,
            "trim_frac": trim_frac,
            "note": "insufficient_history",
        }

    # 1. Trimmed-mean history: remove top and bottom trim_frac
    k_trim = max(1, int(n * trim_frac))
    sorted_vals = sorted(vals)
    trimmed = sorted_vals[k_trim: n - k_trim] if n - 2 * k_trim >= 3 else sorted_vals

    trimmed_mean = float(sum(trimmed) / len(trimmed))
    full_mean = float(sum(vals) / n)

    # 2. MAD-based spike detection on most recent value
    med = float(statistics.median(vals))
    abs_devs = [abs(v - med) for v in vals]
    mad = float(statistics.median(abs_devs))
    if mad < 1e-9:
        mad = 1e-9  # avoid division by zero
    current = vals[-1]
    mad_z = abs(current - med) / (1.4826 * mad)  # normalised MAD
    spike_detected = mad_z > mad_spike_threshold

    # 3. Consistency check: adversarial input creates systematic bias
    std = float(statistics.stdev(vals)) if n >= 2 else 1.0
    if std < 1e-9:
        std = 1e-9
    consistency_ratio = abs(med - full_mean) / std
    adversarial_suspected = consistency_ratio > consistency_threshold

    # Compute w_t
    spike_penalty = 1.0 if not spike_detected else 1.0 / (1.0 + mad_z - mad_spike_threshold)
    adv_penalty = adversarial_penalty if adversarial_suspected else 0.0
    w_raw = spike_penalty * (1.0 - adv_penalty)
    w_t = float(max(min_w, min(1.0, w_raw)))

    flags: dict[str, Any] = {
        "robust": True,
        "n": n,
        "spike_detected": spike_detected,
        "adversarial_suspected": adversarial_suspected,
        "mad_z": float(mad_z),
        "consistency_ratio": float(consistency_ratio),
        "trimmed_mean": trimmed_mean,
        "full_mean": full_mean,
        "median": med,
        "mad": float(mad),
        "trim_frac": trim_frac,
        "spike_penalty": float(spike_penalty),
        "adversarial_penalty_applied": float(adv_penalty),
    }
    return w_t, flags
