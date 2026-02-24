"""Metrics for CPSBench-IoT forecast quality, control safety, and trace completeness."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def _as_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).reshape(-1)


def _picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def compute_forecast_metrics(
    *,
    y_true: Iterable[float] | np.ndarray,
    y_pred: Iterable[float] | np.ndarray,
    lower_90: Iterable[float] | np.ndarray,
    upper_90: Iterable[float] | np.ndarray,
    lower_95: Iterable[float] | np.ndarray | None = None,
    upper_95: Iterable[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute MAE/RMSE and interval calibration metrics."""
    yt = _as_array(y_true)
    yp = _as_array(y_pred)
    lo90 = _as_array(lower_90)
    hi90 = _as_array(upper_90)
    if not (len(yt) == len(yp) == len(lo90) == len(hi90)):
        raise ValueError("Forecast arrays must have identical length")

    if lower_95 is None or upper_95 is None:
        half = 0.5 * (hi90 - lo90)
        mid = 0.5 * (hi90 + lo90)
        lo95 = mid - 1.2 * half
        hi95 = mid + 1.2 * half
    else:
        lo95 = _as_array(lower_95)
        hi95 = _as_array(upper_95)
        if not (len(lo95) == len(hi95) == len(yt)):
            raise ValueError("95% interval arrays must match y_true length")

    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    picp90 = _picp(yt, lo90, hi90)
    picp95 = _picp(yt, lo95, hi95)
    mean_width = float(np.mean(hi90 - lo90))
    return {
        "mae": mae,
        "rmse": rmse,
        "picp_90": picp90,
        "picp_95": picp95,
        "mean_interval_width": mean_width,
    }


def _compute_recovery_time(violations: np.ndarray, event_log: pd.DataFrame | None) -> float:
    if event_log is None or event_log.empty:
        return 0.0
    fault_cols = [c for c in event_log.columns if c in {"dropout", "delay_jitter", "out_of_order", "spikes", "stale_sensor", "covariate_drift", "label_drift"}]
    if not fault_cols:
        return 0.0
    active_fault = event_log[fault_cols].sum(axis=1).to_numpy(dtype=float) > 0.0
    idx_fault = np.where(active_fault)[0]
    if idx_fault.size == 0:
        return 0.0

    recovery_steps: list[float] = []
    for idx in idx_fault:
        segment = violations[idx:]
        cleared = np.where(~segment)[0]
        if cleared.size > 0:
            recovery_steps.append(float(cleared[0]))
    if not recovery_steps:
        return float(len(violations))
    return float(np.mean(recovery_steps))


def compute_control_metrics(
    *,
    proposed_charge_mw: Iterable[float] | np.ndarray,
    proposed_discharge_mw: Iterable[float] | np.ndarray,
    safe_charge_mw: Iterable[float] | np.ndarray,
    safe_discharge_mw: Iterable[float] | np.ndarray,
    soc_mwh: Iterable[float] | np.ndarray,
    constraints: Mapping[str, Any],
    event_log: pd.DataFrame | None = None,
    intervention_eps: float = 1e-6,
) -> dict[str, float]:
    """Compute safety/control metrics including violation and intervention rates."""
    p_ch = _as_array(proposed_charge_mw)
    p_dis = _as_array(proposed_discharge_mw)
    s_ch = _as_array(safe_charge_mw)
    s_dis = _as_array(safe_discharge_mw)
    soc = _as_array(soc_mwh)
    n = len(soc)
    if not (len(p_ch) == len(p_dis) == len(s_ch) == len(s_dis) == n):
        raise ValueError("Control arrays must have identical length")

    max_power = float(constraints.get("max_power_mw", max(np.max(s_ch, initial=0.0), np.max(s_dis, initial=0.0))))
    min_soc = float(constraints.get("min_soc_mwh", np.min(soc, initial=0.0)))
    max_soc = float(constraints.get("max_soc_mwh", np.max(soc, initial=0.0)))
    ramp_mw = float(constraints.get("ramp_mw", 0.0))

    net = s_dis - s_ch
    ramp_violation = np.zeros(n, dtype=bool)
    ramp_excess = np.zeros(n, dtype=float)
    if ramp_mw > 0 and n > 1:
        delta = np.abs(np.diff(net))
        excess = np.maximum(0.0, delta - ramp_mw)
        ramp_violation[1:] = excess > 1e-9
        ramp_excess[1:] = excess

    power_violation = (s_ch > max_power + 1e-9) | (s_dis > max_power + 1e-9) | ((s_ch > 1e-9) & (s_dis > 1e-9))
    soc_violation = (soc < min_soc - 1e-9) | (soc > max_soc + 1e-9)
    violations = power_violation | soc_violation | ramp_violation

    severity = (
        np.maximum(0.0, s_ch - max_power)
        + np.maximum(0.0, s_dis - max_power)
        + np.maximum(0.0, min_soc - soc)
        + np.maximum(0.0, soc - max_soc)
        + ramp_excess
    )
    interventions = (np.abs(p_ch - s_ch) > intervention_eps) | (np.abs(p_dis - s_dis) > intervention_eps)
    recovery_time = _compute_recovery_time(violations=violations, event_log=event_log)

    return {
        "violation_rate": float(np.mean(violations)),
        "violation_severity": float(np.sum(severity)),
        "recovery_time": float(recovery_time),
        "intervention_rate": float(np.mean(interventions)),
    }


def compute_trace_metrics(
    certificates: Iterable[Mapping[str, Any] | None],
    required_fields: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute certificate availability and missing-field counts."""
    req = list(required_fields or ["command_id", "certificate_hash", "proposed_action", "safe_action"])
    certs = list(certificates)
    if not certs:
        return {"certificate_presence_rate": 0.0, "certificate_missing_fields": float(len(req))}

    present = [c for c in certs if isinstance(c, Mapping)]
    presence_rate = float(len(present) / len(certs))
    missing_fields_total = 0
    for cert in present:
        missing_fields_total += sum(1 for field in req if field not in cert)
    if not present:
        missing_fields_total = len(req) * len(certs)
    return {
        "certificate_presence_rate": presence_rate,
        "certificate_missing_fields": float(missing_fields_total),
    }


def compute_all_metrics(
    *,
    y_true: Iterable[float] | np.ndarray,
    y_pred: Iterable[float] | np.ndarray,
    lower_90: Iterable[float] | np.ndarray,
    upper_90: Iterable[float] | np.ndarray,
    proposed_charge_mw: Iterable[float] | np.ndarray,
    proposed_discharge_mw: Iterable[float] | np.ndarray,
    safe_charge_mw: Iterable[float] | np.ndarray,
    safe_discharge_mw: Iterable[float] | np.ndarray,
    soc_mwh: Iterable[float] | np.ndarray,
    constraints: Mapping[str, Any],
    certificates: Iterable[Mapping[str, Any] | None],
    event_log: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Aggregate all CPSBench metric families into one dict."""
    metrics = {}
    metrics.update(
        compute_forecast_metrics(
            y_true=y_true,
            y_pred=y_pred,
            lower_90=lower_90,
            upper_90=upper_90,
        )
    )
    metrics.update(
        compute_control_metrics(
            proposed_charge_mw=proposed_charge_mw,
            proposed_discharge_mw=proposed_discharge_mw,
            safe_charge_mw=safe_charge_mw,
            safe_discharge_mw=safe_discharge_mw,
            soc_mwh=soc_mwh,
            constraints=constraints,
            event_log=event_log,
        )
    )
    metrics.update(compute_trace_metrics(certificates))
    return metrics

