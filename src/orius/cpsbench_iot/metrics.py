"""Metrics for CPSBench-IoT forecast quality, control safety, and trace completeness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


def _as_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).reshape(-1)


def _picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    error = y_true - y_pred
    q = float(quantile)
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def _winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    alpha = max(float(alpha), 1e-9)
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    score = width.copy()
    score[below] = width[below] + (2.0 / alpha) * (lower[below] - y_true[below])
    score[above] = width[above] + (2.0 / alpha) * (y_true[above] - upper[above])
    return float(np.mean(score))


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
    midpoint = 0.5 * (lo90 + hi90)
    return {
        "mae": mae,
        "rmse": rmse,
        "picp_90": picp90,
        "picp_95": picp95,
        "mean_interval_width": mean_width,
        "pinball_loss_q05": _pinball_loss(yt, lo90, 0.05),
        "pinball_loss_q50": _pinball_loss(yt, midpoint, 0.50),
        "pinball_loss_q95": _pinball_loss(yt, hi90, 0.95),
        "pinball_loss_mean": float(
            np.mean(
                [
                    _pinball_loss(yt, lo90, 0.05),
                    _pinball_loss(yt, midpoint, 0.50),
                    _pinball_loss(yt, hi90, 0.95),
                ]
            )
        ),
        "winkler_score_90": _winkler_score(yt, lo90, hi90, 0.10),
    }


def _compute_recovery_time(violations: np.ndarray, event_log: pd.DataFrame | None) -> float:
    if event_log is None or event_log.empty:
        return 0.0
    fault_cols = [
        c
        for c in event_log.columns
        if c
        in {
            "dropout",
            "delay_jitter",
            "out_of_order",
            "spikes",
            "stale_sensor",
            "covariate_drift",
            "label_drift",
        }
    ]
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


def _mean_recovery_steps(violation_mask: np.ndarray) -> float:
    """Average number of steps until violations clear."""
    idx = np.where(violation_mask)[0]
    if idx.size == 0:
        return 0.0
    steps: list[float] = []
    for i in idx:
        cleared = np.where(~violation_mask[i:])[0]
        if cleared.size > 0:
            steps.append(float(cleared[0]))
        else:
            steps.append(float(len(violation_mask) - i))
    return float(np.mean(steps)) if steps else 0.0


def _next_soc(
    *,
    soc_now: float,
    charge_mw: float,
    discharge_mw: float,
    dt_hours: float,
    charge_efficiency: float,
    discharge_efficiency: float,
) -> float:
    dt = max(float(dt_hours), 1e-9)
    eta_c = max(float(charge_efficiency), 1e-6)
    eta_d = max(float(discharge_efficiency), 1e-6)
    return float(soc_now + dt * (eta_c * max(charge_mw, 0.0) - (max(discharge_mw, 0.0) / eta_d)))


def summarize_true_soc_violations(true_soc: np.ndarray, min_soc: float, max_soc: float) -> dict[str, float]:
    true_soc = np.asarray(true_soc, dtype=float)
    below = true_soc < float(min_soc)
    above = true_soc > float(max_soc)
    violated = below | above

    severity = np.zeros_like(true_soc, dtype=float)
    severity[below] = float(min_soc) - true_soc[below]
    severity[above] = true_soc[above] - float(max_soc)

    if violated.any():
        severity_mean = float(np.mean(severity[violated]))
        severity_p95 = float(np.quantile(severity[violated], 0.95))
    else:
        severity_mean = 0.0
        severity_p95 = 0.0

    return {
        "true_soc_violation_rate": float(np.mean(violated)),
        "true_soc_violation_steps": float(np.sum(violated)),
        "true_soc_violation_severity_mean_mwh": severity_mean,
        "true_soc_violation_severity_p95_mwh": severity_p95,
        # Backward-compatible aliases consumed by existing scripts/tests.
        "true_soc_violation_severity_mean": severity_mean,
        "true_soc_violation_severity_p95": severity_p95,
        "true_soc_min_mwh": float(np.min(true_soc)) if len(true_soc) else 0.0,
        "true_soc_max_mwh": float(np.max(true_soc)) if len(true_soc) else 0.0,
    }


def compute_control_metrics(
    *,
    proposed_charge_mw: Iterable[float] | np.ndarray,
    proposed_discharge_mw: Iterable[float] | np.ndarray,
    safe_charge_mw: Iterable[float] | np.ndarray,
    safe_discharge_mw: Iterable[float] | np.ndarray,
    soc_mwh: Iterable[float] | np.ndarray,
    true_soc_mwh: Iterable[float] | np.ndarray | None = None,
    constraints: Mapping[str, Any],
    event_log: pd.DataFrame | None = None,
    bms_trip_mask: Iterable[float] | np.ndarray | None = None,
    load_true: Iterable[float] | np.ndarray | None = None,
    renewables_true: Iterable[float] | np.ndarray | None = None,
    intervention_eps: float = 1e-6,
) -> dict[str, Any]:
    """Compute safety/control metrics including violation and intervention rates."""
    p_ch = _as_array(proposed_charge_mw)
    p_dis = _as_array(proposed_discharge_mw)
    s_ch = _as_array(safe_charge_mw)
    s_dis = _as_array(safe_discharge_mw)
    soc = _as_array(soc_mwh)
    n = len(soc)
    if not (len(p_ch) == len(p_dis) == len(s_ch) == len(s_dis) == n):
        raise ValueError("Control arrays must have identical length")

    max_power = float(
        constraints.get("max_power_mw", max(np.max(s_ch, initial=0.0), np.max(s_dis, initial=0.0)))
    )
    max_charge = float(constraints.get("max_charge_mw", max_power))
    max_discharge = float(constraints.get("max_discharge_mw", max_power))
    min_soc = float(constraints.get("min_soc_mwh", np.min(soc, initial=0.0)))
    max_soc = float(constraints.get("max_soc_mwh", np.max(soc, initial=0.0)))
    dt = float(constraints.get("time_step_hours", 1.0))
    charge_eff = float(constraints.get("charge_efficiency", constraints.get("efficiency", 1.0)))
    discharge_eff = float(constraints.get("discharge_efficiency", constraints.get("efficiency", 1.0)))
    initial_soc = float(constraints.get("initial_soc_mwh", soc[0] if len(soc) else min_soc))
    max_grid_import = constraints.get("max_grid_import_mw")
    ramp_mw = float(constraints.get("ramp_mw", 0.0))

    net = s_dis - s_ch
    ramp_violation = np.zeros(n, dtype=bool)
    ramp_excess = np.zeros(n, dtype=float)
    if ramp_mw > 0 and n > 1:
        delta = np.abs(np.diff(net))
        excess = np.maximum(0.0, delta - ramp_mw)
        ramp_violation[1:] = excess > 1e-9
        ramp_excess[1:] = excess

    power_violation = (
        (s_ch > max_power + 1e-9)
        | (s_dis > max_power + 1e-9)
        | (s_ch > max_charge + 1e-9)
        | (s_dis > max_discharge + 1e-9)
        | ((s_ch > 1e-9) & (s_dis > 1e-9))
    )
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

    true_soc = soc if true_soc_mwh is None else _as_array(true_soc_mwh)
    true_soc_violation_low = np.maximum(0.0, min_soc - true_soc)
    true_soc_violation_high = np.maximum(0.0, true_soc - max_soc)
    true_soc_violation_severity = true_soc_violation_low + true_soc_violation_high
    true_soc_violation_mask = true_soc_violation_severity > 1e-9
    true_soc_summary = summarize_true_soc_violations(true_soc=true_soc, min_soc=min_soc, max_soc=max_soc)

    pre_soc = np.empty_like(true_soc)
    if len(pre_soc) > 0:
        pre_soc[0] = initial_soc
        if len(pre_soc) > 1:
            pre_soc[1:] = true_soc[:-1]
    proposed_next_soc = np.asarray(
        [
            _next_soc(
                soc_now=float(pre_soc[i]),
                charge_mw=float(p_ch[i]),
                discharge_mw=float(p_dis[i]),
                dt_hours=dt,
                charge_efficiency=charge_eff,
                discharge_efficiency=discharge_eff,
            )
            for i in range(n)
        ],
        dtype=float,
    )
    proposed_unsafe = (
        (p_ch > max_charge + 1e-9)
        | (p_dis > max_discharge + 1e-9)
        | ((p_ch > 1e-9) & (p_dis > 1e-9))
        | (proposed_next_soc < min_soc - 1e-9)
        | (proposed_next_soc > max_soc + 1e-9)
    )

    bms_trip = np.zeros(n, dtype=bool)
    if bms_trip_mask is not None:
        bms_trip = _as_array(bms_trip_mask).astype(float) > 0.0

    grid_import_violation_rate = 0.0
    if max_grid_import is not None and load_true is not None and renewables_true is not None:
        load_arr = _as_array(load_true)
        renew_arr = _as_array(renewables_true)
        horizon = min(len(load_arr), len(renew_arr), len(s_ch))
        grid_import = np.maximum(
            0.0,
            load_arr[:horizon] - renew_arr[:horizon] - s_dis[:horizon] + s_ch[:horizon],
        )
        grid_import_violation_rate = float(np.mean(grid_import > float(max_grid_import) + 1e-9))

    return {
        "violation_rate": float(np.mean(violations)),
        "violation_severity": float(np.sum(severity)),
        "recovery_time": float(recovery_time),
        "intervention_rate": float(np.mean(interventions)),
        **true_soc_summary,
        "recovery_time_mean": float(_mean_recovery_steps(true_soc_violation_mask)),
        "unsafe_command_rate": float(np.mean(proposed_unsafe)),
        "bms_trip_rate": float(np.mean(bms_trip)),
        "grid_import_violation_rate": float(grid_import_violation_rate),
        "true_soc_violation_mask": true_soc_violation_mask,
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
    true_soc_mwh: Iterable[float] | np.ndarray | None = None,
    constraints: Mapping[str, Any],
    certificates: Iterable[Mapping[str, Any] | None],
    event_log: pd.DataFrame | None = None,
    bms_trip_mask: Iterable[float] | np.ndarray | None = None,
    load_true: Iterable[float] | np.ndarray | None = None,
    renewables_true: Iterable[float] | np.ndarray | None = None,
) -> dict[str, Any]:
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
            true_soc_mwh=true_soc_mwh,
            constraints=constraints,
            event_log=event_log,
            bms_trip_mask=bms_trip_mask,
            load_true=load_true,
            renewables_true=renewables_true,
        )
    )
    metrics.update(compute_trace_metrics(certificates))
    return metrics
