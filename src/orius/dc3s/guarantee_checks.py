"""Deterministic guarantee checks for repaired DC3S actions."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _f(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def check_no_simultaneous_charge_discharge(action: Mapping[str, Any], eps: float = 1e-9) -> bool:
    """Battery cannot charge and discharge at the same time."""
    charge = max(0.0, _f(action.get("charge_mw"), 0.0))
    discharge = max(0.0, _f(action.get("discharge_mw"), 0.0))
    return not (charge > eps and discharge > eps)


def check_power_bounds(action: Mapping[str, Any], constraints: Mapping[str, Any], eps: float = 1e-9) -> bool:
    """Charge/discharge commands must stay inside configured inverter bounds."""
    charge = max(0.0, _f(action.get("charge_mw"), 0.0))
    discharge = max(0.0, _f(action.get("discharge_mw"), 0.0))
    max_power = _f(
        constraints.get("max_power_mw"),
        max(
            _f(constraints.get("max_charge_mw"), 0.0),
            _f(constraints.get("max_discharge_mw"), 0.0),
        ),
    )
    max_charge = _f(constraints.get("max_charge_mw"), max_power)
    max_discharge = _f(constraints.get("max_discharge_mw"), max_power)
    return charge <= max_charge + eps and discharge <= max_discharge + eps


def next_soc(
    *,
    current_soc: float,
    action: Mapping[str, Any],
    dt_hours: float,
    charge_efficiency: float,
    discharge_efficiency: float,
) -> float:
    """Apply one-step SOC dynamics without clipping."""
    charge = max(0.0, _f(action.get("charge_mw"), 0.0))
    discharge = max(0.0, _f(action.get("discharge_mw"), 0.0))
    # Efficiencies are physical: must be in (0, 1].  Values > 1.0 would
    # violate thermodynamics and produce nonsensical SOC projections.
    eta_c = float(np.clip(_f(charge_efficiency, 1.0), 1e-6, 1.0))
    eta_d = float(np.clip(_f(discharge_efficiency, 1.0), 1e-6, 1.0))
    dt = max(_f(dt_hours, 1.0), 1e-9)
    return float(current_soc + dt * (eta_c * charge - (discharge / eta_d)))


def check_soc_invariance(
    current_soc: float,
    action: Mapping[str, Any],
    constraints: Mapping[str, Any],
    dt_hours: float | None = None,
    charge_efficiency: float | None = None,
    discharge_efficiency: float | None = None,
    eps: float = 1e-9,
) -> bool:
    """One-step forward invariance check for SOC bounds."""
    dt = _f(dt_hours, _f(constraints.get("time_step_hours"), 1.0))
    eta_c = _f(
        charge_efficiency,
        _f(constraints.get("charge_efficiency"), _f(constraints.get("efficiency"), 1.0)),
    )
    eta_d = _f(
        discharge_efficiency,
        _f(constraints.get("discharge_efficiency"), _f(constraints.get("efficiency"), 1.0)),
    )
    min_soc = _f(constraints.get("min_soc_mwh"), 0.0)
    max_soc = _f(constraints.get("max_soc_mwh"), _f(constraints.get("capacity_mwh"), current_soc))
    projected = next_soc(
        current_soc=current_soc,
        action=action,
        dt_hours=dt,
        charge_efficiency=eta_c,
        discharge_efficiency=eta_d,
    )
    return (projected >= min_soc - eps) and (projected <= max_soc + eps)


def evaluate_guarantee_checks(
    *,
    current_soc: float,
    action: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> tuple[bool, list[str], float]:
    """Run all deterministic safety checks and return pass flag + reasons + next SOC."""
    reasons: list[str] = []
    if not check_no_simultaneous_charge_discharge(action):
        reasons.append("simultaneous_charge_discharge")
    if not check_power_bounds(action, constraints):
        reasons.append("power_bounds")
    if not check_soc_invariance(current_soc, action, constraints):
        reasons.append("soc_invariance")

    dt = _f(constraints.get("time_step_hours"), 1.0)
    eta_c = _f(constraints.get("charge_efficiency"), _f(constraints.get("efficiency"), 1.0))
    eta_d = _f(constraints.get("discharge_efficiency"), _f(constraints.get("efficiency"), 1.0))
    projected = next_soc(
        current_soc=float(current_soc),
        action=action,
        dt_hours=dt,
        charge_efficiency=eta_c,
        discharge_efficiency=eta_d,
    )
    return len(reasons) == 0, reasons, projected
