"""Theory helpers for reliability-conditioned SOC tubes and one-step safety."""
from __future__ import annotations

from typing import Any, Mapping

from .guarantee_checks import next_soc


def _f(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def reliability_error_bound(
    *,
    reliability_w: float,
    max_error_mwh: float,
    min_error_mwh: float = 0.0,
    power: float = 1.0,
) -> float:
    """Monotone error envelope that grows as telemetry reliability deteriorates."""
    w = min(max(_f(reliability_w, 1.0), 0.0), 1.0)
    max_err = max(_f(max_error_mwh, 0.0), 0.0)
    min_err = min(max(_f(min_error_mwh, 0.0), 0.0), max_err)
    exponent = max(_f(power, 1.0), 0.0)
    return float(min_err + ((1.0 - w) ** exponent) * (max_err - min_err))


def tightened_soc_bounds(
    *,
    min_soc_mwh: float,
    max_soc_mwh: float,
    error_bound_mwh: float,
    q_rac_mwh: float | None = None,
) -> tuple[float, float]:
    """Return the observed-state tube that guarantees true-state feasibility.

    When *q_rac_mwh* is provided (the inflated conformal half-width from
    DC3S calibration), it is used as the tightening margin per the PDF
    formulation.  Otherwise falls back to *error_bound_mwh*.
    """
    margin = max(_f(q_rac_mwh, 0.0), 0.0) if q_rac_mwh is not None else max(_f(error_bound_mwh, 0.0), 0.0)
    lower = _f(min_soc_mwh, 0.0) + margin
    upper = _f(max_soc_mwh, lower) - margin
    if lower > upper:
        midpoint = 0.5 * (lower + upper)
        return float(midpoint), float(midpoint)
    return float(lower), float(upper)


def check_tightened_soc_invariance(
    *,
    current_soc_obs: float,
    action: Mapping[str, Any],
    constraints: Mapping[str, Any],
    reliability_w: float | None = None,
    error_bound_mwh: float | None = None,
    max_error_mwh: float | None = None,
    min_error_mwh: float = 0.0,
    power: float = 1.0,
    q_rac_mwh: float | None = None,
    eps: float = 1e-9,
) -> dict[str, float | bool]:
    """Check one-step observed feasibility inside a tightened reliability tube."""
    dt = _f(constraints.get("time_step_hours"), 1.0)
    eta_c = _f(constraints.get("charge_efficiency"), _f(constraints.get("efficiency"), 1.0))
    eta_d = _f(constraints.get("discharge_efficiency"), _f(constraints.get("efficiency"), 1.0))
    if error_bound_mwh is None:
        error_bound_mwh = reliability_error_bound(
            reliability_w=1.0 if reliability_w is None else reliability_w,
            max_error_mwh=_f(max_error_mwh, 0.0),
            min_error_mwh=min_error_mwh,
            power=power,
        )
    lower, upper = tightened_soc_bounds(
        min_soc_mwh=_f(constraints.get("min_soc_mwh"), 0.0),
        max_soc_mwh=_f(constraints.get("max_soc_mwh"), _f(constraints.get("capacity_mwh"), current_soc_obs)),
        error_bound_mwh=error_bound_mwh,
        q_rac_mwh=q_rac_mwh,
    )
    projected_obs = next_soc(
        current_soc=float(current_soc_obs),
        action=action,
        dt_hours=dt,
        charge_efficiency=eta_c,
        discharge_efficiency=eta_d,
    )
    observed_safe = lower - eps <= projected_obs <= upper + eps
    true_lower = projected_obs - float(error_bound_mwh)
    true_upper = projected_obs + float(error_bound_mwh)
    true_safe = (
        true_lower >= _f(constraints.get("min_soc_mwh"), 0.0) - eps
        and true_upper <= _f(constraints.get("max_soc_mwh"), _f(constraints.get("capacity_mwh"), current_soc_obs)) + eps
    )
    return {
        "error_bound_mwh": float(error_bound_mwh),
        "q_rac_mwh": float(q_rac_mwh) if q_rac_mwh is not None else None,
        "tightened_min_soc_mwh": float(lower),
        "tightened_max_soc_mwh": float(upper),
        "projected_soc_obs_mwh": float(projected_obs),
        "observed_safe": bool(observed_safe),
        "true_safe_if_bound_holds": bool(observed_safe and true_safe),
        "projected_true_lower_mwh": float(true_lower),
        "projected_true_upper_mwh": float(true_upper),
    }


def safety_filter_projection_summary(
    *,
    current_soc_obs: float,
    action: Mapping[str, Any],
    constraints: Mapping[str, Any],
    reliability_w: float,
    max_error_mwh: float,
    min_error_mwh: float = 0.0,
    power: float = 1.0,
) -> dict[str, float | bool]:
    """Expose the tightened-tube quantities behind the projection safety filter."""
    result = check_tightened_soc_invariance(
        current_soc_obs=current_soc_obs,
        action=action,
        constraints=constraints,
        reliability_w=reliability_w,
        max_error_mwh=max_error_mwh,
        min_error_mwh=min_error_mwh,
        power=power,
    )
    result["projection_interpretable_as_safety_filter"] = True
    return result
