"""Battery-specific theorem helpers built on top of the universal kernel.

These utilities stay battery-specific by design; the universal theorem layer
should not encode SOC, MWh, or battery-only fallback semantics directly.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


def _f(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_action_delta_mwh(
    action: Mapping[str, Any],
    *,
    dt_hours: float,
    charge_efficiency: float,
    discharge_efficiency: float,
) -> float:
    charge = max(0.0, _f(action.get("charge_mw"), 0.0))
    discharge = max(0.0, _f(action.get("discharge_mw"), 0.0))
    eta_c = max(1e-9, float(charge_efficiency))
    eta_d = max(1e-9, float(discharge_efficiency))
    dt = max(1e-9, float(dt_hours))
    return dt * (eta_c * charge - (discharge / eta_d))


def forward_tube(
    *,
    interval_lower_mwh: float,
    interval_upper_mwh: float,
    safe_action: Mapping[str, Any],
    horizon_steps: int,
    sigma_d: float,
    dt_hours: float = 1.0,
    charge_efficiency: float = 1.0,
    discharge_efficiency: float = 1.0,
) -> dict[str, float]:
    """Construct the battery-domain forward tube for a fixed safe action."""
    if horizon_steps < 0:
        raise ValueError("horizon_steps must be non-negative.")
    if sigma_d < 0.0:
        raise ValueError("sigma_d must be non-negative.")

    net_delta = int(horizon_steps) * safe_action_delta_mwh(
        safe_action,
        dt_hours=dt_hours,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
    )
    radius = float(sigma_d) * math.sqrt(float(horizon_steps))
    return {
        "lower_mwh": float(interval_lower_mwh + net_delta - radius),
        "upper_mwh": float(interval_upper_mwh + net_delta + radius),
        "radius_mwh": float(radius),
        "net_action_delta_mwh": float(net_delta),
    }


def certificate_validity_horizon(
    *,
    interval_lower_mwh: float,
    interval_upper_mwh: float,
    safe_action: Mapping[str, Any],
    constraints: Mapping[str, Any],
    sigma_d: float,
    max_steps: int = 4096,
) -> dict[str, float | int]:
    """Compute the largest horizon whose battery tube stays inside bounds."""
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")

    dt_hours = _f(constraints.get("time_step_hours"), 1.0)
    charge_efficiency = _f(
        constraints.get("charge_efficiency"),
        _f(constraints.get("efficiency"), 1.0),
    )
    discharge_efficiency = _f(
        constraints.get("discharge_efficiency"),
        _f(constraints.get("efficiency"), 1.0),
    )
    soc_min = _f(constraints.get("min_soc_mwh"), 0.0)
    soc_max = _f(constraints.get("max_soc_mwh"), soc_min)

    last_valid = 0
    last_tube = forward_tube(
        interval_lower_mwh=interval_lower_mwh,
        interval_upper_mwh=interval_upper_mwh,
        safe_action=safe_action,
        horizon_steps=0,
        sigma_d=sigma_d,
        dt_hours=dt_hours,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
    )
    for step in range(max_steps + 1):
        tube = forward_tube(
            interval_lower_mwh=interval_lower_mwh,
            interval_upper_mwh=interval_upper_mwh,
            safe_action=safe_action,
            horizon_steps=step,
            sigma_d=sigma_d,
            dt_hours=dt_hours,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
        )
        if tube["lower_mwh"] < soc_min - 1e-9 or tube["upper_mwh"] > soc_max + 1e-9:
            break
        last_valid = step
        last_tube = tube

    return {
        "tau_t": int(last_valid),
        "soc_min_mwh": float(soc_min),
        "soc_max_mwh": float(soc_max),
        "tube_lower_mwh": float(last_tube["lower_mwh"]),
        "tube_upper_mwh": float(last_tube["upper_mwh"]),
    }


def certificate_expiration_bound(
    *,
    interval_lower_mwh: float,
    interval_upper_mwh: float,
    soc_min_mwh: float,
    soc_max_mwh: float,
    sigma_d: float,
) -> dict[str, float | int]:
    """Compute the battery-domain expiration lower bound."""
    if sigma_d <= 0.0:
        raise ValueError("sigma_d must be positive.")
    delta_bnd = min(
        float(interval_lower_mwh) - float(soc_min_mwh),
        float(soc_max_mwh) - float(interval_upper_mwh),
    )
    delta_bnd = max(0.0, float(delta_bnd))
    bound = math.floor((delta_bnd ** 2) / (float(sigma_d) ** 2))
    return {
        "delta_bnd_mwh": float(delta_bnd),
        "tau_expire_lb": int(bound),
    }


def zero_dispatch_fallback() -> dict[str, float]:
    """Return the canonical battery fallback action."""
    return {"charge_mw": 0.0, "discharge_mw": 0.0}


def validate_battery_fallback(
    *,
    current_soc_mwh: float,
    constraints: Mapping[str, Any],
    model_error_mwh: float = 0.0,
) -> dict[str, float | bool | dict[str, float]]:
    """Validate that zero dispatch preserves a safe interior SOC interval."""
    soc_min = _f(constraints.get("min_soc_mwh"), 0.0)
    soc_max = _f(constraints.get("max_soc_mwh"), soc_min)
    current_soc = float(current_soc_mwh)
    eps = max(0.0, float(model_error_mwh))
    next_lower = current_soc - eps
    next_upper = current_soc + eps
    passed = (
        current_soc >= soc_min - 1e-9
        and current_soc <= soc_max + 1e-9
        and next_lower >= soc_min - 1e-9
        and next_upper <= soc_max + 1e-9
    )
    return {
        "passed": bool(passed),
        "fallback_action": zero_dispatch_fallback(),
        "next_soc_lower_mwh": float(next_lower),
        "next_soc_upper_mwh": float(next_upper),
    }


def evaluate_graceful_degradation_dominance(
    graceful_violations: list[bool] | list[float],
    uncontrolled_violations: list[bool] | list[float],
) -> dict[str, float | bool]:
    """Check that graceful degradation does no worse than uncontrolled dispatch."""
    graceful_count = float(sum(1.0 for x in graceful_violations if bool(x)))
    uncontrolled_count = float(sum(1.0 for x in uncontrolled_violations if bool(x)))
    dominates = graceful_count <= uncontrolled_count + 1e-9
    strict = uncontrolled_count <= 0.0 or graceful_count < uncontrolled_count - 1e-9
    return {
        "graceful_violation_count": graceful_count,
        "uncontrolled_violation_count": uncontrolled_count,
        "dominates": bool(dominates),
        "strict_if_uncontrolled_violates": bool(strict),
    }


def certificate_half_life(
    *,
    tau_t: int,
    decay_rate: float = 0.5,
) -> dict[str, float | int]:
    """Compute certificate confidence half-life for battery certificates."""
    if tau_t <= 0:
        return {"half_life_steps": 0, "tau_t": int(tau_t)}
    if decay_rate <= 0.0 or decay_rate >= 1.0:
        return {"half_life_steps": int(tau_t), "tau_t": int(tau_t)}
    half_life = int(tau_t * math.log(2.0) / abs(math.log(decay_rate)))
    return {"half_life_steps": int(half_life), "tau_t": int(tau_t)}


def should_renew_certificate(
    *,
    tau_t: int,
    steps_since_renewal: int = 0,
    renewal_threshold_steps: int = 5,
) -> dict[str, bool | int | str]:
    """Return True when the battery certificate is approaching expiry."""
    remaining = max(0, tau_t - steps_since_renewal)
    return {
        "should_renew": bool(remaining <= renewal_threshold_steps and tau_t > 0),
        "remaining_certified_steps": int(remaining),
        "tau_t": int(tau_t),
        "renewal_trigger_reason": (
            "tau_t_remaining_le_threshold"
            if remaining <= renewal_threshold_steps and tau_t > 0
            else ""
        ),
    }


def should_expire_certificate(
    *,
    tau_t: int,
    steps_since_renewal: int = 0,
) -> dict[str, bool | int | str]:
    """Return True when the battery certificate has expired."""
    remaining = max(0, tau_t - steps_since_renewal)
    expiration_reason = "tau_t_exhausted" if remaining <= 0 else ""
    return {
        "should_expire": bool(remaining <= 0),
        "remaining_certified_steps": int(remaining),
        "tau_t": int(tau_t),
        "expiration_trigger_reason": expiration_reason,
        "expiration_reason": expiration_reason,
    }
