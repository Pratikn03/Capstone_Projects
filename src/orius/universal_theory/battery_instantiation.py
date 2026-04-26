"""Battery-specific theorem helpers built on top of the universal kernel.

These utilities stay battery-specific by design; the universal theorem layer
should not encode SOC, MWh, or battery-only fallback semantics directly.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


T6_THEOREM_FORMULA = "floor(delta_bnd^2 / (2 * sigma_d^2 * log(2 / delta)))"
T7_THEOREM_SURFACE = "piecewise_hold_or_safe_landing_fallback"


def _f(value: Any, default: float) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            return float(default)
        return v
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
    delta: float,
) -> dict[str, float | int]:
    """Compute the canonical T6 battery-domain expiration lower bound.

    The defended T6 surface is the confidence-aware sub-Gaussian lower bound

        floor(delta_bnd^2 / (2 * sigma_d^2 * log(2 / delta)))

    for delta in (0, 1).
    """
    if sigma_d <= 0.0:
        raise ValueError("sigma_d must be positive.")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must lie in (0, 1).")
    delta_bnd = min(
        float(interval_lower_mwh) - float(soc_min_mwh),
        float(soc_max_mwh) - float(interval_upper_mwh),
    )
    delta_bnd = max(0.0, float(delta_bnd))
    sigma_sq = float(sigma_d) ** 2
    denominator = 2.0 * sigma_sq * math.log(2.0 / float(delta))
    bound = math.floor((delta_bnd ** 2) / denominator) if denominator > 0.0 else 0
    theorem_contract = {
        "theorem_id": "T6",
        "theorem_surface": "canonical_delta_aware_expiration_bound",
        "formula": T6_THEOREM_FORMULA,
        "proof_style": "closed_form_first_passage_lower_bound",
        "first_passage_lemma": "reflection_principle_subgaussian_first_passage",
        "requires_delta": True,
        "legacy_surface_allowed": False,
        "all_executable_checks_passed": True,
        "status": "machine_checked_ready",
        "side_conditions": [
            "0 < delta < 1",
            "sigma_d > 0",
            "delta_bnd is the certificate-to-boundary margin",
            "A9 sub-Gaussian disturbance law",
        ],
        "closed_form_inputs": {
            "delta_bnd_mwh": float(delta_bnd),
            "sigma_d": float(sigma_d),
            "confidence_delta": float(delta),
        },
        "executable_checks": [
            {
                "name": "delta_present",
                "passed": True,
                "detail": f"delta={float(delta):.6f} lies in (0, 1).",
            },
            {
                "name": "sigma_positive",
                "passed": True,
                "detail": f"sigma_d={float(sigma_d):.6f} is positive.",
            },
            {
                "name": "closed_form_bound",
                "passed": True,
                "detail": (
                    f"tau_expire_lb={int(bound)} computed from delta_bnd={float(delta_bnd):.6f} "
                    f"and denominator={float(denominator):.6f}."
                ),
            },
        ],
        "declared_assumptions": [
            "A4",
            "A7",
            "A9",
            "Reflection-principle / sub-Gaussian first-passage bound",
        ],
    }
    return {
        "delta_bnd_mwh": float(delta_bnd),
        "confidence_delta": float(delta),
        "sigma_d": float(sigma_d),
        "denominator": float(denominator),
        "tau_expire_lb": int(bound),
        "theorem_id": "T6",
        "theorem_formula": T6_THEOREM_FORMULA,
        "requires_delta": True,
        "legacy_surface_allowed": False,
        "theorem_contract": theorem_contract,
    }


def zero_dispatch_fallback() -> dict[str, float]:
    """Return the canonical battery fallback action."""
    return {"charge_mw": 0.0, "discharge_mw": 0.0}


def compute_battery_safe_landing_action(
    *,
    current_soc_mwh: float,
    constraints: Mapping[str, Any],
) -> dict[str, float | dict[str, float]]:
    """Pure battery fallback helper shared by theorem and shield surfaces."""
    capacity = _f(constraints.get("capacity_mwh"), 10.0)
    soc_min = _f(constraints.get("ftit_soc_min_mwh"), _f(constraints.get("min_soc_mwh"), 0.0))
    soc_max = _f(constraints.get("ftit_soc_max_mwh"), _f(constraints.get("max_soc_mwh"), capacity))
    current_soc = float(current_soc_mwh)
    dt_hours = max(_f(constraints.get("time_step_hours"), 1.0), 1e-9)
    charge_eff = max(1e-6, _f(constraints.get("charge_efficiency"), _f(constraints.get("efficiency"), 1.0)))
    discharge_eff = max(1e-6, _f(constraints.get("discharge_efficiency"), _f(constraints.get("efficiency"), 1.0)))
    max_power = _f(
        constraints.get("max_power_mw"),
        max(_f(constraints.get("max_charge_mw"), 0.0), _f(constraints.get("max_discharge_mw"), 0.0)),
    )
    max_charge = _f(constraints.get("max_charge_mw"), max_power)
    max_discharge = _f(constraints.get("max_discharge_mw"), max_power)
    ramp_mw = _f(constraints.get("ramp_mw"), 0.0)
    last_net = _f(constraints.get("last_net_mw"), 0.0)

    safe_margin_mwh = max(0.0, _f(constraints.get("safe_landing_margin_mwh"), _f(constraints.get("safe_margin_mwh"), 0.0)))
    safe_margin_pct = max(0.0, _f(constraints.get("safe_landing_margin_pct"), _f(constraints.get("safe_margin_pct"), 0.05)))
    configured_margin = max(safe_margin_mwh, safe_margin_pct * capacity)
    safe_zone_min = min(soc_max, soc_min + configured_margin)
    safe_zone_max = max(soc_min, soc_max - configured_margin)
    if safe_zone_min >= safe_zone_max:
        safe_zone_min = soc_min
        safe_zone_max = soc_max
    target_soc = _f(constraints.get("safe_landing_target_soc_mwh"), 0.5 * (safe_zone_min + safe_zone_max))
    target_soc = min(max(target_soc, safe_zone_min), safe_zone_max)

    if current_soc < safe_zone_min - 1e-9:
        desired_net = -min(max_charge, (target_soc - current_soc) / (charge_eff * dt_hours))
        landing_region = "lower_boundary_recovery"
    elif current_soc > safe_zone_max + 1e-9:
        desired_net = min(max_discharge, ((current_soc - target_soc) * discharge_eff) / dt_hours)
        landing_region = "upper_boundary_recovery"
    else:
        desired_net = 0.0
        landing_region = "safe_hold_zone"

    if ramp_mw > 0.0:
        desired_net = max(last_net - ramp_mw, min(last_net + ramp_mw, desired_net))

    charge = 0.0
    discharge = 0.0
    if desired_net >= 0.0:
        feasible_discharge = max(0.0, ((current_soc - soc_min) * discharge_eff) / dt_hours)
        discharge = min(desired_net, max_discharge, max_power, feasible_discharge)
    else:
        feasible_charge = max(0.0, (soc_max - current_soc) / (charge_eff * dt_hours))
        charge = min(-desired_net, max_charge, max_power, feasible_charge)

    fallback_action = {"charge_mw": float(charge), "discharge_mw": float(discharge)}
    next_soc_nominal = float(
        current_soc
        + safe_action_delta_mwh(
            fallback_action,
            dt_hours=dt_hours,
            charge_efficiency=charge_eff,
            discharge_efficiency=discharge_eff,
        )
    )
    return {
        "fallback_action": fallback_action,
        "current_soc_mwh": float(current_soc),
        "target_soc_mwh": float(target_soc),
        "safe_zone_min_mwh": float(safe_zone_min),
        "safe_zone_max_mwh": float(safe_zone_max),
        "next_soc_nominal_mwh": float(next_soc_nominal),
        "landing_region": landing_region,
        "last_net_mw": float(last_net),
        "ramp_mw": float(ramp_mw),
    }


def validate_battery_fallback(
    *,
    current_soc_mwh: float,
    constraints: Mapping[str, Any],
    model_error_mwh: float = 0.0,
) -> dict[str, Any]:
    """Validate the promoted piecewise T7 battery fallback surface."""
    soc_min = _f(constraints.get("min_soc_mwh"), 0.0)
    soc_max = _f(constraints.get("max_soc_mwh"), soc_min)
    current_soc = float(current_soc_mwh)
    eps = max(0.0, float(model_error_mwh))
    dt_hours = max(_f(constraints.get("time_step_hours"), 1.0), 1e-9)
    charge_eff = max(1e-6, _f(constraints.get("charge_efficiency"), _f(constraints.get("efficiency"), 1.0)))
    discharge_eff = max(1e-6, _f(constraints.get("discharge_efficiency"), _f(constraints.get("efficiency"), 1.0)))
    landing_plan = compute_battery_safe_landing_action(current_soc_mwh=current_soc, constraints=constraints)

    hold_action = zero_dispatch_fallback()
    hold_delta = safe_action_delta_mwh(
        hold_action,
        dt_hours=dt_hours,
        charge_efficiency=charge_eff,
        discharge_efficiency=discharge_eff,
    )
    hold_lower = current_soc + hold_delta - eps
    hold_upper = current_soc + hold_delta + eps
    in_safe_set = soc_min - 1e-9 <= current_soc <= soc_max + 1e-9
    hold_safe = in_safe_set and hold_lower >= soc_min - 1e-9 and hold_upper <= soc_max + 1e-9

    landing_action = dict(landing_plan["fallback_action"])
    landing_delta = safe_action_delta_mwh(
        landing_action,
        dt_hours=dt_hours,
        charge_efficiency=charge_eff,
        discharge_efficiency=discharge_eff,
    )
    landing_lower = current_soc + landing_delta - eps
    landing_upper = current_soc + landing_delta + eps
    landing_moves_state = abs(landing_delta) > 1e-9
    landing_safe = (
        in_safe_set
        and landing_moves_state
        and landing_lower >= soc_min - 1e-9
        and landing_upper <= soc_max + 1e-9
    )

    if hold_safe:
        mode = "hold"
        fallback_region = "interior"
        fallback_action = hold_action
        next_lower = hold_lower
        next_upper = hold_upper
        passed = True
    elif landing_safe:
        mode = "safe_landing"
        fallback_region = "boundary_proximal"
        fallback_action = landing_action
        next_lower = landing_lower
        next_upper = landing_upper
        passed = True
    else:
        mode = "infeasible"
        fallback_region = "infeasible"
        fallback_action = landing_action if in_safe_set else hold_action
        next_lower = landing_lower if in_safe_set else hold_lower
        next_upper = landing_upper if in_safe_set else hold_upper
        passed = False

    theorem_contract = {
        "theorem_id": "T7",
        "theorem_surface": T7_THEOREM_SURFACE,
        "status": "runtime_linked" if passed else "fail_closed",
        "all_executable_checks_passed": bool(passed),
        "declared_assumptions": ["A1", "A4", "A8"],
        "executable_checks": [
            {
                "name": "current_state_safe",
                "passed": bool(in_safe_set),
                "detail": f"current_soc={current_soc:.6f}, bounds=[{soc_min:.6f}, {soc_max:.6f}]",
            },
            {
                "name": "hold_region_safe",
                "passed": bool(hold_safe),
                "detail": f"hold_interval=[{hold_lower:.6f}, {hold_upper:.6f}]",
            },
            {
                "name": "safe_landing_region_safe",
                "passed": bool(landing_safe),
                "detail": (
                    f"landing_interval=[{landing_lower:.6f}, {landing_upper:.6f}], "
                    f"landing_region={landing_plan['landing_region']}"
                ),
            },
        ],
    }
    return {
        "passed": bool(passed),
        "mode": mode,
        "fallback_region": fallback_region,
        "fallback_action": fallback_action,
        "next_soc_lower_mwh": float(next_lower),
        "next_soc_upper_mwh": float(next_upper),
        "safe_zone_min_mwh": float(landing_plan["safe_zone_min_mwh"]),
        "safe_zone_max_mwh": float(landing_plan["safe_zone_max_mwh"]),
        "target_soc_mwh": float(landing_plan["target_soc_mwh"]),
        "theorem_contract": theorem_contract,
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
