"""Paper 2: Conformal reachability set propagation."""
from __future__ import annotations

from typing import Any, Dict, List


def propagate_reachability_set(
    interval_lower_mwh: float,
    interval_upper_mwh: float,
    safe_action: Dict[str, float],
    horizon_steps: int,
    sigma_d: float,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
) -> Dict[str, Any]:
    """Propagates the reachability set forward in time."""
    lower_tube = [interval_lower_mwh]
    upper_tube = [interval_upper_mwh]

    charge = safe_action.get("charge_mw", 0.0)
    discharge = safe_action.get("discharge_mw", 0.0)
    net_action_effect = (charge * charge_efficiency) - (discharge / discharge_efficiency)

    for k in range(1, horizon_steps + 1):
        center_k_minus_1 = (lower_tube[-1] + upper_tube[-1]) / 2.0
        center_k = center_k_minus_1 + net_action_effect

        radius_k_minus_1 = (upper_tube[-1] - lower_tube[-1]) / 2.0
        radius_k = radius_k_minus_1 + k * sigma_d

        lower_tube.append(center_k - radius_k)
        upper_tube.append(center_k + radius_k)

    return {
        "lower_mwh": lower_tube,
        "upper_mwh": upper_tube,
        "radius_mwh": (upper_tube[-1] - lower_tube[-1]) / 2.0,
    }


def compute_validity_horizon_from_reachability(
    interval_lower_mwh: float,
    interval_upper_mwh: float,
    safe_action: Dict[str, float],
    constraints: Dict[str, float],
    sigma_d: float,
    max_steps: int,
) -> Dict[str, Any]:
    """Computes validity horizon by propagating reachability set until it hits a constraint."""
    soc_min = constraints["min_soc_mwh"]
    soc_max = constraints["max_soc_mwh"]

    tube = propagate_reachability_set(
        interval_lower_mwh, interval_upper_mwh, safe_action, max_steps, sigma_d,
        constraints.get("charge_efficiency", 0.95),
        constraints.get("discharge_efficiency", 0.95),
    )

    for k, (lower, upper) in enumerate(zip(tube["lower_mwh"], tube["upper_mwh"])):
        if lower < soc_min or upper > soc_max:
            return {"tau_t": k}

    return {"tau_t": max_steps}


def compute_expiration_bound(
    interval_lower_mwh: float,
    interval_upper_mwh: float,
    soc_min_mwh: float,
    soc_max_mwh: float,
    sigma_d: float,
) -> Dict[str, Any]:
    """Computes a lower bound on the time to expiration."""
    center = (interval_lower_mwh + interval_upper_mwh) / 2.0
    radius = (interval_upper_mwh - interval_lower_mwh) / 2.0
    delta_bnd_mwh = min(center - soc_min_mwh, soc_max_mwh - center)

    tau_expire_lb = (delta_bnd_mwh - radius) / sigma_d if sigma_d > 1e-6 else float('inf')

    return {
        "tau_expire_lb": max(0, int(tau_expire_lb)),
        "delta_bnd_mwh": delta_bnd_mwh,
    }