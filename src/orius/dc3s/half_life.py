"""Paper 2: Certificate half-life and validity horizon logic."""
from __future__ import annotations

import math
from typing import Any, Dict


def compute_validity_horizon(
    observed_state: Dict[str, float],
    quality_score: float,
    safety_margin_mwh: float,
    constraints: Dict[str, float],
    sigma_d: float,
) -> Dict[str, Any]:
    """Computes the validity horizon tau_t."""
    soc_obs = observed_state["current_soc_mwh"]
    soc_min = constraints["min_soc_mwh"]
    soc_max = constraints["max_soc_mwh"]

    # Effective margin depends on quality score
    effective_margin = safety_margin_mwh / max(0.01, quality_score)

    # Widened interval for current SOC
    soc_lower_bound = soc_obs - effective_margin
    soc_upper_bound = soc_obs + effective_margin

    # Distance to nearest constraint boundary
    dist_to_min = soc_lower_bound - soc_min
    dist_to_max = soc_max - soc_upper_bound

    # If already outside, horizon is 0
    if dist_to_min < 0 or dist_to_max < 0:
        return {"tau_t": 0, "soc_min_mwh": soc_lower_bound, "soc_max_mwh": soc_upper_bound}

    min_dist = min(dist_to_min, dist_to_max)

    # Simplified model: how many steps until random walk of variance sigma_d^2 crosses the boundary
    if sigma_d > 1e-6:
        tau_t = (min_dist / sigma_d) ** 2
    else:
        tau_t = float('inf')

    return {"tau_t": int(tau_t), "soc_min_mwh": soc_lower_bound, "soc_max_mwh": soc_upper_bound}


def compute_half_life_from_horizon(tau_t: float, decay_rate: float) -> Dict[str, Any]:
    """Computes the half-life from the validity horizon."""
    # This is a simplified model where decay_rate is a discount factor.
    half_life_steps = tau_t * (1.0 - decay_rate)
    return {"tau_t": tau_t, "half_life_steps": half_life_steps}


def compute_certificate_state(
    observed_state: Dict[str, float],
    quality_score: float,
    safety_margin_mwh: float,
    constraints: Dict[str, float],
    sigma_d: float,
    current_step: int = 0,
    renewal_threshold: int = 5,
    fallback_threshold: int = 1,
) -> Dict[str, Any]:
    """Computes the full certificate state, including validity horizon and status."""
    horizon_result = compute_validity_horizon(
        observed_state, quality_score, safety_margin_mwh, constraints, sigma_d
    )
    H_t = horizon_result["tau_t"]
    half_life = H_t / 2.0
    expires_at_step = current_step + H_t

    status = "valid"
    if H_t <= 0:
        status = "expired"
    elif H_t <= fallback_threshold:
        status = "expiring"

    return {
        "status": status,
        "H_t": H_t,
        "half_life": half_life,
        "expires_at_step": expires_at_step,
        "renewal_ready": H_t <= renewal_threshold,
        "fallback_required": H_t <= fallback_threshold or status == "expired",
    }