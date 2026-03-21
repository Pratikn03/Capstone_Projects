"""Certificate half-life and validity-horizon engine.

Implements the temporal safety certificate logic for DC3S (Paper 2):
- compute_validity_horizon: tau_t steps until conformal region likely violates constraint
- compute_half_life_from_horizon: converts tau_t to half-life using decay model
- compute_certificate_state: full certificate status including degraded/fallback/renewal tiers
- check_renewal_trigger: decides whether to renew an existing certificate
- time_to_expiration: remaining valid steps and hours for a certificate
"""
from __future__ import annotations

from typing import Any, Dict

MAX_HORIZON_STEPS: int = 4096
FALLBACK_QUALITY_THRESHOLD: float = 0.05


def compute_validity_horizon(
    observed_state: Dict[str, float],
    quality_score: float,
    safety_margin_mwh: float,
    constraints: Dict[str, float],
    sigma_d: float,
) -> Dict[str, Any]:
    """Compute the validity horizon tau_t.

    tau_t is the estimated number of steps before the conformal uncertainty
    region, inflated by 1/quality_score, is expected to reach a constraint
    boundary under a random-walk disturbance model with step std sigma_d.

    Returns dict with keys: tau_t (int), soc_min_mwh, soc_max_mwh.
    """
    soc_obs = observed_state["current_soc_mwh"]
    soc_min = constraints["min_soc_mwh"]
    soc_max = constraints["max_soc_mwh"]

    effective_margin = safety_margin_mwh / max(0.01, quality_score)
    soc_lower_bound = soc_obs - effective_margin
    soc_upper_bound = soc_obs + effective_margin

    dist_to_min = soc_lower_bound - soc_min
    dist_to_max = soc_max - soc_upper_bound

    if dist_to_min < 0 or dist_to_max < 0:
        return {"tau_t": 0, "soc_min_mwh": soc_lower_bound, "soc_max_mwh": soc_upper_bound}

    min_dist = min(dist_to_min, dist_to_max)

    if sigma_d > 1e-6:
        tau_t = min((min_dist / sigma_d) ** 2, MAX_HORIZON_STEPS)
    else:
        tau_t = MAX_HORIZON_STEPS

    return {"tau_t": int(tau_t), "soc_min_mwh": soc_lower_bound, "soc_max_mwh": soc_upper_bound}


def compute_half_life_from_horizon(tau_t: float, decay_rate: float) -> Dict[str, Any]:
    """Convert a validity horizon into a half-life.

    Uses the geometric decay model: half_life = tau_t * decay_rate / (1 - decay_rate).
    At decay_rate=0.5 this yields half_life = tau_t (intuitive: half-life equals horizon).

    Returns dict with keys: tau_t, half_life_steps.
    """
    if decay_rate <= 0.0 or decay_rate >= 1.0:
        half_life_steps = float(tau_t)
    else:
        half_life_steps = tau_t * decay_rate / (1.0 - decay_rate)
    return {"tau_t": tau_t, "half_life_steps": half_life_steps}


def compute_certificate_state(
    observed_soc_mwh: float | None = None,
    quality_score: float = 1.0,
    safety_margin_mwh: float = 100.0,
    sigma_d: float = 50.0,
    constraints: Dict[str, float] | None = None,
    current_step: int = 0,
    renewal_threshold: int = 5,
    fallback_threshold: int = 1,
    decay_rate: float = 0.5,
    # Legacy dict-based calling convention kept for backwards compatibility
    observed_state: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Compute the full certificate validity state.

    Accepts either:
      - observed_soc_mwh (float) directly, OR
      - observed_state {"current_soc_mwh": float} dict (legacy)

    Status tiers (mutually exclusive, checked in order):
      "fallback_required" — quality_score < FALLBACK_QUALITY_THRESHOLD (0.05)
      "expired"           — H_t == 0
      "degraded"          — H_t in (fallback_threshold, renewal_threshold]
      "valid"             — H_t > renewal_threshold

    Returns dict with keys:
      status, H_t, half_life, expires_at_step,
      renewal_ready, fallback_required, confidence.
    """
    if constraints is None:
        constraints = {
            "min_soc_mwh": 0.0,
            "max_soc_mwh": 10000.0,
        }

    # Resolve SOC from either calling convention
    if observed_soc_mwh is not None:
        _obs = {"current_soc_mwh": float(observed_soc_mwh)}
    elif observed_state is not None:
        _obs = dict(observed_state)
    else:
        _obs = {"current_soc_mwh": 0.0}

    # Force fallback when observation quality is below reliability floor
    if quality_score < FALLBACK_QUALITY_THRESHOLD:
        return {
            "status": "fallback_required",
            "H_t": 0,
            "half_life": 0.0,
            "expires_at_step": current_step,
            "renewal_ready": False,
            "fallback_required": True,
            "confidence": 0.0,
        }

    horizon = compute_validity_horizon(_obs, quality_score, safety_margin_mwh, constraints, sigma_d)
    H_t = horizon["tau_t"]

    hl = compute_half_life_from_horizon(H_t, decay_rate)
    half_life = hl["half_life_steps"]

    expires_at_step = current_step + H_t
    confidence = max(0.0, 1.0 - current_step / max(1, H_t))

    if H_t == 0:
        status = "expired"
    elif H_t <= fallback_threshold:
        status = "degraded"
    elif H_t <= renewal_threshold:
        status = "degraded"
    else:
        status = "valid"

    renewal_ready = H_t > renewal_threshold

    return {
        "status": status,
        "H_t": H_t,
        "half_life": half_life,
        "expires_at_step": expires_at_step,
        "renewal_ready": renewal_ready,
        "fallback_required": H_t <= fallback_threshold or status in ("expired", "fallback_required"),
        "confidence": float(confidence),
    }


def check_renewal_trigger(
    certificate_state: Dict[str, Any],
    new_quality_score: float,
    renewal_quality_threshold: float = 0.5,
) -> Dict[str, bool]:
    """Decide whether to renew an existing certificate.

    Renewal is triggered when:
    - The current certificate status is not "valid", AND
    - The new quality score is at least renewal_quality_threshold (0.5).

    Returns dict: {"should_renew": bool}.
    """
    status = certificate_state.get("status", "valid")
    should_renew = (status != "valid") and (new_quality_score >= renewal_quality_threshold)
    return {"should_renew": bool(should_renew)}


def time_to_expiration(
    H_t: int,
    elapsed_since_issue: int,
    dt_hours: float = 1.0,
) -> Dict[str, Any]:
    """Compute remaining validity from a certificate with horizon H_t.

    Args:
        H_t: Validity horizon in steps at issuance.
        elapsed_since_issue: Steps elapsed since the certificate was issued.
        dt_hours: Duration of each step in hours (default 1.0).

    Returns dict: remaining_steps (int), remaining_hours (float), expired (bool).
    """
    remaining_steps = max(0, H_t - elapsed_since_issue)
    expired = remaining_steps == 0
    return {
        "remaining_steps": remaining_steps,
        "remaining_hours": float(remaining_steps) * dt_hours,
        "expired": expired,
    }
