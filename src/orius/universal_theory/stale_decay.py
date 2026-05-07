"""Stale-observation uncertainty growth helpers (Tstale)."""
from __future__ import annotations


def stale_uncertainty_growth(r_t: float, drift_bound_l: float, stale_steps: int) -> float:
    if stale_steps < 0:
        raise ValueError("stale_steps must be non-negative")
    return float(r_t) + float(drift_bound_l) * stale_steps


def stale_hold_radius(r_t: float, drift_bound_l: float, stale_steps: int) -> float:
    return stale_uncertainty_growth(r_t, drift_bound_l, stale_steps)


def stale_certificate_expiry(horizon: int, stale_steps: int) -> int:
    return max(0, int(horizon) - int(stale_steps))
