"""Brownian certificate half-life: generic domain-agnostic model.

Provides the closed-form half-life, validity probability, and
empirical measurement for any domain with a scalar safe-set margin.

This complements the battery-specific ``half_life.py`` which implements
the operational certificate-state engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = [
    "CertificateHalfLifeResult",
    "certificate_half_life",
    "empirical_half_life",
    "validity_probability",
]

PHI_INV_075: float = float(norm.ppf(0.75))


@dataclass(frozen=True)
class CertificateHalfLifeResult:
    """Closed-form validity summary for a runtime certificate."""

    half_life_steps: float
    half_life_seconds: float
    initial_margin: float
    disturbance_std: float
    domain_name: str


def certificate_half_life(
    initial_margin: float,
    disturbance_std: float,
    step_duration_sec: float = 1.0,
    domain_name: str = "unnamed",
) -> CertificateHalfLifeResult:
    r"""Closed-form certificate half-life under Brownian disturbance.

    .. math::
        \tau_{1/2} = \left(\frac{d_0}{\Phi^{-1}(0.75)\,\sigma_d}\right)^2
    """
    margin = max(float(initial_margin), 0.0)
    sigma_d = max(float(disturbance_std), 1e-12)
    half_life_steps = (margin / (PHI_INV_075 * sigma_d)) ** 2
    return CertificateHalfLifeResult(
        half_life_steps=float(half_life_steps),
        half_life_seconds=float(half_life_steps * float(step_duration_sec)),
        initial_margin=margin,
        disturbance_std=float(disturbance_std),
        domain_name=domain_name,
    )


def validity_probability(t: float, d_0: float, sigma_d: float) -> float:
    """Return the survival probability of a certificate at time *t*."""
    if t <= 0:
        return 1.0
    if d_0 <= 0 or sigma_d <= 0:
        return 0.0
    z_score = float(d_0) / (float(sigma_d) * np.sqrt(float(t)))
    return float(2.0 * norm.cdf(z_score) - 1.0)


def empirical_half_life(violation_times: np.ndarray) -> float:
    """Measure empirical half-life from logged certificate lifetimes.

    Parameters
    ----------
    violation_times : ndarray
        Per-certificate time-to-violation (``inf`` if never violated).

    Returns
    -------
    float
        Median time-to-violation (empirical τ_{1/2}).
    """
    times = np.asarray(violation_times, dtype=float).reshape(-1)
    finite = times[np.isfinite(times) & (times >= 0.0)]
    if len(finite) == 0:
        return float("inf")
    return float(np.median(finite))
