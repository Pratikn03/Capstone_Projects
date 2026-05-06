"""Derived Inflation Law: analytic k_q from concentration inequality.

Derives the linear inflation ``gamma(w_t) = 1 + k_q (1 - w_t)`` from
first principles, providing a rigorous foundation for the heuristic
used in the DC3S pipeline.

Complements ``calibration.py`` (runtime inflation) with the analytic form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DerivedInflationResult",
    "derived_inflation",
    "derived_k_q",
    "inflation_curve",
    "verify_heuristic_vs_derived",
]


@dataclass(frozen=True)
class DerivedInflationResult:
    """Result of a derived inflation law computation."""

    gamma: float
    k_q_derived: float
    sigma: float
    alpha: float
    q_hat: float
    w_t: float


def derived_k_q(sigma: float, alpha: float, q_hat: float) -> float:
    r"""Analytic k_q from the concentration-inequality derivation.

    .. math::
        k_q = \frac{\sigma \sqrt{2 \ln(2/\alpha)}}{\hat{q}}
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if q_hat <= 0:
        raise ValueError("q_hat must be positive")
    return float(sigma * np.sqrt(2.0 * np.log(2.0 / alpha)) / q_hat)


def derived_inflation(
    w_t: float,
    sigma: float,
    alpha: float,
    q_hat: float,
    infl_max: float = 5.0,
) -> DerivedInflationResult:
    r"""Compute the derived inflation factor gamma(w_t).

    .. math::
        \gamma(w_t) = \min(1 + k_q (1 - w_t),\; \gamma_{\max})
    """
    kq = derived_k_q(sigma, alpha, q_hat)
    raw = 1.0 + kq * (1.0 - w_t)
    gamma = min(raw, infl_max)
    return DerivedInflationResult(
        gamma=gamma,
        k_q_derived=kq,
        sigma=sigma,
        alpha=alpha,
        q_hat=q_hat,
        w_t=w_t,
    )


def verify_heuristic_vs_derived(
    k_q_heuristic: float,
    sigma: float,
    alpha: float,
    q_hat: float,
) -> dict[str, float]:
    """Compare the existing heuristic k_q to the derived value."""
    kq_deriv = derived_k_q(sigma, alpha, q_hat)
    rel_dev = abs(k_q_heuristic - kq_deriv) / kq_deriv if kq_deriv != 0 else 0.0
    return {
        "k_q_heuristic": k_q_heuristic,
        "k_q_derived": kq_deriv,
        "relative_deviation": rel_dev,
    }


def inflation_curve(
    w_values: np.ndarray,
    sigma: float,
    alpha: float,
    q_hat: float,
    infl_max: float = 5.0,
) -> np.ndarray:
    """Return gamma(w) for an array of reliability values (vectorized)."""
    kq = derived_k_q(sigma, alpha, q_hat)
    raw = 1.0 + kq * (1.0 - w_values)
    return np.clip(raw, 1.0, infl_max)
