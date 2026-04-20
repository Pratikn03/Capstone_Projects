"""Reliability-Weighted Conformal Prediction (RWCP).

Extends standard split-conformal prediction by reweighting calibration
residuals with inverse reliability scores, so degraded-sensor regimes
produce wider prediction sets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RWCPResult",
    "calibrate_rwcp",
    "predict_rwcp",
    "weighted_quantile",
]


@dataclass(frozen=True)
class RWCPResult:
    """Output of a reliability-weighted conformal calibration."""

    quantile_threshold: float
    effective_sample_size: float
    n_calibration: int
    alpha: float
    domain_name: str


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """Compute the *q*-th weighted quantile.

    Sorts by value, cumulates normalized weights, and picks the first
    value whose cumulative weight >= *q*.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not (0 < q <= 1):
        raise ValueError("q must be in (0, 1]")

    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    cum_w /= cum_w[-1]
    idx = int(np.searchsorted(cum_w, q))
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def calibrate_rwcp(
    nonconformity_scores: np.ndarray,
    reliability_scores: np.ndarray,
    alpha: float = 0.1,
    domain_name: str = "unnamed",
) -> RWCPResult:
    r"""Calibrate a reliability-weighted conformal prediction set.

    Parameters
    ----------
    nonconformity_scores : ndarray, shape (n,)
        ``s_i = |y_i - \hat{y}_i|`` on calibration data.
    reliability_scores : ndarray, shape (n,)
        ``w_i \in (0, 1]``.
    alpha : float
        Desired miscoverage rate.
    domain_name : str
        Label.
    """
    n = len(nonconformity_scores)
    if n != len(reliability_scores):
        raise ValueError("scores and reliability must have the same length")
    if np.any(reliability_scores <= 0):
        raise ValueError("reliability_scores must be > 0")

    inv_weights = 1.0 / reliability_scores
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    threshold = weighted_quantile(nonconformity_scores, inv_weights, q_level)

    norm_w = inv_weights / inv_weights.sum()
    ess = 1.0 / np.sum(norm_w**2)

    return RWCPResult(
        quantile_threshold=threshold,
        effective_sample_size=float(ess),
        n_calibration=n,
        alpha=alpha,
        domain_name=domain_name,
    )


def predict_rwcp(
    point_prediction: np.ndarray | float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RWCP prediction interval ``[ŷ - threshold, ŷ + threshold]``."""
    pp = np.asarray(point_prediction)
    return pp - threshold, pp + threshold
