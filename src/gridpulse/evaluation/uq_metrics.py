"""Standardized UQ metric contract for R1 publication.

All uncertainty evaluation must produce exactly this metric set.
Definitions follow Gneiting & Raftery (2007) and Romano et al. (2019).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UQMetricContract:
    """Required UQ metrics for every evaluation report."""

    picp_90: float
    picp_95: float
    mean_interval_width_90: float
    mean_interval_width_95: float
    pinball_loss_05: float
    pinball_loss_95: float
    winkler_score_90: float
    winkler_score_95: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ── Metric definitions ────────────────────────────────────────────────────────


def compute_picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability.

    PICP = (1/n) * sum( 1[lower_i <= y_i <= upper_i] )
    A 90%-nominal interval should yield PICP ≈ 0.90 under correct calibration.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower, upper must have the same shape")
    covered = (y >= lo) & (y <= hi)
    return float(np.mean(covered))


def compute_pinball_loss(
    y_true: np.ndarray,
    y_pred_quantile: np.ndarray,
    tau: float,
) -> float:
    """Pinball (quantile) loss at quantile level *tau*.

    L_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)
    """
    y = np.asarray(y_true, dtype=float).ravel()
    q = np.asarray(y_pred_quantile, dtype=float).ravel()
    residual = y - q
    return float(np.mean(np.where(residual >= 0, tau * residual, (tau - 1.0) * residual)))


def compute_winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Winkler interval score at miscoverage level *alpha*.

    S_alpha = width + (2/alpha)*penalty
    Lower is better.  Penalizes both width and miscoverage.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    width = hi - lo
    penalty_low = (2.0 / alpha) * np.maximum(lo - y, 0.0)
    penalty_high = (2.0 / alpha) * np.maximum(y - hi, 0.0)
    return float(np.mean(width + penalty_low + penalty_high))


def compute_mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    return float(np.mean(hi - lo))


# ── Full contract computation ─────────────────────────────────────────────────


def compute_uq_contract(
    y_true: np.ndarray,
    lower_90: np.ndarray,
    upper_90: np.ndarray,
    lower_95: np.ndarray | None = None,
    upper_95: np.ndarray | None = None,
) -> UQMetricContract:
    """Compute the full UQ metric contract from prediction arrays.

    Parameters
    ----------
    y_true : array  – ground-truth values
    lower_90, upper_90 : arrays  – 90 % prediction interval bounds
    lower_95, upper_95 : arrays  – 95 % prediction interval bounds (optional;
        approximated via normal-quantile rescaling when absent)
    """
    y = np.asarray(y_true, dtype=float).ravel()
    lo90 = np.asarray(lower_90, dtype=float).ravel()
    hi90 = np.asarray(upper_90, dtype=float).ravel()

    if lower_95 is None or upper_95 is None:
        mid = 0.5 * (lo90 + hi90)
        half_90 = 0.5 * (hi90 - lo90)
        # z_{0.975} / z_{0.95} ≈ 1.96 / 1.645
        scale = 1.96 / 1.645
        lo95 = mid - half_90 * scale
        hi95 = mid + half_90 * scale
    else:
        lo95 = np.asarray(lower_95, dtype=float).ravel()
        hi95 = np.asarray(upper_95, dtype=float).ravel()

    return UQMetricContract(
        picp_90=compute_picp(y, lo90, hi90),
        picp_95=compute_picp(y, lo95, hi95),
        mean_interval_width_90=compute_mean_interval_width(lo90, hi90),
        mean_interval_width_95=compute_mean_interval_width(lo95, hi95),
        pinball_loss_05=compute_pinball_loss(y, lo90, tau=0.05),
        pinball_loss_95=compute_pinball_loss(y, hi90, tau=0.95),
        winkler_score_90=compute_winkler_score(y, lo90, hi90, alpha=0.10),
        winkler_score_95=compute_winkler_score(y, lo95, hi95, alpha=0.05),
        n_samples=int(len(y)),
    )


# ── LaTeX definitions for manuscript ──────────────────────────────────────────

METRIC_DEFINITIONS_LATEX = r"""
% ── UQ Metric Definitions (paste into paper preamble or methods section) ──
%
% PICP: Prediction Interval Coverage Probability
%   PICP@\alpha = \frac{1}{n}\sum_{i=1}^{n}
%                 \mathbf{1}[\hat{l}_i \leq y_i \leq \hat{u}_i]
%   where [\hat{l}_i, \hat{u}_i] is the (1-\alpha) prediction interval.
%
% Pinball (Quantile) Loss:
%   L_\tau(y, \hat{q}) = \tau\,\max(y - \hat{q}, 0)
%                      + (1-\tau)\,\max(\hat{q} - y, 0)
%
% Winkler Interval Score:
%   S_\alpha = \bar{w} + \frac{2}{\alpha}\,\overline{p}
%   where \bar{w} = mean interval width, \overline{p} = mean penalty
%   for miscoverage.  Lower is better.
"""
