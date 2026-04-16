"""Online conformal recalibration for DC3S under distribution shift.

Gap closed
----------
The static calibration path computes a conformal quantile once offline and
inflates it by a fixed scalar when drift is detected.  Under distribution
shift the offline quantile becomes stale and coverage degrades.

This module implements adaptive conformal inference (Gibbs & Candès, NeurIPS
2021) extended with exponential forgetting so calibration samples near a
detected shift boundary are down-weighted relative to recent residuals.

Public API
----------
OnlineCalibrator
    Stateful sliding-window calibrator.  Call ``update(residual)`` each step
    and ``quantile(alpha)`` to retrieve the current coverage-guaranteed
    quantile.  Call ``notify_drift()`` after a Page-Hinkley detection to
    trigger immediate re-weighting.
calibration_contract_check
    One-shot check: verifies the calibration set is approximately
    exchangeable (no detectable trend) and returns a corrected quantile with
    a finite-sample Bonferroni adjustment.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import numpy as np

__all__ = [
    "OnlineCalibrator",
    "calibration_contract_check",
]

# Minimum calibration window before the quantile estimate is reliable.
_MIN_WINDOW: int = 20


class OnlineCalibrator:
    """Adaptive sliding-window conformal calibrator.

    At each step the calibrator maintains a fixed-size window of recent
    absolute residuals, weighted exponentially so that older residuals
    contribute less.  After a drift event the forgetting factor is
    temporarily sharpened, effectively discarding pre-drift calibration data.

    Parameters
    ----------
    window_size:
        Maximum number of residuals retained.  Older residuals are evicted
        when the window is full.
    forgetting_factor:
        Base exponential weight decay γ ∈ (0, 1].  Weight of sample i steps
        ago is γ^i.  Default 0.98 (slow forgetting, good for stationary).
    drift_forgetting_factor:
        Sharper forgetting factor used for ``post_drift_steps`` steps after
        ``notify_drift()`` is called.  Default 0.70 (fast forgetting).
    post_drift_steps:
        Number of steps after drift detection to use the sharp forgetting
        factor before reverting to ``forgetting_factor``.
    alpha:
        Default nominal miscoverage level when ``quantile()`` is called
        without an explicit alpha.
    """

    def __init__(
        self,
        *,
        window_size: int = 200,
        forgetting_factor: float = 0.98,
        drift_forgetting_factor: float = 0.70,
        post_drift_steps: int = 30,
        alpha: float = 0.10,
    ) -> None:
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError("forgetting_factor must be in (0, 1]")
        if not (0.0 < drift_forgetting_factor <= 1.0):
            raise ValueError("drift_forgetting_factor must be in (0, 1]")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._window: deque[float] = deque(maxlen=int(window_size))
        self._gamma_base = float(forgetting_factor)
        self._gamma_drift = float(drift_forgetting_factor)
        self._post_drift_steps = int(post_drift_steps)
        self._alpha_default = float(alpha)
        self._steps_since_drift: int | None = None  # None = no drift seen yet

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, residual: float) -> None:
        """Record a new absolute residual."""
        self._window.append(float(abs(residual)))
        if self._steps_since_drift is not None:
            self._steps_since_drift += 1
            if self._steps_since_drift >= self._post_drift_steps:
                self._steps_since_drift = None  # revert to slow forgetting

    def notify_drift(self) -> None:
        """Signal that a distribution shift has been detected.

        Triggers ``post_drift_steps`` of accelerated forgetting so that
        pre-drift calibration samples have minimal influence on the next
        quantile estimate.
        """
        self._steps_since_drift = 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def quantile(self, alpha: float | None = None) -> float:
        """Return the current weighted (1-alpha)-quantile of residuals.

        Uses the weighted empirical quantile with Bonferroni finite-sample
        correction (+1) as per Gibbs & Candès (2021).

        Returns ``float('inf')`` when fewer than ``_MIN_WINDOW`` samples
        are available, signalling that the calibration window is not yet
        reliable.
        """
        residuals = list(self._window)
        n = len(residuals)
        if n < _MIN_WINDOW:
            return float("inf")

        a = float(alpha if alpha is not None else self._alpha_default)
        gamma = self._active_forgetting_factor()

        # Exponential weights: most recent sample gets weight 1, oldest gets
        # weight gamma^(n-1).  Array is oldest-first from the deque.
        weights = np.array([gamma ** (n - 1 - i) for i in range(n)], dtype=float)
        weights /= weights.sum()

        # Sort residuals ascending, carry weights along.
        order = np.argsort(residuals)
        sorted_r = np.asarray(residuals, dtype=float)[order]
        sorted_w = weights[order]
        cumw = np.cumsum(sorted_w)

        # Weighted (1-alpha) quantile with +1/(n+1) Bonferroni correction.
        target = 1.0 - a + 1.0 / (n + 1)
        idx = int(np.searchsorted(cumw, min(target, 1.0)))
        idx = min(idx, n - 1)
        return float(sorted_r[idx])

    def coverage_rate(self, threshold: float | None = None, alpha: float | None = None) -> float:
        """Empirical coverage of residuals against the current quantile."""
        q = threshold if threshold is not None else self.quantile(alpha)
        if not math.isfinite(q):
            return float("nan")
        residuals = list(self._window)
        if not residuals:
            return float("nan")
        covered = sum(1 for r in residuals if r <= q)
        return covered / len(residuals)

    @property
    def n_samples(self) -> int:
        return len(self._window)

    @property
    def in_drift_recovery(self) -> bool:
        return self._steps_since_drift is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _active_forgetting_factor(self) -> float:
        if self._steps_since_drift is not None:
            return self._gamma_drift
        return self._gamma_base

    def state_dict(self) -> dict:
        """Serialisable snapshot for certificate persistence."""
        return {
            "residuals": list(self._window),
            "steps_since_drift": self._steps_since_drift,
            "gamma_base": self._gamma_base,
            "gamma_drift": self._gamma_drift,
            "post_drift_steps": self._post_drift_steps,
            "alpha_default": self._alpha_default,
            "n_samples": self.n_samples,
            "in_drift_recovery": self.in_drift_recovery,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "OnlineCalibrator":
        obj = cls(
            window_size=max(len(state.get("residuals", [])), _MIN_WINDOW),
            forgetting_factor=float(state.get("gamma_base", 0.98)),
            drift_forgetting_factor=float(state.get("gamma_drift", 0.70)),
            post_drift_steps=int(state.get("post_drift_steps", 30)),
            alpha=float(state.get("alpha_default", 0.10)),
        )
        for r in state.get("residuals", []):
            obj._window.append(float(r))
        obj._steps_since_drift = state.get("steps_since_drift")
        return obj


# ---------------------------------------------------------------------------
# One-shot calibration contract check
# ---------------------------------------------------------------------------

def calibration_contract_check(
    residuals: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.10,
    exchangeability_alpha: float = 0.05,
    tolerance: float = 0.02,
) -> dict:
    """Check the calibration contract and return a corrected quantile.

    Gap closed
    ----------
    The conformal prediction guarantee ``P(Y ∈ Ĉ(X)) ≥ 1-α`` requires the
    calibration set to be exchangeable with the test distribution.  This
    function checks approximate exchangeability via a Mann-Kendall trend test
    and returns a Bonferroni-corrected quantile together with diagnostic flags.

    Parameters
    ----------
    residuals:
        Absolute calibration residuals (|y - ŷ|), ordered by time.
    alpha:
        Nominal miscoverage level.
    exchangeability_alpha:
        Significance level for the trend test.  If the p-value is below this
        threshold the calibration set is flagged as non-exchangeable.
    tolerance:
        Allowed shortfall below 1-alpha in the empirical coverage check.

    Returns
    -------
    dict with keys:
        quantile_corrected  — Bonferroni-adjusted (1-alpha) quantile.
        quantile_raw        — Unadjusted empirical quantile.
        empirical_coverage  — Fraction of residuals ≤ quantile_corrected.
        exchangeable        — True if trend test passed.
        trend_p_value       — Mann-Kendall p-value (approximate).
        coverage_passed     — True if empirical_coverage ≥ 1-alpha-tolerance.
        n_samples           — Number of calibration residuals.
        assumptions_satisfied — True if both exchangeable and coverage_passed.
    """
    r = np.asarray(residuals, dtype=float).reshape(-1)
    n = int(r.size)

    if n < 2:
        return {
            "quantile_corrected": float("inf"),
            "quantile_raw": float("inf"),
            "empirical_coverage": float("nan"),
            "exchangeable": False,
            "trend_p_value": float("nan"),
            "coverage_passed": False,
            "n_samples": n,
            "assumptions_satisfied": False,
            "note": "Fewer than 2 calibration samples.",
        }

    # --- Bonferroni-corrected quantile (conformal +1 correction) ---
    level = min(1.0 - alpha + 1.0 / (n + 1), 1.0)
    q_raw = float(np.quantile(r, 1.0 - alpha))
    q_corrected = float(np.quantile(r, level))

    # --- Empirical coverage check ---
    emp_cov = float(np.mean(r <= q_corrected))
    coverage_passed = emp_cov >= (1.0 - alpha - tolerance)

    # --- Approximate Mann-Kendall trend test ---
    # Computes the S statistic and approximates the p-value via normal
    # approximation (valid for n ≥ 10).
    s_stat = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = r[j] - r[i]
            if diff > 0:
                s_stat += 1.0
            elif diff < 0:
                s_stat -= 1.0

    # Variance under H0 (no trend): Var(S) = n(n-1)(2n+5)/18
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s > 0 and n >= 10:
        z = (s_stat - (1.0 if s_stat > 0 else -1.0 if s_stat < 0 else 0.0)) / math.sqrt(var_s)
        # Two-sided p-value via normal approximation
        p_value = float(2.0 * (1.0 - _normal_cdf(abs(z))))
    else:
        p_value = 1.0  # vacuously pass for tiny samples

    exchangeable = p_value >= exchangeability_alpha

    return {
        "quantile_corrected": q_corrected,
        "quantile_raw": q_raw,
        "empirical_coverage": emp_cov,
        "exchangeable": exchangeable,
        "trend_p_value": p_value,
        "coverage_passed": coverage_passed,
        "n_samples": n,
        "alpha": float(alpha),
        "exchangeability_alpha": float(exchangeability_alpha),
        "tolerance": float(tolerance),
        "assumptions_satisfied": bool(exchangeable and coverage_passed),
        "note": (
            "Calibration contract satisfied."
            if (exchangeable and coverage_passed)
            else (
                "Non-exchangeable trend detected — consider online recalibration."
                if not exchangeable
                else "Coverage shortfall — inflate quantile or expand calibration set."
            )
        ),
    }


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))
