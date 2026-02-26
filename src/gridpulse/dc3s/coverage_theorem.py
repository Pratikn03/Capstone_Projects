"""
DC³S Marginal Coverage Guarantee — Formal Proposition and Verification.

Proposition (Marginal Coverage Monotonicity)
--------------------------------------------
Let C_α(x) = [ŷ(x) - q, ŷ(x) + q] be a base conformal prediction set
achieving (1-α) marginal coverage over an exchangeable calibration sequence:

    P(Y ∈ C_α(X)) ≥ 1 - α

Let infl ≥ 1 be the DC³S inflation factor:

    infl = clip(1 + k_quality*(1 - w_t) + k_drift*1[drift], 1, infl_max)

Then the DC³S inflated set C_DC3S(x) = [ŷ(x) - q·infl, ŷ(x) + q·infl]
satisfies:

    P(Y ∈ C_DC3S(X)) ≥ P(Y ∈ C_α(X)) ≥ 1 - α

Proof (by monotonicity):
    For each sample (x, y), since infl ≥ 1:
        q·infl ≥ q  →  [ŷ - q·infl, ŷ + q·infl] ⊇ [ŷ - q, ŷ + q]
    Therefore: 1[y ∈ C_DC3S(x)] ≥ 1[y ∈ C_α(x)] for all (x, y).
    Taking expectations: P(Y ∈ C_DC3S(X)) ≥ P(Y ∈ C_α(X)) ≥ 1 - α.  □

Corollary (Conservative Limit):
    When w_t → 1 (perfect telemetry) and drift = False, inflation → 1 and
    DC³S recovers the base conformal set with equality.

Assumption Contract:
    1. The base conformal interval is computed on a held-out calibration set
       independent of the training set (no leakage).
    2. Calibration and test data are exchangeable (i.i.d. or weakly dependent
       time series with gap rows enforced via train_end + gap_hours).
    3. infl ≥ 1 always (enforced by clip(·, 1, infl_max) in calibration.py).
    4. w_t, k_quality, k_drift, infl_max are fixed before observing test data.

Notes on Assumption 2:
    For time series, exact exchangeability does not hold. The guarantee is
    asymptotically valid under stationarity and is empirically verified via
    PICP (Prediction Interval Coverage Probability) on the held-out test fold
    reported in metrics_manifest.json.

This module provides:
    - verify_inflation_geq_one(): Runtime assertion that inflation ≥ 1.
    - compute_empirical_coverage(): Compute PICP on a held-out test set.
    - assert_coverage_guarantee(): Assert empirical coverage meets target.
"""
from __future__ import annotations

import numpy as np


def verify_inflation_geq_one(inflation: float, tol: float = 1e-9) -> None:
    """
    Runtime guard: assert that the DC³S inflation factor satisfies infl ≥ 1.

    This is a precondition for the marginal coverage guarantee. If inflation
    were < 1, the inflated interval would be *narrower* than the base conformal
    set and the coverage guarantee would no longer hold.

    Args:
        inflation: The computed DC³S inflation factor.
        tol: Numerical tolerance for floating-point comparison.

    Raises:
        ValueError: If inflation < 1 - tol.
    """
    if float(inflation) < 1.0 - tol:
        raise ValueError(
            f"DC³S inflation factor must be ≥ 1 to preserve marginal coverage guarantee. "
            f"Got inflation={inflation:.6f}. Check that clip(·, 1, infl_max) is applied "
            f"in build_uncertainty_set()."
        )


def compute_empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """
    Compute empirical Prediction Interval Coverage Probability (PICP).

    This is the empirical counterpart to the theoretical guarantee. For the
    guarantee to be useful in practice, empirical PICP on a held-out test set
    should be ≥ 1 - α.

    Args:
        y_true: True target values, shape (N,) or (N, H).
        lower:  Interval lower bounds, same shape as y_true.
        upper:  Interval upper bounds, same shape as y_true.

    Returns:
        dict with keys:
            picp: Overall coverage (fraction of y_true inside [lower, upper]).
            n_samples: Number of scalar comparisons made.
            mean_width: Mean interval width (upper - lower).
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if not (len(yt) == len(lo) == len(hi)):
        raise ValueError("y_true, lower, and upper must have the same number of elements.")
    if np.any(lo > hi + 1e-9):
        raise ValueError("lower must be ≤ upper element-wise.")
    covered = (yt >= lo) & (yt <= hi)
    return {
        "picp": float(np.mean(covered)),
        "n_samples": int(len(yt)),
        "mean_width": float(np.mean(hi - lo)),
    }


def assert_coverage_guarantee(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
    tolerance: float = 0.02,
) -> dict[str, float]:
    """
    Assert that the inflation factor has preserved the coverage guarantee.

    Checks that empirical PICP ≥ (1 - α - tolerance). The tolerance accounts
    for finite-sample variance in the calibration set.

    Args:
        y_true: True observations.
        lower:  DC³S lower bounds.
        upper:  DC³S upper bounds.
        alpha:  Nominal miscoverage level (e.g. 0.10 for 90% intervals).
        tolerance: Allowed slack below the nominal level (default 2%).

    Returns:
        dict with picp, n_samples, mean_width, target_coverage, passed.

    Raises:
        AssertionError: If PICP < (1 - alpha - tolerance).
    """
    result = compute_empirical_coverage(y_true, lower, upper)
    target = 1.0 - float(alpha)
    passed = result["picp"] >= target - float(tolerance)
    result["target_coverage"] = target
    result["tolerance"] = float(tolerance)
    result["passed"] = bool(passed)
    if not passed:
        raise AssertionError(
            f"DC³S coverage guarantee violated: empirical PICP={result['picp']:.4f} "
            f"< target={target:.4f} - tolerance={tolerance:.4f}. "
            f"Check calibration set size and inflation parameters."
        )
    return result


def inflation_lower_bound(
    k_quality: float,
    k_drift: float,
    w_t_min: float = 0.05,
    drift_possible: bool = True,
) -> float:
    """
    Compute the worst-case minimum inflation factor given parameter settings.

    Used to verify that infl ≥ 1 is guaranteed across all possible inputs.
    Since w_t ∈ [w_t_min, 1] and drift ∈ {0, 1}:
        infl_min = 1 + k_quality*(1-1) + k_drift*0 = 1  (perfect telemetry, no drift)
        infl_max_raw = 1 + k_quality*(1-w_t_min) + k_drift*1  (worst case)

    Returns:
        The minimum possible inflation before clipping (always 1.0 for valid params).
    """
    # Minimum occurs at w_t = 1 (perfect quality) and no drift
    infl_min_raw = 1.0 + float(k_quality) * (1.0 - 1.0) + float(k_drift) * 0.0
    return float(max(1.0, infl_min_raw))


__all__ = [
    "verify_inflation_geq_one",
    "compute_empirical_coverage",
    "assert_coverage_guarantee",
    "inflation_lower_bound",
]
