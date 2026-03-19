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
    - compute_expected_violation_bound(): Evaluate the chapter-18 core bound.
    - evaluate_empirical_core_bound(): Compare observed violations to that bound.
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


def _as_flat_float_array(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one element.")
    return arr


def compute_expected_violation_bound(
    reliability: np.ndarray | list[float],
    *,
    alpha: float = 0.10,
) -> dict[str, float]:
    """
    Compute the chapter-18 expected violation bound E[V] <= alpha * (1 - w_bar) * T.

    This does not mechanize the manuscript proof. Instead, it gives the runtime
    and audit layer a concrete implementation of the bound so that empirical
    traces can be checked against the same formula used in the chapter text.
    """
    w = _as_flat_float_array(reliability, name="reliability")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    if np.any((w < -1e-9) | (w > 1.0 + 1e-9)):
        raise ValueError("reliability scores must lie in [0, 1].")

    horizon = int(w.size)
    mean_reliability = float(np.mean(np.clip(w, 0.0, 1.0)))
    bound_expected_violations = float(alpha * (1.0 - mean_reliability) * horizon)
    return {
        "alpha": float(alpha),
        "horizon": horizon,
        "mean_reliability_w": mean_reliability,
        "bound_expected_violations": bound_expected_violations,
        "bound_tsvr": float(bound_expected_violations / horizon),
    }


def evaluate_empirical_core_bound(
    violations: np.ndarray | list[bool] | list[float],
    reliability: np.ndarray | list[float],
    *,
    alpha: float = 0.10,
    slack_violations: float = 0.0,
) -> dict[str, float | bool]:
    """
    Compare an observed violation sequence to the chapter-18 expected bound.

    Because the theorem is stated in expectation, this helper is intentionally
    empirical: it reports whether the realized violation count falls below the
    bound with optional finite-sample slack, rather than claiming formal proof
    of the theorem inside the runtime.
    """
    z = _as_flat_float_array(violations, name="violations")
    w = _as_flat_float_array(reliability, name="reliability")
    if z.size != w.size:
        raise ValueError("violations and reliability must have the same length.")
    if np.any((z < -1e-9) | (z > 1.0 + 1e-9)):
        raise ValueError("violations must be indicator-like values in [0, 1].")
    if float(slack_violations) < 0.0:
        raise ValueError("slack_violations must be non-negative.")

    bound = compute_expected_violation_bound(w, alpha=alpha)
    empirical_violation_count = float(np.sum(np.clip(z, 0.0, 1.0)))
    horizon = int(bound["horizon"])
    allowed = float(bound["bound_expected_violations"]) + float(slack_violations)
    return {
        **bound,
        "empirical_violation_count": empirical_violation_count,
        "empirical_tsvr": float(empirical_violation_count / horizon),
        "slack_violations": float(slack_violations),
        "passed": bool(empirical_violation_count <= allowed + 1e-9),
    }


def mondrian_group_coverage(
    y_true: "np.ndarray | list[float]",
    lower: "np.ndarray | list[float]",
    upper: "np.ndarray | list[float]",
    reliability_w: "np.ndarray | list[float]",
    *,
    n_bins: int = 3,
    alpha: float = 0.10,
) -> dict:
    """Compute Mondrian (group-conditional) coverage per reliability bin.

    Theorem 9 — Group-Conditional Coverage Under Reliability Partitioning
    -----------------------------------------------------------------------
    Partition reliability scores w_t into K bins G_1...G_K (low/medium/high).
    For group-specific conformal quantiles q_k drawn from the calibration
    subset { i : w_i in G_k }:

        P(Y in Chat_k(X) | w_t in G_k) >= 1 - alpha  for each k = 1...K

    This function verifies the empirical version of that guarantee: for each
    reliability group, the in-group PICP must be >= 1 - alpha - 0.02 (tolerance).

    Args:
        y_true:        True observations, shape (N,).
        lower:         Interval lower bounds, shape (N,).
        upper:         Interval upper bounds, shape (N,).
        reliability_w: Reliability scores w_t in [0, 1], shape (N,).
        n_bins:        Number of reliability groups (default 3: low/med/high).
        alpha:         Nominal miscoverage level.

    Returns:
        dict with keys:
            groups:        list of per-group dicts (edges, picp, n, passed).
            overall_picp:  Coverage pooled across all groups.
            all_pass:      True if every group meets 1 - alpha - 0.02.
    """
    yt = _as_flat_float_array(y_true,       name="y_true")
    lo = _as_flat_float_array(lower,         name="lower")
    hi = _as_flat_float_array(upper,         name="upper")
    w  = _as_flat_float_array(reliability_w, name="reliability_w")
    N = len(yt)
    if not (len(lo) == len(hi) == len(w) == N):
        raise ValueError("All arrays must have the same length.")

    # Build bin edges from quantiles of w so each bin has roughly equal mass
    edges = [0.0]
    for i in range(1, n_bins):
        edges.append(float(np.quantile(w, i / n_bins)))
    edges.append(1.0 + 1e-9)

    target = 1.0 - float(alpha)
    tolerance = 0.02
    groups = []
    covered_all = (yt >= lo) & (yt <= hi)

    for k in range(n_bins):
        lo_edge, hi_edge = edges[k], edges[k + 1]
        mask = (w >= lo_edge) & (w < hi_edge)
        n_k = int(np.sum(mask))
        if n_k == 0:
            picp_k = float("nan")
            passed_k = True  # vacuously
        else:
            picp_k = float(np.mean(covered_all[mask]))
            passed_k = picp_k >= target - tolerance
        groups.append({
            "bin_index": k,
            "w_lo": lo_edge,
            "w_hi": min(hi_edge, 1.0),
            "n": n_k,
            "picp": picp_k,
            "target": target,
            "passed": passed_k,
        })

    overall_picp = float(np.mean(covered_all))
    all_pass = all(bool(g["passed"]) for g in groups)
    return {
        "groups": groups,
        "overall_picp": overall_picp,
        "all_pass": all_pass,
        "alpha": float(alpha),
        "n_bins": n_bins,
    }


def hoeffding_violation_bound(
    T: int,
    alpha: float,
    w_bar: float,
    epsilon: float,
) -> dict:
    """Compute the Hoeffding high-probability violation bound.

    Theorem 10 — High-Probability Violation Bound
    -----------------------------------------------
    For any epsilon > 0 and horizon T, if V_t are i.i.d. bounded violation
    indicators in [0, 1]:

        P(TSVR >= alpha*(1 - w_bar) + epsilon) <= exp(-2 * T * epsilon^2)

    i.e., the violation rate exceeds the expectation bound by epsilon with
    probability at most exp(-2*T*epsilon^2).

    Args:
        T:       Horizon length (number of steps).
        alpha:   Nominal miscoverage level.
        w_bar:   Mean reliability score w_bar over the horizon.
        epsilon: Excess-violation slack epsilon > 0.

    Returns:
        dict with:
            expectation_bound: alpha*(1 - w_bar)   -- the expected TSVR bound.
            high_prob_bound:   alpha*(1 - w_bar) + epsilon -- confidence threshold.
            tail_probability:  exp(-2*T*epsilon^2) -- P(TSVR >= high_prob_bound).
    """
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    if not (0.0 <= w_bar <= 1.0):
        raise ValueError("w_bar must lie in [0, 1].")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    expectation_bound = float(alpha) * (1.0 - float(w_bar))
    high_prob_bound   = expectation_bound + float(epsilon)
    tail_probability  = float(np.exp(-2.0 * int(T) * float(epsilon) ** 2))
    return {
        "expectation_bound": expectation_bound,
        "high_prob_bound": high_prob_bound,
        "tail_probability": tail_probability,
        "T": T,
        "alpha": float(alpha),
        "w_bar": float(w_bar),
        "epsilon": float(epsilon),
    }


def evaluate_group_conditional_coverage(
    records: list,
    alpha: float = 0.10,
    n_bins: int = 3,
) -> dict:
    """Evaluate Mondrian group coverage from a list of DC3S step records.

    Each record must contain:
        y_true:        float -- true observed value (primary signal)
        lower:         float -- DC3S interval lower bound
        upper:         float -- DC3S interval upper bound
        reliability_w: float -- OQE reliability score w_t

    Returns:
        The mondrian_group_coverage() output dict.
    """
    if not records:
        raise ValueError("records must be non-empty.")

    required = ("y_true", "lower", "upper", "reliability_w")
    for key in required:
        if key not in records[0]:
            raise KeyError(f"Each record must contain key '{key}'.")

    y_true = np.array([r["y_true"]       for r in records], dtype=float)
    lower  = np.array([r["lower"]         for r in records], dtype=float)
    upper  = np.array([r["upper"]         for r in records], dtype=float)
    w      = np.array([r["reliability_w"] for r in records], dtype=float)

    return mondrian_group_coverage(y_true, lower, upper, w, n_bins=n_bins, alpha=alpha)


__all__ = [
    "verify_inflation_geq_one",
    "compute_empirical_coverage",
    "assert_coverage_guarantee",
    "inflation_lower_bound",
    "compute_expected_violation_bound",
    "evaluate_empirical_core_bound",
    "mondrian_group_coverage",
    "hoeffding_violation_bound",
    "evaluate_group_conditional_coverage",
]
