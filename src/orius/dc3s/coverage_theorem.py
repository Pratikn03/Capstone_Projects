"""Backward-compatible access to universal degraded-observation bounds."""
from __future__ import annotations

import numpy as np
from orius.universal_theory.risk_bounds import (
    assert_coverage_guarantee,
    compute_empirical_coverage,
    compute_episode_risk_bound,
    evaluate_empirical_core_bound,
    risk_envelope_assumptions,
    verify_inflation_geq_one,
)


def _as_flat_float_array(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one element.")
    return arr


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


def compute_expected_violation_bound(
    reliability: np.ndarray | list[float],
    *,
    alpha: float = 0.10,
) -> dict:
    """Backward-compatible wrapper for the narrowed T3 risk-envelope helper."""
    return compute_episode_risk_bound(reliability, alpha=alpha)


def get_core_bound_assumptions() -> tuple[str, ...]:
    """Expose the explicit assumptions behind the core-envelope helper."""
    return risk_envelope_assumptions()




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

    Legacy auxiliary coverage surface
    ---------------------------------
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
    """Compute a legacy Hoeffding-style high-probability violation envelope.

    Legacy auxiliary coverage surface
    ---------------------------------
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
    "get_core_bound_assumptions",
    "evaluate_empirical_core_bound",
    "mondrian_group_coverage",
    "hoeffding_violation_bound",
    "evaluate_group_conditional_coverage",
]
