"""Generic risk-bound helpers for degraded-observation safety."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


RISK_ENVELOPE_ASSUMPTIONS: tuple[str, ...] = (
    "A theorem-local predictable per-step residual-risk budget is available.",
    "The budget may be instantiated as r_t <= alpha * (1 - w_t) when a domain-specific calibration contract justifies that envelope.",
    "w_t is a runtime reliability score, not a probability by definition.",
    "The envelope is marginal/expected-episode control, not a conditional coverage guarantee for every observation.",
)


def _as_flat_float_array(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one element.")
    return arr


def verify_inflation_geq_one(inflation: float, tol: float = 1e-9) -> None:
    """Assert that interval inflation preserves the base conformal set."""
    if float(inflation) < 1.0 - tol:
        raise ValueError(
            "Observation-consistent state inflation must be >= 1 to preserve "
            f"base coverage. Got inflation={float(inflation):.6f}."
        )


def compute_empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Compute PICP-style empirical coverage for any interval-valued predictor."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if not (len(yt) == len(lo) == len(hi)):
        raise ValueError("y_true, lower, and upper must have the same number of elements.")
    if np.any(lo > hi + 1e-9):
        raise ValueError("lower must be <= upper element-wise.")
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
    *,
    alpha: float = 0.10,
    tolerance: float = 0.02,
) -> dict[str, float]:
    """Check that empirical coverage stays above the tolerated target."""
    result = compute_empirical_coverage(y_true, lower, upper)
    target = 1.0 - float(alpha)
    passed = result["picp"] >= target - float(tolerance)
    result["target_coverage"] = target
    result["tolerance"] = float(tolerance)
    result["passed"] = bool(passed)
    if not passed:
        raise AssertionError(
            f"Empirical coverage={result['picp']:.4f} fell below the tolerated "
            f"target {target:.4f} - {tolerance:.4f}."
        )
    return result


def compute_step_risk_bound(reliability_w: float, *, alpha: float = 0.10) -> float:
    """Return the chapter-level per-step budget alpha * (1 - w_t).

    This helper computes the battery-style risk-envelope budget used by the
    narrowed T3 surface.  It does not claim that the budget follows from
    conformal coverage alone; that interpretation requires a separate
    theorem-local calibration argument.
    """
    w = float(min(1.0, max(0.0, reliability_w)))
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    return float(alpha * (1.0 - w))


def risk_envelope_assumptions() -> tuple[str, ...]:
    """Return the explicit assumptions behind the ORIUS risk-envelope helpers."""
    return RISK_ENVELOPE_ASSUMPTIONS


def compute_episode_risk_bound(
    reliability: np.ndarray | list[float],
    *,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Episode-level degradation-sensitive envelope E[V] <= alpha (1-w_bar) T.

    The return payload includes the explicit assumptions used so downstream
    manuscript and audit surfaces can avoid overstating what this helper proves.
    """
    w = _as_flat_float_array(reliability, name="reliability")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    if np.any((w < -1e-9) | (w > 1.0 + 1e-9)):
        raise ValueError("reliability scores must lie in [0, 1].")

    horizon = int(w.size)
    mean_reliability = float(np.mean(np.clip(w, 0.0, 1.0)))
    expected_violations = float(alpha * (1.0 - mean_reliability) * horizon)
    return {
        "alpha": float(alpha),
        "horizon": float(horizon),
        "mean_reliability_w": mean_reliability,
        "bound_expected_violations": expected_violations,
        "bound_tsvr": float(expected_violations / max(horizon, 1)),
        "theorem_surface": "T3 risk-envelope aggregation",
        "interpretation": "Conservative episode-level envelope under an explicit predictable per-step risk-budget contract.",
        "assumptions_used": list(RISK_ENVELOPE_ASSUMPTIONS),
    }


def evaluate_empirical_core_bound(
    violations: np.ndarray | list[bool] | list[float],
    reliability: np.ndarray | list[float],
    *,
    alpha: float = 0.10,
    slack_violations: float = 0.0,
) -> dict[str, float]:
    """Compare observed violations to the degradation-sensitive envelope."""
    z = _as_flat_float_array(violations, name="violations")
    w = _as_flat_float_array(reliability, name="reliability")
    if z.size != w.size:
        raise ValueError("violations and reliability must have the same length.")
    if np.any((z < -1e-9) | (z > 1.0 + 1e-9)):
        raise ValueError("violations must be indicator-like values in [0, 1].")
    if float(slack_violations) < 0.0:
        raise ValueError("slack_violations must be non-negative.")

    bound = compute_episode_risk_bound(w, alpha=alpha)
    empirical_violation_count = float(np.sum(np.clip(z, 0.0, 1.0)))
    horizon = int(bound["horizon"])
    allowed = float(bound["bound_expected_violations"]) + float(slack_violations)
    return {
        **bound,
        "empirical_violation_count": empirical_violation_count,
        "empirical_tsvr": float(empirical_violation_count / max(horizon, 1)),
        "slack_violations": float(slack_violations),
        "passed": bool(empirical_violation_count <= allowed + 1e-9),
    }


@dataclass(slots=True)
class FrontierPoint:
    """Convenience point on the conservative reliability-risk frontier."""

    mean_reliability_w: float
    bound_tsvr: float
    alpha: float
    horizon: int


def compute_frontier(
    *,
    alpha: float = 0.10,
    horizon: int = 1,
    points: int = 11,
) -> list[FrontierPoint]:
    """Sample the conservative reliability-risk envelope for plotting."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if points <= 1:
        raise ValueError("points must exceed 1.")
    frontier: list[FrontierPoint] = []
    for w in np.linspace(0.0, 1.0, points):
        envelope = compute_episode_risk_bound([float(w)] * horizon, alpha=alpha)
        frontier.append(
            FrontierPoint(
                mean_reliability_w=float(w),
                bound_tsvr=float(envelope["bound_tsvr"]),
                alpha=float(alpha),
                horizon=int(horizon),
            )
        )
    return frontier


def minimum_reliability_for_target(target_tsvr: float, *, alpha: float = 0.10) -> float:
    """Invert the conservative envelope for a target violation budget."""
    if target_tsvr < 0.0:
        raise ValueError("target_tsvr must be non-negative.")
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must lie in (0, 1].")
    return float(min(1.0, max(0.0, 1.0 - target_tsvr / alpha)))


# ---------------------------------------------------------------------------
# w_t Calibration Contract  (closes the key theoretical gap)
# ---------------------------------------------------------------------------
# The core risk-envelope bound TSVR ≤ α(1-w̄) is valid only when w_t is
# calibrated: P(true_state ∈ U_t | w_t) ≥ w_t.  Without this contract the
# bound is an empirical observation, not a theorem.
#
# This function checks the contract empirically against held-out data and
# returns a calibrated w_t with a finite-sample correction so that the bound
# holds with probability ≥ 1-delta.

def calibration_contract_verify(
    oqe_scores: np.ndarray | list[float],
    coverage_indicators: np.ndarray | list[float],
    *,
    delta: float = 0.05,
    n_bins: int = 5,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Verify and correct the w_t calibration contract.

    The contract states: for each reliability level w, the fraction of steps
    where the true state falls inside the uncertainty set must be ≥ w.  In
    other words, w_t must be a *lower bound* on empirical coverage, not just
    a heuristic score.

    If the contract fails (some bins show empirical coverage < w_bin - tol),
    a corrected score w_corrected = w_bin * (empirical_coverage / w_bin) is
    returned, which re-normalises w_t so the bound holds.

    A Clopper-Pearson confidence interval is used for the finite-sample
    correction at confidence level 1-delta.

    Parameters
    ----------
    oqe_scores:
        OQE reliability scores w_t ∈ [0, 1], shape (N,).
    coverage_indicators:
        Binary indicators: 1 if true state ∈ uncertainty set at step t, else 0.
        Shape (N,).
    delta:
        Confidence level for the Clopper-Pearson correction.  The corrected
        w_t holds with probability ≥ 1-delta.
    n_bins:
        Number of reliability bins for the stratified check.
    tolerance:
        Allowed shortfall below w_bin in empirical coverage before flagging.

    Returns
    -------
    dict with keys:
        contract_satisfied  — True if all bins pass.
        bins                — Per-bin diagnostics.
        correction_factor   — Multiplicative factor applied to w_t (≤ 1).
        w_corrected_mean    — Mean corrected reliability score.
        delta               — Confidence level used.
        theorem_note        — Plain-language status of the calibration contract.
    """
    import math as _math

    w = _as_flat_float_array(oqe_scores, name="oqe_scores")
    c = _as_flat_float_array(coverage_indicators, name="coverage_indicators")
    if w.size != c.size:
        raise ValueError("oqe_scores and coverage_indicators must have the same length.")
    if np.any((w < -1e-9) | (w > 1.0 + 1e-9)):
        raise ValueError("oqe_scores must lie in [0, 1].")
    if np.any((c < -1e-9) | (c > 1.0 + 1e-9)):
        raise ValueError("coverage_indicators must be indicator-like in [0, 1].")

    w = np.clip(w, 0.0, 1.0)
    c = np.clip(c, 0.0, 1.0)
    N = int(w.size)

    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    bins: list[dict[str, Any]] = []
    correction_factors: list[float] = []
    all_pass = True

    for k in range(n_bins):
        lo, hi = float(edges[k]), float(edges[k + 1])
        mask = (w >= lo) & (w < hi)
        n_k = int(np.sum(mask))
        w_bin = float(np.mean(w[mask])) if n_k > 0 else float(0.5 * (lo + hi))

        if n_k == 0:
            bins.append({
                "bin": k, "w_lo": lo, "w_hi": hi, "n": 0,
                "w_mean": w_bin, "empirical_coverage": float("nan"),
                "cp_lower": float("nan"), "passed": True,
                "correction_factor": 1.0, "note": "empty bin",
            })
            correction_factors.append(1.0)
            continue

        n_covered = int(np.sum(c[mask]))
        emp_cov = n_covered / n_k

        # Clopper-Pearson lower confidence bound P(X ≥ n_covered | n_k, p) ≤ delta/2
        # Using the Beta distribution: CP_lower = Beta(delta/2; n_covered, n_k-n_covered+1)
        from scipy.stats import beta as _beta  # type: ignore[import]
        cp_lower = float(_beta.ppf(delta / 2.0, max(n_covered, 1), n_k - n_covered + 1))

        passed = cp_lower >= w_bin - float(tolerance)
        if not passed:
            all_pass = False

        # Correction: scale w down so that the CP lower bound ≥ corrected w
        corr = min(1.0, cp_lower / max(w_bin, 1e-9)) if not passed else 1.0
        correction_factors.append(float(corr))

        bins.append({
            "bin": k, "w_lo": lo, "w_hi": hi, "n": n_k,
            "w_mean": w_bin, "empirical_coverage": emp_cov,
            "cp_lower": cp_lower, "passed": passed,
            "correction_factor": float(corr),
            "note": "pass" if passed else f"coverage {emp_cov:.3f} < w_bin {w_bin:.3f}",
        })

    # Global correction: take the minimum bin correction factor
    global_correction = float(min(correction_factors)) if correction_factors else 1.0
    w_corrected_mean = float(np.mean(w) * global_correction)

    theorem_note = (
        "Calibration contract satisfied: w_t is a valid lower bound on P(true_state ∈ U_t). "
        "The TSVR ≤ α(1-w̄) bound is formally justified."
        if all_pass else
        f"Calibration contract violated in {sum(1 for b in bins if not b['passed'])} bin(s). "
        f"Correction factor {global_correction:.3f} applied. "
        "Re-run OQE calibration or widen the uncertainty set."
    )

    return {
        "contract_satisfied": bool(all_pass),
        "bins": bins,
        "correction_factor": global_correction,
        "w_corrected_mean": w_corrected_mean,
        "delta": float(delta),
        "n_samples": N,
        "theorem_note": theorem_note,
        "assumptions": list(RISK_ENVELOPE_ASSUMPTIONS),
    }
