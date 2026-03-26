"""Generic risk-bound helpers for degraded-observation safety."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
    """Conservative one-step violation budget under degraded observation."""
    w = float(min(1.0, max(0.0, reliability_w)))
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    return float(alpha * (1.0 - w))


def compute_episode_risk_bound(
    reliability: np.ndarray | list[float],
    *,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Episode-level degradation-sensitive envelope E[V] <= alpha (1-w_bar) T."""
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
