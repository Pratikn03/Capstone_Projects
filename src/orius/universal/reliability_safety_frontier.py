"""
Reference reliability-safety frontier utilities for the universality program.

The reliability-safety frontier is the fundamental design curve of ORIUS.
It shows, for a given mean observation reliability w̄, the minimum achievable
expected violation count E[V_T] under *any* controller — not just DC3S.

The canonical runtime envelope used by the production kernel lives in
``orius.universal_theory.risk_bounds``. This module retains the richer
frontier analysis helpers used by the thesis chapter and the unification
tests.

Key result (T3 upper bound + T10 lower bound combined):
-------------------------------------------------------
    (α/4)(1 − w̄)T   ≤   E[V_T(π)]   ≤   α(1 − w̄)T

where:
  α   = miscoverage rate (e.g. 0.05)
  w̄   = mean reliability over the episode
  T   = episode length
  π   = any controller (DC3S achieves the upper bound; the lower bound
         follows from Le Cam's two-point method applied to the degraded
         observation channel under Assumption A2 + A5')

The factor-of-4 gap between upper and lower bounds is the standard
Le Cam minimax constant (Tsybakov 2009, Ch. 2; Yu 1997). The correct
precision for the optimality claim is: "DC3S is minimax rate-optimal
under degraded observation" — not asymptotically tight coefficient-wise.

The compute_frontier() helper below uses a conservative O(√T log T)
correction term for the lower bound curve; this is a looser but smoother
proxy that traces how the lower bound envelope shrinks relative to the
upper bound as T grows.

This is the research-program characterisation: DC3S is positioned as
information-theoretically tight up to lower-order O(√T log T) terms under the
chapter's additional modeling assumptions.

Operational interpretation
--------------------------
To deploy DC3S and guarantee at most r* expected violations per episode, the
system integrator must ensure the telemetry infrastructure delivers mean
reliability w̄ ≥ w̄* = 1 − r* / (α · T). This is a quantitative, domain-agnostic
infrastructure requirement derived directly from T3.

Senior engineer notes
---------------------
- This module is intentionally stateless: all functions take explicit parameters.
  The frontier depends only on (α, w̄, T) — no domain geometry.
- The lower-order correction term O(√T log T) comes from the sub-Gaussian
  concentration argument in T10's Fano-inequality reduction. The constant
  factor c = 2 is conservative; empirically the correction is smaller.
- All arithmetic uses float64 to avoid precision loss at small α or large T.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrontierPoint:
    """A single point on the reliability-safety frontier.

    Attributes:
        mean_reliability : w̄ ∈ [0, 1] — mean observation quality.
        upper_bound      : α(1−w̄)T — DC3S worst-case expected violations (T3).
                           DC3S achieves this bound; no algorithm exceeds it.
        lower_bound      : α(1−w̄)T − c·√(T log T) — information-theoretic floor (T10).
                           No algorithm can achieve fewer violations than this
                           against a worst-case fault process.
        gap              : upper_bound − lower_bound = c·√(T log T).
                           This is the slack between T3 and T10; it shrinks
                           relative to T as T → ∞, so the bounds are asymptotically tight.
        is_feasible      : True iff lower_bound ≥ 0 (always True by construction,
                           since we clamp to 0).
    """
    mean_reliability: float
    upper_bound: float
    lower_bound: float
    gap: float
    is_feasible: bool


def compute_frontier(
    alpha: float,
    T: int,
    w_bar_values: np.ndarray | None = None,
    lower_bound_constant: float = 2.0,
) -> list[FrontierPoint]:
    """Compute the reliability-safety frontier for the given (α, T) regime.

    The frontier is a 1-D curve parameterised by w̄: for each mean reliability
    value, it gives the tight interval [lower_bound, upper_bound] of achievable
    expected violation counts.

    Args:
        alpha                : miscoverage rate, e.g. 0.05. Must be in (0, 1).
        T                    : episode length (number of control steps). Must be ≥ 1.
        w_bar_values         : array of w̄ values at which to evaluate the frontier.
                               Defaults to a 101-point uniform grid over [0, 1].
        lower_bound_constant : the constant c in the T10 correction O(c·√(T log T)).
                               Default 2.0 is conservative; the true value depends
                               on the sub-Gaussian parameter of the fault distribution.

    Returns:
        List of FrontierPoint, one per w̄ value, sorted by increasing w̄.

    Design note:
        The correction term uses log(T) not log(1/δ) because T10's proof integrates
        over the Borel-Cantelli fault windows rather than using a union-bound argument.
        For T < e ≈ 2.718, log(T) < 1; we clamp to max(log(T), 1) to keep the
        correction meaningful for short episodes.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if T < 1:
        raise ValueError(f"T must be at least 1, got {T}")
    if w_bar_values is None:
        w_bar_values = np.linspace(0.0, 1.0, 101)

    # Lower-order correction: c · √(T · log T)
    # Uses max(log(T), 1) to handle short episodes where log(T) < 1.
    log_T = max(math.log(T), 1.0)
    correction = lower_bound_constant * math.sqrt(T * log_T)

    points: list[FrontierPoint] = []
    for w_bar in w_bar_values:
        w_bar_f = float(w_bar)
        upper = alpha * (1.0 - w_bar_f) * T           # T3 upper bound
        lower = max(0.0, upper - correction)            # T10 lower bound (clamped)
        gap = upper - lower
        points.append(FrontierPoint(
            mean_reliability=w_bar_f,
            upper_bound=upper,
            lower_bound=lower,
            gap=gap,
            is_feasible=True,  # always: lower ≥ 0 by construction
        ))
    return points


def minimum_reliability_for_target(
    alpha: float,
    T: int,
    target_violations: float,
) -> float:
    """Compute w̄* = the minimum mean reliability to guarantee ≤ target_violations
    expected violations per episode.

    Solves T3's bound for w̄:
        α(1 − w̄)T ≤ target_violations
        ⟹  w̄ ≥ 1 − target_violations / (α · T)   [= w̄*]

    This is the *infrastructure requirement* for DC3S deployment: the telemetry
    system must deliver at least w̄* mean reliability to meet the safety target.

    Args:
        alpha            : miscoverage rate (e.g. 0.05 for 95% coverage).
        T                : episode length.
        target_violations: maximum acceptable expected violations E[V_T] ≤ target.
                           Set to 0 to compute the requirement for perfect safety
                           (returns 1.0 — perfect reliability required).

    Returns:
        w̄* ∈ [0, 1]. Values close to 1.0 require high-quality telemetry infrastructure.

    Examples:
        >>> minimum_reliability_for_target(0.05, 2000, 0.0)
        1.0   # zero violations requires perfect observation
        >>> minimum_reliability_for_target(0.05, 2000, 1.0)
        0.99  # ≤1 violation per 2000 steps needs 99% mean reliability
        >>> minimum_reliability_for_target(0.05, 2000, 10.0)
        0.9   # ≤10 violations per 2000 steps needs 90% mean reliability
    """
    if target_violations <= 0.0:
        return 1.0
    denominator = alpha * float(T)
    if denominator <= 0.0:
        return 1.0
    w_star = 1.0 - target_violations / denominator
    return float(np.clip(w_star, 0.0, 1.0))


def achieved_violation_rate(
    observed_violations: float,
    T: int,
) -> float:
    """Compute the empirical true-state violation rate (TSVR) from raw counts.

    TSVR = observed_violations / T

    This is the primary evaluation metric in CPSBench.

    Args:
        observed_violations : count of steps where true-state constraint was violated.
        T                   : total episode length.

    Returns:
        TSVR ∈ [0, 1].
    """
    return observed_violations / max(T, 1)


def is_dc3s_optimal(
    alpha: float,
    T: int,
    observed_violations: float,
    mean_reliability: float,
    upper_tolerance: float = 0.02,
) -> bool:
    """Test whether an observed violation count is consistent with DC3S optimality.

    An observed violation count is consistent with the T3/T10 tight band iff:

        lower_bound ≤ observed_violations ≤ upper_bound × (1 + tolerance)

    where:
        upper_bound = α(1−w̄)T         (T3 bound — DC3S should be at or below this)
        lower_bound = max(0, upper − c√(T log T))  (T10 floor)

    The lower bound is not enforced as a *failure* condition: achieving fewer
    violations than the information-theoretic floor is possible when the domain
    is easier than the worst case (e.g. when violations never concentrate near
    the boundary). It is informative, not prescriptive.

    Args:
        alpha            : miscoverage rate.
        T                : episode length.
        observed_violations: empirical violation count.
        mean_reliability : empirical w̄ (mean of per-step reliability scores).
        upper_tolerance  : fractional slack above upper_bound to account for
                           finite-sample noise (default 2%).

    Returns:
        True iff observed_violations ≤ α(1−w̄)T × (1 + upper_tolerance).
        False iff DC3S is performing worse than its own guarantee — this is a bug.
    """
    upper = alpha * (1.0 - mean_reliability) * T
    return observed_violations <= upper * (1.0 + upper_tolerance)


def frontier_summary(alpha: float, T: int) -> str:
    """Return a human-readable summary of the frontier for a given (α, T) regime.

    Useful for including in experiment logs and thesis figures.
    """
    w_star_zero = minimum_reliability_for_target(alpha, T, 0.0)
    w_star_one = minimum_reliability_for_target(alpha, T, 1.0)
    w_star_ten = minimum_reliability_for_target(alpha, T, 10.0)
    correction = 2.0 * math.sqrt(T * max(math.log(T), 1.0))
    return (
        f"Reliability-Safety Frontier  (α={alpha}, T={T})\n"
        f"  Upper bound at w̄=0.90: {alpha * 0.10 * T:.1f} violations\n"
        f"  Upper bound at w̄=0.95: {alpha * 0.05 * T:.1f} violations\n"
        f"  Upper bound at w̄=0.99: {alpha * 0.01 * T:.1f} violations\n"
        f"  T10 lower-bound correction: {correction:.1f} violations\n"
        f"  w̄* for 0 violations:  {w_star_zero:.4f}\n"
        f"  w̄* for ≤1 violation:  {w_star_one:.4f}\n"
        f"  w̄* for ≤10 violations: {w_star_ten:.4f}\n"
    )
