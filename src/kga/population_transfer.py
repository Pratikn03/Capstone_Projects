"""kga.population_transfer -- Conditional population-benefit interval composition.

This module implements the conservative bridge from cell-level empirical calibration
to population-benefit bounds (Proposition 3 and Appendix A in the manuscript).

The interval is:
    I_pop = [delta_hat - (epsilon + b), delta_hat + (epsilon + b)]

where:
    * delta_hat is the point prediction of benefit.
    * epsilon is the exact-rank conformal residual radius at level alpha_cell.
    * b is the fresh-sample Hoeffding paired-accuracy radius at level delta_sampling:
          b = min(2.0, sqrt(2.0 * log(2.0 / delta_sampling) / m))
    * alpha_cell + delta_sampling <= alpha_population.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kga.policy import Decision


@dataclass(frozen=True)
class ConditionalPopulationInterval:
    """Represents a compound conformal-concentration population benefit interval."""

    delta_hat: float
    epsilon: float
    r_samp: float
    population_radius: float
    interval_lower: float
    interval_upper: float
    action: Decision
    alpha_cell: float
    delta_sampling: float
    alpha_population: float
    spent_budget: float


def hoeffding_paired_accuracy_radius(*, n: int, delta: float) -> float:
    """Compute the finite-sample Hoeffding paired-accuracy radius for m fresh samples.

    For paired 0-1 loss differences W_j in [-1, 1], the range of each observation is 2.
    By Hoeffding's inequality, for any delta in (0, 1):
        Pr[ |B - Delta| >= b ] <= 2 exp( - 2 * n^2 * b^2 / (n * 2^2) )
                                = 2 exp( - n * b^2 / 2 )
    Equating to delta yields:
        b = min(2.0, sqrt(2.0 * ln(2.0 / delta) / n)).

    Parameters
    ----------
    n : int
        Fresh evaluation sample count (must be positive integer, not bool).
    delta : float
        Sampling error probability tolerance in (0, 1).

    Returns
    -------
    float
        Sampling radius b.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"Sample count n must be an integer, got {type(n).__name__}")
    if n <= 0:
        raise ValueError(f"Sample count n must be positive, got {n}")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool):
        raise TypeError(f"Tolerance delta must be a float, got {type(delta).__name__}")
    if math.isnan(delta) or delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"Tolerance delta must be in (0, 1), got {delta}")

    b_unbounded = math.sqrt(2.0 * math.log(2.0 / float(delta)) / float(n))
    return min(2.0, b_unbounded)


def compose_conditional_population_interval(
    *,
    delta_hat: float,
    epsilon: float,
    r_samp: float,
    alpha_cell: float,
    delta_sampling: float,
    alpha_population: float,
) -> ConditionalPopulationInterval:
    """Compose a cell conformal radius and sampling radius into a population interval.

    Parameters
    ----------
    delta_hat : float
        Estimated benefit.
    epsilon : float
        Nonnegative or infinite cell calibration radius.
    r_samp : float
        Nonnegative sampling radius (e.g. from hoeffding_paired_accuracy_radius).
    alpha_cell : float
        Cell calibration error budget in (0, 1).
    delta_sampling : float
        Sampling error budget in (0, 1).
    alpha_population : float
        Total allowable population risk budget in (0, 1).

    Returns
    -------
    ConditionalPopulationInterval
        The composed interval and strict trichotomy action.
    """
    # Validate budgets
    for name, val in [
        ("alpha_cell", alpha_cell),
        ("delta_sampling", delta_sampling),
        ("alpha_population", alpha_population),
    ]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(val).__name__}")
        if math.isnan(val) or val <= 0.0 or val >= 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {val}")

    spent_budget = float(alpha_cell) + float(delta_sampling)
    if spent_budget > float(alpha_population) + 1e-12:
        raise ValueError(
            f"Budget overspent: alpha_cell ({alpha_cell}) + delta_sampling ({delta_sampling}) "
            f"= {spent_budget} > alpha_population ({alpha_population})"
        )

    # Validate delta_hat
    if isinstance(delta_hat, bool) or not isinstance(delta_hat, (int, float)) or math.isnan(delta_hat):
        raise ValueError(f"delta_hat must be a valid real number, got {delta_hat}")

    # Validate epsilon
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or math.isnan(epsilon):
        raise ValueError(f"epsilon must be a valid number, got {epsilon}")
    if epsilon < 0.0:
        raise ValueError(f"epsilon cannot be negative, got {epsilon}")

    # Validate r_samp
    if isinstance(r_samp, bool) or not isinstance(r_samp, (int, float)) or math.isnan(r_samp):
        raise ValueError(f"r_samp must be a valid number, got {r_samp}")
    if r_samp < 0.0 or math.isinf(r_samp):
        raise ValueError(f"r_samp must be finite and nonnegative, got {r_samp}")

    # Infinite radius handling
    if math.isinf(epsilon):
        return ConditionalPopulationInterval(
            delta_hat=float(delta_hat),
            epsilon=float("inf"),
            r_samp=float(r_samp),
            population_radius=float("inf"),
            interval_lower=float("-inf"),
            interval_upper=float("inf"),
            action=Decision.ABSTAIN,
            alpha_cell=float(alpha_cell),
            delta_sampling=float(delta_sampling),
            alpha_population=float(alpha_population),
            spent_budget=spent_budget,
        )

    pop_radius = float(epsilon) + float(r_samp)
    lower = float(delta_hat) - pop_radius
    upper = float(delta_hat) + pop_radius

    # Strict trichotomy rules
    if lower > 0.0:
        action = Decision.ADAPT
    elif upper < 0.0:
        action = Decision.FREEZE
    else:
        action = Decision.ABSTAIN

    return ConditionalPopulationInterval(
        delta_hat=float(delta_hat),
        epsilon=float(epsilon),
        r_samp=float(r_samp),
        population_radius=pop_radius,
        interval_lower=lower,
        interval_upper=upper,
        action=action,
        alpha_cell=float(alpha_cell),
        delta_sampling=float(delta_sampling),
        alpha_population=float(alpha_population),
        spent_budget=spent_budget,
    )
