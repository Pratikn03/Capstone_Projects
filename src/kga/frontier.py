"""Population strict-commitment frontier for a declared target class.

This module is intentionally separate from :mod:`kga.policy`. The population
frontier consumes an observable population margin ``M`` and an externally
declared drift budget ``beta``. Empirical KGA consumes ``Delta_hat`` and
``epsilon``. Neither pair estimates or aliases the other.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from kga.policy import Decision


@dataclass(frozen=True)
class FrontierAssessment:
    """Auditable result of one population-frontier evaluation."""

    margin: float
    drift_budget: float
    lower_benefit: float
    upper_benefit: float
    action: Decision


def assess_frontier(M: float, beta: float | None) -> FrontierAssessment:
    """Evaluate the maximal sound strict action over ``|gamma| <= beta``.

    ``beta`` must come from the declared deployment class. ``None`` is not
    interpreted as zero: when no credible budget exists, the function fails
    closed instead of manufacturing a commitment certificate.
    """
    margin = float(M)
    if beta is None:
        raise ValueError("beta is required and must be externally specified")
    drift_budget = float(beta)
    if not math.isfinite(margin):
        raise ValueError(f"M must be finite, got {margin}")
    if not math.isfinite(drift_budget) or drift_budget < 0:
        raise ValueError(f"beta must be finite and nonnegative, got {drift_budget}")

    lower = margin - drift_budget
    upper = margin + drift_budget
    if lower > 0:
        action = Decision.ADAPT
    elif upper < 0:
        action = Decision.FREEZE
    else:
        action = Decision.ABSTAIN
    return FrontierAssessment(margin, drift_budget, lower, upper, action)


def frontier_action(M: float, beta: float | None) -> Decision:
    """Return ADAPT/FREEZE/ABSTAIN under the population frontier."""
    return assess_frontier(M, beta).action


def frontier_sensitivity(M: float, beta_values: Iterable[float]) -> list[FrontierAssessment]:
    """Evaluate externally supplied drift budgets without selecting one from outcomes."""
    values = list(beta_values)
    if not values:
        raise ValueError("beta_values must be nonempty")
    return [assess_frontier(M, beta) for beta in values]
