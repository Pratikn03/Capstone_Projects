"""Population strict-commitment frontier for a declared target class (K-Bound Theorems 1-3).

This module is intentionally separate from :mod:`kga.policy`. The population
frontier consumes an observable population centered-correctness margin ``M``
on the disagreement region ``D = {x : f_0(x) != f_a(x)}`` and an externally
declared calibration-residual budget ``beta`` (where ``|gamma| <= beta``).
Empirical KGA consumes ``Delta_hat`` and ``epsilon``. Neither pair estimates
or aliases the other.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from kga.policy import Decision

Action = Decision
ADAPT = Decision.ADAPT
FREEZE = Decision.FREEZE
ABSTAIN = Decision.ABSTAIN


def decide_action(delta_hat: float, epsilon: float) -> Decision:
    """Empirical KGA interval decision: ADAPT if L > 0, FREEZE if U < 0, else ABSTAIN."""
    if not math.isfinite(epsilon) or epsilon < 0:
        return Decision.ABSTAIN
    if delta_hat - epsilon > 0.0:
        return Decision.ADAPT
    if delta_hat + epsilon < 0.0:
        return Decision.FREEZE
    return Decision.ABSTAIN


@dataclass(frozen=True)
class PopulationBenefitInterval:
    """Population-benefit interval after an explicit disagreement-mass conversion.

    For disagreement probability ``d = mu_T(D) in (0, 1]``, the feasible
    population benefit set under ``|gamma| <= beta`` is
    ``Delta(C_beta) = 2 * d * [max(-1/2, M - beta), min(1/2, M + beta)]``.
    """

    mass: float
    lower: float
    upper: float


@dataclass(frozen=True)
class FrontierAssessment:
    """Auditable centered-correctness result of one K-Bound frontier evaluation."""

    margin: float
    residual_budget: float
    lower_centered_correctness: float
    upper_centered_correctness: float
    action: Decision

    @property
    def drift_budget(self) -> float:
        """Backward-compatible alias for ``residual_budget``."""
        return self.residual_budget

    @property
    def lower_benefit(self) -> float:
        """Backward-compatible alias for ``lower_centered_correctness``."""
        return self.lower_centered_correctness

    @property
    def upper_benefit(self) -> float:
        """Backward-compatible alias for ``upper_centered_correctness``."""
        return self.upper_centered_correctness

    @property
    def fibre_audit_floor(self) -> float:
        """Exact label-free fibre-wise audit floor Gamma_z(C_beta) = beta (Theorem 3)."""
        return min(0.5, self.residual_budget)

    def population_benefit_interval(self, *, mass: float) -> PopulationBenefitInterval:
        """Convert to population-benefit units using disagreement probability ``mass = mu_T(D)``."""
        if isinstance(mass, bool) or not isinstance(mass, Real):
            raise ValueError("mass must be a real number, not boolean or text")
        population_mass = float(mass)
        if not math.isfinite(population_mass) or not 0 < population_mass <= 1:
            raise ValueError(f"mass must be finite and lie in (0, 1], got {population_mass}")

        scale = 2.0 * population_mass
        return PopulationBenefitInterval(
            mass=population_mass,
            lower=scale * self.lower_centered_correctness,
            upper=scale * self.upper_centered_correctness,
        )


def assess_frontier(M: float, beta: float | None) -> FrontierAssessment:
    """Evaluate the maximal sound strict action over ``|gamma| <= beta`` (Theorem 2).

    ``M`` is the disagreement-conditional centered-correctness margin in ``[-0.5, 0.5]``.
    ``beta`` bounds the target disagreement-conditional calibration residual ``gamma``.
    ``None`` is not interpreted as zero: when no credible budget exists, the function
    fails closed instead of manufacturing a commitment certificate.
    """
    if isinstance(M, bool) or not isinstance(M, Real):
        raise ValueError("M must be a real number, not boolean or text")
    margin = float(M)
    if beta is None:
        raise ValueError("beta is required and must be externally specified")
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise ValueError("beta must be a real number, not boolean or text")
    residual_budget = float(beta)
    if not math.isfinite(margin) or not -0.5 <= margin <= 0.5:
        raise ValueError(f"M must be finite and lie in [-0.5, 0.5], got {margin}")
    if not math.isfinite(residual_budget) or residual_budget < 0:
        raise ValueError(f"beta must be finite and nonnegative, got {residual_budget}")

    lower = max(-0.5, margin - residual_budget)
    upper = min(0.5, margin + residual_budget)
    if lower > 0:
        action = Decision.ADAPT
    elif upper < 0:
        action = Decision.FREEZE
    else:
        action = Decision.ABSTAIN
    return FrontierAssessment(margin, residual_budget, lower, upper, action)


def frontier_action(M: float, beta: float | None) -> Decision:
    """Return ADAPT/FREEZE/ABSTAIN under the K-Bound population frontier."""
    return assess_frontier(M, beta).action


def frontier_sensitivity(M: float, beta_values: Iterable[float]) -> list[FrontierAssessment]:
    """Evaluate declared residual budgets without selecting one from outcomes."""
    values = list(beta_values)
    if not values:
        raise ValueError("beta_values must be nonempty")
    return [assess_frontier(M, beta) for beta in values]
