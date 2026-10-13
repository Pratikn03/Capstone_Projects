"""Sensitivity analysis and Manski partial-identification frontiers for KGA certificates.

This module formalizes continuous sensitivity analysis over unobserved calibration
bias on the disagreement region, generalizing Rosenbaum's Gamma and Manski's
partial identification bounds to online test-time model adaptation.

Mathematical Formulation:
-------------------------
Let Delta = R(f_0) - R(f_1) denote the true population advantage of adapted f_1 over f_0.
Under an unobserved calibration residual bounded by |gamma(x)| <= beta on the disagreement
set D = {x : f_1(x) != f_0(x)} with probability d = P(D), the identifiable interval
satisfies:
    Delta in [Delta_0 - 2 * beta * d, Delta_0 + 2 * beta * d]
where Delta_0 is the nominal point estimate. Given a statistical certificate with
finite-sample uncertainty radius epsilon (so that [lower, upper] = [Delta_0 - epsilon, Delta_0 + epsilon]),
the sensitivity interval under residual budget beta is:
    [lower - 2 * beta * d,  upper + 2 * beta * d]

The Break-Even Sensitivity Threshold beta*(Delta) is the minimum calibration residual
required to overturn the certificate decision:
    beta*(Delta) = max(0.0, (|Delta_0| - epsilon) / (2 * d))
When d is unknown or bounded by worst-case d = 1, this evaluates directly to:
    beta* = max(0.0, lower / 2)   if ADAPT (lower > 0)
    beta* = max(0.0, -upper / 2)  if FREEZE (upper < 0)
    beta* = 0.0                   if ABSTAIN (lower <= 0 <= upper)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, Union


@dataclass(frozen=True)
class SensitivityPoint:
    """A single evaluation point along the sensitivity frontier."""
    beta: float
    lower_bound: float
    upper_bound: float
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "action": self.action,
        }


def compute_break_even_beta(
    lower: float,
    upper: float,
    *,
    disagreement_rate: float = 1.0,
) -> float:
    """Compute the minimum unobserved residual beta* needed to overturn the decision.

    Parameters
    ----------
    lower : float
        Lower bound of the certificate confidence interval.
    upper : float
        Upper bound of the certificate confidence interval.
    disagreement_rate : float, optional
        Fraction of samples where f_1 != f_0, d in (0, 1]. Default 1.0 (conservative).

    Returns
    -------
    float
        beta* >= 0.0. If evidence is already ambiguous (lower <= 0 <= upper), returns 0.0.
    """
    if not (0.0 < disagreement_rate <= 1.0):
        raise ValueError(f"disagreement_rate must be in (0, 1], got {disagreement_rate!r}")
    if lower > upper:
        raise ValueError(f"lower bound {lower} exceeds upper bound {upper}")

    slack_factor = 2.0 * disagreement_rate
    if lower > 0.0:
        return float(lower / slack_factor)
    if upper < 0.0:
        return float(-upper / slack_factor)
    return 0.0


def compute_sensitivity_interval(
    lower: float,
    upper: float,
    beta: float,
    *,
    disagreement_rate: float = 1.0,
) -> tuple[float, float]:
    """Compute the Manski partial-identification interval under residual budget beta.

    Parameters
    ----------
    lower : float
        Certificate lower bound.
    upper : float
        Certificate upper bound.
    beta : float
        Declared calibration residual budget (beta >= 0).
    disagreement_rate : float, optional
        Disagreement set probability d in (0, 1]. Default 1.0.

    Returns
    -------
    tuple[float, float]
        [lower - 2 * beta * d, upper + 2 * beta * d]
    """
    if beta < 0.0 or not math.isfinite(beta):
        raise ValueError(f"beta must be non-negative and finite, got {beta!r}")
    if not (0.0 < disagreement_rate <= 1.0):
        raise ValueError(f"disagreement_rate must be in (0, 1], got {disagreement_rate!r}")
    slack = 2.0 * float(beta) * float(disagreement_rate)
    return (lower - slack, upper + slack)


def compute_sensitivity_frontier(
    lower: float,
    upper: float,
    *,
    max_beta: float = 0.20,
    steps: int = 21,
    disagreement_rate: float = 1.0,
) -> list[dict[str, Any]]:
    """Compute the decision frontier across a grid of declared beta values.

    Parameters
    ----------
    lower : float
        Certificate lower bound.
    upper : float
        Certificate upper bound.
    max_beta : float, optional
        Maximum beta to evaluate. Default 0.20.
    steps : int, optional
        Number of evaluation points. Default 21.
    disagreement_rate : float, optional
        Disagreement rate d in (0, 1]. Default 1.0.

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries with keys: 'beta', 'lower_bound', 'upper_bound', 'action'.
    """
    if max_beta <= 0.0 or not math.isfinite(max_beta):
        raise ValueError(f"max_beta must be positive, got {max_beta!r}")
    if steps < 2:
        raise ValueError(f"steps must be >= 2, got {steps}")

    step_size = max_beta / (steps - 1)
    frontier: list[dict[str, Any]] = []
    for i in range(steps):
        b = i * step_size
        low, high = compute_sensitivity_interval(lower, upper, b, disagreement_rate=disagreement_rate)
        if low > 0.0:
            action = "ADAPT"
        elif high < 0.0:
            action = "FREEZE"
        else:
            action = "ABSTAIN"
        frontier.append({
            "beta": round(b, 6),
            "lower_bound": low,
            "upper_bound": high,
            "action": action,
        })
    return frontier


class SensitivityFrontier:
    """Wrapper providing sensitivity analysis on top of any Certificate or bounds."""

    def __init__(
        self,
        certificate_or_lower: Any,
        upper: float | None = None,
        *,
        disagreement_rate: float = 1.0,
    ) -> None:
        if hasattr(certificate_or_lower, "lower") and hasattr(certificate_or_lower, "upper"):
            self.lower = float(certificate_or_lower.lower)
            self.upper = float(certificate_or_lower.upper)
            action_attr = getattr(certificate_or_lower, "action", None)
            if action_attr is not None:
                self.action = action_attr.name if hasattr(action_attr, "name") else str(action_attr)
            else:
                self.action = "ADAPT" if self.lower > 0.0 else ("FREEZE" if self.upper < 0.0 else "ABSTAIN")
        elif upper is not None:
            self.lower = float(certificate_or_lower)
            self.upper = float(upper)
            self.action = "ADAPT" if self.lower > 0.0 else ("FREEZE" if self.upper < 0.0 else "ABSTAIN")
        else:
            raise ValueError("Must provide either a Certificate object or both lower and upper bounds")

        if not (0.0 < disagreement_rate <= 1.0):
            raise ValueError(f"disagreement_rate must be in (0, 1], got {disagreement_rate!r}")
        self.disagreement_rate = float(disagreement_rate)

    @property
    def break_even_beta(self) -> float:
        """Minimum residual budget beta* required to overturn the decision."""
        return compute_break_even_beta(self.lower, self.upper, disagreement_rate=self.disagreement_rate)

    def interval(self, beta: float) -> tuple[float, float]:
        """Manski partial-identification interval at declared residual budget beta."""
        return compute_sensitivity_interval(self.lower, self.upper, beta, disagreement_rate=self.disagreement_rate)

    def frontier(self, max_beta: float = 0.20, steps: int = 21) -> list[dict[str, Any]]:
        """Compute the sensitivity frontier across a grid of beta values."""
        return compute_sensitivity_frontier(
            self.lower,
            self.upper,
            max_beta=max_beta,
            steps=steps,
            disagreement_rate=self.disagreement_rate,
        )

    def summary(self) -> dict[str, Any]:
        """Return a structured dictionary summarizing the sensitivity status."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "disagreement_rate": self.disagreement_rate,
            "nominal_action": self.action,
            "break_even_beta": self.break_even_beta,
            "is_robust_to_1pct_bias": self.break_even_beta >= 0.01,
            "is_robust_to_5pct_bias": self.break_even_beta >= 0.05,
        }
