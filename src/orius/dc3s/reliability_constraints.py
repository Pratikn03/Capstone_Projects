"""Reliability-Conditioned Safety Constraints.

Tightens the safe-set boundary based on the current reliability score
``w_t``.  When telemetry reliability is degraded (``w_t < 1``), the
actual state may be further from the boundary than the observation
suggests.  This module shrinks the effective safe set by a margin
proportional to ``(1 - w_t)``.

Constraint form::

    h_eff(x, w_t) = h(x) - margin_fn(1 - w_t) >= 0
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ConstraintResult",
    "ConstraintSetResult",
    "MarginFn",
    "constraint_tightening_curve",
    "evaluate_constraint",
    "evaluate_constraint_set",
    "linear_margin",
]

MarginFn = Callable[[float], float]


@dataclass(frozen=True)
class ConstraintResult:
    """Evaluation of a reliability-conditioned constraint at a single state."""

    h_nominal: float
    h_effective: float
    margin: float
    w_t: float
    satisfied: bool


def linear_margin(k: float = 1.0) -> MarginFn:
    """Return a linear margin function ``margin(d) = k * d``."""

    def _fn(degradation: float) -> float:
        return k * degradation

    return _fn


def evaluate_constraint(
    h_nominal: float,
    w_t: float,
    margin_fn: MarginFn | None = None,
    k: float = 1.0,
) -> ConstraintResult:
    """Evaluate a reliability-conditioned safety constraint.

    Parameters
    ----------
    h_nominal : float
        Nominal constraint value at current state.
    w_t : float
        Reliability score in [0, 1].
    margin_fn : callable, optional
        Custom margin function.  Defaults to ``linear_margin(k)``.
    k : float
        Coefficient for the default linear margin.
    """
    if margin_fn is None:
        margin_fn = linear_margin(k)

    degradation = 1.0 - float(np.clip(w_t, 0.0, 1.0))
    margin = margin_fn(degradation)
    h_eff = h_nominal - margin

    return ConstraintResult(
        h_nominal=h_nominal,
        h_effective=h_eff,
        margin=margin,
        w_t=float(w_t),
        satisfied=h_eff >= 0,
    )


@dataclass(frozen=True)
class ConstraintSetResult:
    """Evaluation of multiple reliability-conditioned constraints."""

    individual: tuple[ConstraintResult, ...]
    all_satisfied: bool
    min_effective_margin: float


def evaluate_constraint_set(
    h_nominals: np.ndarray,
    w_t: float,
    margin_fn: MarginFn | None = None,
    k: float = 1.0,
) -> ConstraintSetResult:
    """Evaluate a set of reliability-conditioned constraints."""
    results = tuple(evaluate_constraint(float(h), w_t, margin_fn, k) for h in h_nominals)
    return ConstraintSetResult(
        individual=results,
        all_satisfied=all(r.satisfied for r in results),
        min_effective_margin=min(r.h_effective for r in results),
    )


def constraint_tightening_curve(
    w_values: np.ndarray,
    h_nominal: float,
    margin_fn: MarginFn | None = None,
    k: float = 1.0,
) -> np.ndarray:
    """Return h_eff(w) for an array of reliability values.

    Useful for plotting how the effective safe-set shrinks with degradation.
    """
    if margin_fn is None:
        margin_fn = linear_margin(k)
    degradations = 1.0 - np.clip(w_values, 0.0, 1.0)
    margins = np.array([margin_fn(float(d)) for d in degradations])
    return h_nominal - margins
