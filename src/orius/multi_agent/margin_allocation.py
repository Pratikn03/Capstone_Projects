"""Margin allocation for shared feeder capacity.

Allocates the feeder capacity among N agents so that local certificates
can be composed safely. Supports proportional, equal, and demand-based schemes.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def allocate_margins(
    feeder_capacity_mw: float,
    n_agents: int,
    *,
    scheme: str = "equal",
    demands: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
) -> list[float]:
    """
    Allocate feeder capacity margin to each agent.

    Parameters
    ----------
    feeder_capacity_mw : float
        Total shared capacity.
    n_agents : int
        Number of agents.
    scheme : str
        "equal" | "proportional" | "demand"
    demands : sequence or None
        Per-agent demand (for scheme="demand").
    weights : sequence or None
        Per-agent weights (for scheme="proportional").

    Returns
    -------
    list[float]
        Per-agent allocated margin in MW.
    """
    if scheme == "equal":
        margin = feeder_capacity_mw / max(n_agents, 1)
        return [margin] * n_agents

    if scheme == "proportional" and weights is not None:
        w = list(weights)
        total = sum(w)
        if total <= 0:
            return [feeder_capacity_mw / n_agents] * n_agents
        return [feeder_capacity_mw * (wi / total) for wi in w]

    if scheme == "demand" and demands is not None:
        d = list(demands)
        total = sum(d)
        if total <= 0:
            return [feeder_capacity_mw / n_agents] * n_agents
        return [feeder_capacity_mw * (di / total) for di in d]

    return [feeder_capacity_mw / n_agents] * n_agents


def allocate_margins_fairness(
    allocations: Sequence[float],
    demands: Sequence[float],
) -> float:
    """Fairness metric: 1 - Gini coefficient of (allocation/demand) ratios."""
    if not allocations or not demands or len(allocations) != len(demands):
        return 1.0
    ratios = [
        a / max(d, 1e-9) for a, d in zip(allocations, demands)
    ]
    n = len(ratios)
    sorted_r = sorted(ratios)
    gini = 0.0
    for i, r in enumerate(sorted_r):
        gini += (2 * (i + 1) - n - 1) * r
    gini /= (n * sum(sorted_r) + 1e-9)
    return max(0.0, 1.0 - gini)
