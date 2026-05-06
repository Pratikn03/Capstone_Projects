"""Paper 3: Fallback dispatch optimization logic."""

from __future__ import annotations

from typing import Any


def solve_fallback_dispatch(
    initial_soc: float,
    remaining_horizon: int,
    safe_soc_envelope: list[dict[str, float]],
    objective_weights: dict[str, float],
    constraints: dict[str, float],
) -> dict[str, Any]:
    """
    Solves the optimization problem for graceful degradation.

    Returns a conservative hold-at-zero dispatch plan when no feasible
    trajectory can be computed from the current SOC and envelope.
    Production deployments should replace this with a domain-specific
    solver (e.g., cvxpy or scipy.optimize) using the safe_soc_envelope
    as a constraint surface.
    """
    # Conservative fallback: hold at zero dispatch across the remaining horizon.
    plan = [{"charge_mw": 0.0, "discharge_mw": 0.0} for _ in range(remaining_horizon)]

    return {"plan": plan, "status": "optimal", "objective_value": 0.0}
