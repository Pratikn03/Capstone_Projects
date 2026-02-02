"""Optimization: objective."""
from __future__ import annotations

def compute_objective(cost_usd: float, carbon_kg: float, cost_weight: float = 1.0, carbon_weight: float = 0.5) -> float:
    # Key: formulate dispatch objective/constraints and compute plans
    return cost_weight * cost_usd + carbon_weight * carbon_kg
