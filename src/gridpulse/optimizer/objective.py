"""Optimization: objective."""
from __future__ import annotations

def compute_objective(cost_usd: float, carbon_kg: float, cost_weight: float = 1.0, carbon_weight: float = 0.5) -> float:
    return cost_weight * cost_usd + carbon_weight * carbon_kg
