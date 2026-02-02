"""Optimization: impact."""
from __future__ import annotations

from typing import Dict, Any


def impact_summary(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    # Key: formulate dispatch objective/constraints and compute plans
    base_cost = float(baseline.get("expected_cost_usd", 0.0))
    opt_cost = float(optimized.get("expected_cost_usd", 0.0))
    base_carbon = float(baseline.get("carbon_kg", 0.0))
    opt_carbon = float(optimized.get("carbon_kg", 0.0))

    cost_savings = base_cost - opt_cost
    carbon_reduction = base_carbon - opt_carbon

    cost_pct = (cost_savings / base_cost * 100.0) if base_cost > 0 else None
    carbon_pct = (carbon_reduction / base_carbon * 100.0) if base_carbon > 0 else None

    return {
        "baseline_cost_usd": base_cost,
        "optimized_cost_usd": opt_cost,
        "cost_savings_usd": cost_savings,
        "cost_savings_pct": cost_pct,
        "baseline_carbon_kg": base_carbon,
        "optimized_carbon_kg": opt_carbon,
        "carbon_reduction_kg": carbon_reduction,
        "carbon_reduction_pct": carbon_pct,
    }
