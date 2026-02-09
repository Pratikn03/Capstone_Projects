"""
Optimization: Impact summary utilities.

This module compares baseline (rule-based) vs optimized (GridPulse) dispatch
plans to compute the economic and environmental benefits.

Key metrics computed:
- Cost savings (USD and percentage)
- Carbon reduction (kg CO₂ and percentage)
- Peak shaving (MW reduction)

These metrics are used in reports, dashboards, and the scientific paper.
"""
from __future__ import annotations

from typing import Dict, Any


def impact_summary(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    """Compare baseline vs optimized dispatch plans and compute impact metrics.
    
    Args:
        baseline: Results from rule-based dispatch (charge low/discharge high)
        optimized: Results from GridPulse MILP optimization
    
    Returns:
        Dictionary containing:
        - baseline_cost_usd: Total cost under baseline strategy
        - optimized_cost_usd: Total cost under optimized strategy
        - cost_savings_usd: Absolute cost reduction
        - cost_savings_pct: Percentage cost reduction
        - baseline_carbon_kg: Total carbon emissions under baseline
        - optimized_carbon_kg: Total carbon emissions under optimized
        - carbon_reduction_kg: Absolute carbon reduction
        - carbon_reduction_pct: Percentage carbon reduction
    """
    # Extract costs from both plans
    base_cost = float(baseline.get("expected_cost_usd", 0.0))
    opt_cost = float(optimized.get("expected_cost_usd", 0.0))
    
    # Extract carbon emissions from both plans
    base_carbon = float(baseline.get("carbon_kg", 0.0))
    opt_carbon = float(optimized.get("carbon_kg", 0.0))

    # Compute absolute savings
    cost_savings = base_cost - opt_cost
    carbon_reduction = base_carbon - opt_carbon

    # Compute percentage improvements (avoid division by zero)
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
