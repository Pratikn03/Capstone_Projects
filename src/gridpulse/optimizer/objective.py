"""
Optimizer: Scalar Objective Function for Multi-Criteria Optimization.

This module provides the objective function that combines multiple optimization
criteria (cost and carbon) into a single scalar value. This is necessary because
linear programming requires a single objective to minimize.

Objective Function:
    objective = cost_weight * cost_usd + carbon_weight * carbon_kg

Weight Interpretation:
    - cost_weight=1, carbon_weight=0: Pure cost minimization (profit-focused)
    - cost_weight=0, carbon_weight=1: Pure carbon minimization (green-focused)
    - cost_weight=1, carbon_weight=0.5: Balanced approach (typical default)

Tradeoff Analysis:
    The Pareto frontier of cost vs carbon can be traced by varying weights.
    Higher carbon_weight leads to "greener" but potentially more expensive
    dispatch schedules.

Usage:
    >>> from gridpulse.optimizer.objective import compute_objective
    >>> obj = compute_objective(cost_usd=100, carbon_kg=50, carbon_weight=0.5)
    >>> print(f"Objective: {obj}")  # 100 + 0.5*50 = 125

See Also:
    - lp_dispatch.py: Uses this objective in the linear program
    - configs/optimization.yaml: Default weights configuration
"""
from __future__ import annotations


def compute_objective(
    cost_usd: float, 
    carbon_kg: float, 
    cost_weight: float = 1.0, 
    carbon_weight: float = 0.5
) -> float:
    """
    Combine cost and carbon metrics into a single optimization objective.
    
    This weighted sum approach is a standard method for multi-objective
    optimization when relative importance can be expressed numerically.
    
    Args:
        cost_usd: Total electricity cost in USD (or EUR, currency-agnostic)
        carbon_kg: Total carbon emissions in kilograms CO2
        cost_weight: Weight for cost term (default: 1.0)
        carbon_weight: Weight for carbon term (default: 0.5)
        
    Returns:
        Scalar objective value to be minimized
        
    Example:
        >>> compute_objective(100, 50, cost_weight=1.0, carbon_weight=0.5)
        125.0  # = 1.0*100 + 0.5*50
    """
    return cost_weight * cost_usd + carbon_weight * carbon_kg
