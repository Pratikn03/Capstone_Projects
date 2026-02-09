"""
Regret analysis for dispatch optimization.

Computes regret as the cost difference between a dispatch policy
and the oracle (perfect foresight) policy.
"""
from __future__ import annotations

import numpy as np
from typing import Any


def compute_regret(
    actual_cost: float,
    oracle_cost: float,
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute regret metrics.
    
    Args:
        actual_cost: Cost achieved by policy with forecasts
        oracle_cost: Cost with perfect foresight (oracle)
        normalize: Whether to compute percentage regret
    
    Returns:
        Dictionary with:
        - absolute_regret: Cost difference (EUR)
        - relative_regret: Percentage increase over oracle
        - oracle_cost: Oracle cost for reference
        - actual_cost: Actual cost for reference
    """
    regret_abs = actual_cost - oracle_cost
    regret_rel = 100 * regret_abs / max(oracle_cost, 1e-6)
    
    return {
        "absolute_regret": float(regret_abs),
        "relative_regret": float(regret_rel),
        "oracle_cost": float(oracle_cost),
        "actual_cost": float(actual_cost),
    }


def compute_multi_scenario_regret(
    scenario_costs: list[float],
    oracle_costs: list[float],
) -> dict[str, Any]:
    """
    Compute regret statistics across multiple scenarios.
    
    Args:
        scenario_costs: Costs for each scenario (n_scenarios,)
        oracle_costs: Oracle costs for each scenario
    
    Returns:
        Dictionary with aggregated regret metrics
    """
    regrets = []
    for actual, oracle in zip(scenario_costs, oracle_costs):
        r = compute_regret(actual, oracle)
        regrets.append(r["relative_regret"])
    
    regrets = np.array(regrets)
    
    return {
        "mean_regret": float(np.mean(regrets)),
        "std_regret": float(np.std(regrets)),
        "median_regret": float(np.median(regrets)),
        "min_regret": float(np.min(regrets)),
        "max_regret": float(np.max(regrets)),
        "n_scenarios": len(regrets),
    }
