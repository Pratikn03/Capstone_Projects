"""
Risk-aware helpers for dispatch optimization inputs.

This module provides conservative forecast adjustment using prediction intervals.
When uncertainty bounds are available, the optimizer can use worst-case scenarios:
- Load: Use upper bound (plan for higher demand)
- Renewables: Use lower bound (plan for less generation)

This ensures the dispatch plan remains feasible even under adverse conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RiskConfig:
    """Configuration for risk-aware dispatch optimization.
    
    Attributes:
        enabled: Whether to apply risk adjustments
        mode: Risk mode ('worst_case_interval' or 'robust')
        load_bound: Which bound to use for load ('upper' or 'lower')
        renew_bound: Which bound to use for renewables ('upper' or 'lower')
        reserve_soc_mwh: Minimum SOC reserve to maintain (emergency buffer)
    """
    enabled: bool = True
    mode: str = "worst_case_interval"
    load_bound: str = "upper"  # Conservative: assume higher demand
    renew_bound: str = "lower"  # Conservative: assume less renewable generation
    reserve_soc_mwh: float = 0.0


def apply_interval_bounds(
    load_point: np.ndarray,
    renew_point: np.ndarray,
    load_lo: Optional[np.ndarray] = None,
    load_hi: Optional[np.ndarray] = None,
    renew_lo: Optional[np.ndarray] = None,
    renew_hi: Optional[np.ndarray] = None,
    cfg: Optional[RiskConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply interval bounds to forecasts for conservative planning.
    
    Args:
        load_point: Point forecast for load (MW)
        renew_point: Point forecast for renewables (MW)
        load_lo: Lower bound of load prediction interval
        load_hi: Upper bound of load prediction interval
        renew_lo: Lower bound of renewables prediction interval
        renew_hi: Upper bound of renewables prediction interval
        cfg: Risk configuration (defaults to conservative settings)
    
    Returns:
        Tuple of (adjusted_load, adjusted_renewables) arrays
    """
    cfg = cfg or RiskConfig()
    load_used = np.asarray(load_point, dtype=float).copy()
    renew_used = np.asarray(renew_point, dtype=float).copy()

    if not cfg.enabled:
        return load_used, renew_used

    # Apply load bound (typically upper for conservative planning)
    if cfg.load_bound == "upper" and load_hi is not None:
        load_used = np.asarray(load_hi, dtype=float)
    elif cfg.load_bound == "lower" and load_lo is not None:
        load_used = np.asarray(load_lo, dtype=float)

    # Apply renewable bound (typically lower for conservative planning)
    if cfg.renew_bound == "lower" and renew_lo is not None:
        renew_used = np.asarray(renew_lo, dtype=float)
    elif cfg.renew_bound == "upper" and renew_hi is not None:
        renew_used = np.asarray(renew_hi, dtype=float)

    return load_used, renew_used
