"""Optimization package.

Contains dispatch optimizers, baselines, and impact evaluation helpers.
"""
from .lp_dispatch import optimize_dispatch as optimize_dispatch
from .baselines import (
    grid_only_dispatch as grid_only_dispatch,
    naive_battery_dispatch as naive_battery_dispatch,
    peak_shaving_dispatch as peak_shaving_dispatch,
    greedy_price_dispatch as greedy_price_dispatch,
)
from .impact import impact_summary as impact_summary

__all__ = [
    "optimize_dispatch",
    "grid_only_dispatch",
    "naive_battery_dispatch",
    "peak_shaving_dispatch",
    "greedy_price_dispatch",
    "impact_summary",
]
