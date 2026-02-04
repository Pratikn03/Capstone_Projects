"""Optimization package.

Contains dispatch optimizers, baselines, and impact evaluation helpers.
"""
from .lp_dispatch import optimize_dispatch
from .baselines import (
    grid_only_dispatch,
    naive_battery_dispatch,
    peak_shaving_dispatch,
    greedy_price_dispatch,
)
from .impact import impact_summary
