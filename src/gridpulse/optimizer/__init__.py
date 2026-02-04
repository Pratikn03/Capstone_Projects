"""Optimization:   init  ."""
from .lp_dispatch import optimize_dispatch
from .baselines import (
    grid_only_dispatch,
    naive_battery_dispatch,
    peak_shaving_dispatch,
    greedy_price_dispatch,
)
from .impact import impact_summary
# Key: formulate dispatch objective/constraints and compute plans
