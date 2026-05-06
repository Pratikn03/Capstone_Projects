"""Optimization package.

Contains dispatch optimizers, baselines, and impact evaluation helpers.
"""

from .baselines import (
    greedy_price_dispatch as greedy_price_dispatch,
)
from .baselines import (
    grid_only_dispatch as grid_only_dispatch,
)
from .baselines import (
    naive_battery_dispatch as naive_battery_dispatch,
)
from .baselines import (
    peak_shaving_dispatch as peak_shaving_dispatch,
)
from .impact import impact_summary as impact_summary
from .lp_dispatch import optimize_dispatch as optimize_dispatch

__all__ = [
    "greedy_price_dispatch",
    "grid_only_dispatch",
    "impact_summary",
    "naive_battery_dispatch",
    "optimize_dispatch",
    "peak_shaving_dispatch",
]
