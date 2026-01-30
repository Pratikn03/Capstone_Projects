"""Linear-programming dispatch optimizer (placeholder).

Implement with:
- scipy.optimize.linprog OR pulp OR ortools
"""
from __future__ import annotations

def optimize_dispatch(forecast_load, forecast_renewables, config: dict):
    # TODO: implement LP with battery constraints and carbon/cost objective
    return {
        "renewables": float(forecast_renewables),
        "grid": float(max(0.0, forecast_load - forecast_renewables)),
        "battery": 0.0,
        "expected_cost_usd": None,
        "carbon_kg": None,
    }
