"""Linear-programming dispatch optimizer using scipy.optimize.linprog."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.optimize import linprog


def _as_array(x) -> np.ndarray:
    # Key: formulate dispatch objective/constraints and compute plans
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
    else:
        arr = np.asarray([x], dtype=float)
    return arr


def optimize_dispatch(
    forecast_load,
    forecast_renewables,
    config: dict,
    forecast_price=None,
    forecast_carbon_kg=None,
) -> Dict[str, Any]:
    load = _as_array(forecast_load)
    ren = _as_array(forecast_renewables)
    # Align renewable series to load horizon (supports scalar inputs).
    if ren.size == 1 and load.size > 1:
        ren = np.full_like(load, float(ren[0]))
    if load.shape != ren.shape:
        raise ValueError("forecast_load and forecast_renewables must have the same length")

    H = len(load)

    cfg = config or {}
    battery = cfg.get("battery", {})
    grid = cfg.get("grid", {})
    penalties = cfg.get("penalties", {})
    objective = cfg.get("objective", {})
    carbon_cfg = cfg.get("carbon", {})

    capacity = float(battery.get("capacity_mwh", 10.0))
    max_power = float(battery.get("max_power_mw", battery.get("max_charge_mw", 2.0)))
    max_discharge = float(battery.get("max_discharge_mw", max_power))
    max_charge = float(battery.get("max_charge_mw", max_power))
    efficiency = float(battery.get("efficiency", 0.95))
    min_soc = float(battery.get("min_soc_mwh", 0.0))
    soc0 = float(battery.get("initial_soc_mwh", capacity / 2))

    max_import = float(grid.get("max_import_mw", grid.get("max_draw_mw", 50.0)))
    
    # Use time-varying price if provided; otherwise fallback to config constant.
    if forecast_price is not None:
        price = _as_array(forecast_price)
        if price.size == 1 and H > 1:
            price = np.full(H, float(price[0]))
    else:
        price = float(grid.get("price_per_mwh", grid.get("price_usd_per_mwh", 50.0)))
        price = np.full(H, price)

    carbon_cost = float(grid.get("carbon_cost_per_mwh", 0.0))
    carbon_kg = float(grid.get("carbon_kg_per_mwh", 0.0))
    # Allow time-varying carbon intensity to drive carbon-aware dispatch.
    if forecast_carbon_kg is not None:
        carbon_series = _as_array(forecast_carbon_kg)
        if carbon_series.size == 1 and H > 1:
            carbon_series = np.full(H, float(carbon_series[0]))
    else:
        carbon_series = np.full(H, carbon_kg)

    curtail_pen = float(penalties.get("curtailment_per_mw", 500.0))
    unmet_pen = float(penalties.get("unmet_load_per_mw", 10000.0))
    peak_pen = float(penalties.get("peak_per_mw", penalties.get("peak_penalty_per_mw", 0.0)))

    cost_weight = float(objective.get("cost_weight", 1.0))
    carbon_weight = float(objective.get("carbon_weight", 0.0))

    # Decision vars order: grid, charge, discharge, curtail, unmet, soc, peak
    n = H
    idx_grid = slice(0, n)
    idx_charge = slice(n, 2 * n)
    idx_discharge = slice(2 * n, 3 * n)
    idx_curtail = slice(3 * n, 4 * n)
    idx_unmet = slice(4 * n, 5 * n)
    idx_soc = slice(5 * n, 6 * n)
    idx_peak = 6 * n
    n_vars = 6 * n + 1

    c = np.zeros(n_vars)
    # Convert carbon cost per MWh into $/kg and scale by time-varying carbon intensity.
    carbon_cost_per_kg = (carbon_cost / carbon_kg) if carbon_kg > 0 else 0.0
    carbon_cost_series = carbon_series * carbon_cost_per_kg
    # Objective: weighted energy cost + weighted carbon cost + peak penalty.
    c[idx_grid] = cost_weight * price + carbon_weight * carbon_cost_series
    c[idx_curtail] = curtail_pen
    c[idx_unmet] = unmet_pen
    c[idx_peak] = peak_pen

    # Equality constraints: load balance and SOC dynamics
    A_eq = []
    b_eq = []

    # load balance: grid + discharge - curtail + unmet = load - renewables
    for t in range(H):
        row = np.zeros(n_vars)
        row[idx_grid.start + t] = 1.0
        row[idx_discharge.start + t] = 1.0
        row[idx_charge.start + t] = -1.0
        row[idx_curtail.start + t] = -1.0
        row[idx_unmet.start + t] = 1.0
        A_eq.append(row)
        b_eq.append(load[t] - ren[t])

    # SOC dynamics
    for t in range(H):
        row = np.zeros(n_vars)
        row[idx_soc.start + t] = 1.0
        if t == 0:
            # Initial SOC sets the starting state; charge/discharge affect SOC immediately.
            row[idx_charge.start + t] = -efficiency
            row[idx_discharge.start + t] = 1.0 / efficiency
            A_eq.append(row)
            b_eq.append(soc0)
        else:
            # SOC[t] = SOC[t-1] + charge*eff - discharge/eff
            row[idx_soc.start + t - 1] = -1.0
            row[idx_charge.start + t] = -efficiency
            row[idx_discharge.start + t] = 1.0 / efficiency
            A_eq.append(row)
            b_eq.append(0.0)

    A_eq = np.vstack(A_eq)
    b_eq = np.asarray(b_eq)

    # Inequality constraints: grid_t <= peak
    A_ub = []
    b_ub = []
    for t in range(H):
        row = np.zeros(n_vars)
        row[idx_grid.start + t] = 1.0
        row[idx_peak] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Optional carbon budget constraint: sum(grid_t * carbon_t) <= budget_kg
    budget_kg = carbon_cfg.get("budget_kg")
    budget_pct = carbon_cfg.get("budget_reduction_pct")
    if budget_pct is None:
        budget_pct = carbon_cfg.get("budget_pct")
    if budget_pct is not None:
        try:
            budget_pct = float(budget_pct)
        except (TypeError, ValueError):
            budget_pct = None
    if budget_kg is not None:
        try:
            budget_kg = float(budget_kg)
        except (TypeError, ValueError):
            budget_kg = None
    if budget_kg is None and budget_pct is not None:
        baseline_grid = np.clip(load - ren, 0.0, max_import)
        baseline_carbon = float(np.sum(baseline_grid * carbon_series))
        budget_kg = baseline_carbon * (1.0 - budget_pct)
    if budget_kg is not None:
        row = np.zeros(n_vars)
        row[idx_grid] = carbon_series
        A_ub.append(row)
        b_ub.append(budget_kg)
    A_ub = np.vstack(A_ub)
    b_ub = np.asarray(b_ub)

    # Bounds
    bounds = []
    for t in range(H):  # grid
        bounds.append((0.0, max_import))
    for t in range(H):  # charge
        bounds.append((0.0, max_charge))
    for t in range(H):  # discharge
        bounds.append((0.0, max_discharge))
    for t in range(H):  # curtailment
        bounds.append((0.0, ren[t]))
    for t in range(H):  # unmet
        bounds.append((0.0, None))
    for t in range(H):  # soc
        bounds.append((min_soc, capacity))
    # Peak bound ties max import; acts as optimization variable.
    bounds.append((0.0, max_import))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        # Fallback: if LP fails, serve a safe grid-only plan.
        grid_plan = np.maximum(0.0, load - ren)
        curtail = np.maximum(0.0, ren - load)
        return {
            "grid_mw": grid_plan.tolist(),
            "battery_charge_mw": [0.0] * H,
            "battery_discharge_mw": [0.0] * H,
            "renewables_used_mw": np.minimum(ren, load).tolist(),
            "curtailment_mw": curtail.tolist(),
            "unmet_load_mw": [0.0] * H,
            "soc_mwh": [soc0] * H,
            "peak_mw": float(np.max(grid_plan)) if len(grid_plan) else None,
            "expected_cost_usd": float(np.sum(grid_plan * price) + np.sum(curtail) * curtail_pen),
            "carbon_kg": float(np.sum(grid_plan * carbon_series)),
            "carbon_cost_usd": float(np.sum(grid_plan * carbon_cost_series)),
            "carbon_budget_kg": float(budget_kg) if budget_kg is not None else None,
            "note": f"linprog failed: {res.message}",
        }

    x = res.x
    grid_plan = x[idx_grid]
    charge = x[idx_charge]
    discharge = x[idx_discharge]
    curtail = x[idx_curtail]
    unmet = x[idx_unmet]
    soc = x[idx_soc]
    peak = float(x[idx_peak])
    renewables_used = ren - curtail

    # Final objective terms for reporting.
    expected_cost = float(np.sum(grid_plan * price) + np.sum(curtail) * curtail_pen + np.sum(unmet) * unmet_pen)
    carbon = float(np.sum(grid_plan * carbon_series))
    carbon_cost = float(np.sum(grid_plan * carbon_cost_series))

    return {
        "grid_mw": grid_plan.tolist(),
        "battery_charge_mw": charge.tolist(),
        "battery_discharge_mw": discharge.tolist(),
        "renewables_used_mw": renewables_used.tolist(),
        "curtailment_mw": curtail.tolist(),
        "unmet_load_mw": unmet.tolist(),
        "soc_mwh": soc.tolist(),
        "peak_mw": peak,
        "expected_cost_usd": expected_cost,
        "carbon_kg": carbon,
        "carbon_cost_usd": carbon_cost,
        "carbon_budget_kg": float(budget_kg) if budget_kg is not None else None,
        "status": res.message,
    }
