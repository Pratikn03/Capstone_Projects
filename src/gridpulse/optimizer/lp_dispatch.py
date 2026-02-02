"""Linear-programming dispatch optimizer using scipy.optimize.linprog."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.optimize import linprog


def _as_array(x) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
    else:
        arr = np.asarray([x], dtype=float)
    return arr


def optimize_dispatch(forecast_load, forecast_renewables, config: dict, forecast_price=None) -> Dict[str, Any]:
    load = _as_array(forecast_load)
    ren = _as_array(forecast_renewables)
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

    capacity = float(battery.get("capacity_mwh", 10.0))
    max_power = float(battery.get("max_power_mw", battery.get("max_charge_mw", 2.0)))
    max_discharge = float(battery.get("max_discharge_mw", max_power))
    max_charge = float(battery.get("max_charge_mw", max_power))
    efficiency = float(battery.get("efficiency", 0.95))
    min_soc = float(battery.get("min_soc_mwh", 0.0))
    soc0 = float(battery.get("initial_soc_mwh", capacity / 2))

    max_import = float(grid.get("max_import_mw", grid.get("max_draw_mw", 50.0)))
    
    if forecast_price is not None:
        price = _as_array(forecast_price)
        if price.size == 1 and H > 1:
            price = np.full(H, float(price[0]))
    else:
        price = float(grid.get("price_per_mwh", grid.get("price_usd_per_mwh", 50.0)))
        price = np.full(H, price)

    carbon_cost = float(grid.get("carbon_cost_per_mwh", 0.0))
    carbon_kg = float(grid.get("carbon_kg_per_mwh", 0.0))

    curtail_pen = float(penalties.get("curtailment_per_mw", 500.0))
    unmet_pen = float(penalties.get("unmet_load_per_mw", 10000.0))

    cost_weight = float(objective.get("cost_weight", 1.0))
    carbon_weight = float(objective.get("carbon_weight", 0.0))

    # Decision vars order: grid, charge, discharge, curtail, unmet, soc
    n = H
    idx_grid = slice(0, n)
    idx_charge = slice(n, 2 * n)
    idx_discharge = slice(2 * n, 3 * n)
    idx_curtail = slice(3 * n, 4 * n)
    idx_unmet = slice(4 * n, 5 * n)
    idx_soc = slice(5 * n, 6 * n)
    n_vars = 6 * n

    c = np.zeros(n_vars)
    c[idx_grid] = cost_weight * price + carbon_weight * carbon_cost
    c[idx_curtail] = curtail_pen
    c[idx_unmet] = unmet_pen

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
            row[idx_charge.start + t] = -efficiency
            row[idx_discharge.start + t] = 1.0 / efficiency
            A_eq.append(row)
            b_eq.append(soc0)
        else:
            row[idx_soc.start + t - 1] = -1.0
            row[idx_charge.start + t] = -efficiency
            row[idx_discharge.start + t] = 1.0 / efficiency
            A_eq.append(row)
            b_eq.append(0.0)

    A_eq = np.vstack(A_eq)
    b_eq = np.asarray(b_eq)

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

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        grid_plan = np.maximum(0.0, load - ren)
        return {
            "grid_mw": grid_plan.tolist(),
            "battery_charge_mw": [0.0] * H,
            "battery_discharge_mw": [0.0] * H,
            "renewables_used_mw": np.minimum(ren, load).tolist(),
            "curtailment_mw": np.maximum(0.0, ren - load).tolist(),
            "unmet_load_mw": [0.0] * H,
            "soc_mwh": [soc0] * H,
            "expected_cost_usd": float(np.sum(grid_plan) * price),
            "carbon_kg": float(np.sum(grid_plan) * carbon_kg),
            "note": f"linprog failed: {res.message}",
        }

    x = res.x
    grid_plan = x[idx_grid]
    charge = x[idx_charge]
    discharge = x[idx_discharge]
    curtail = x[idx_curtail]
    unmet = x[idx_unmet]
    soc = x[idx_soc]
    renewables_used = ren - curtail

    expected_cost = float(np.sum(grid_plan * price) + np.sum(curtail) * curtail_pen + np.sum(unmet) * unmet_pen)
    carbon = float(np.sum(grid_plan) * carbon_kg)
    carbon_cost = float(np.sum(grid_plan) * carbon_cost)

    return {
        "grid_mw": grid_plan.tolist(),
        "battery_charge_mw": charge.tolist(),
        "battery_discharge_mw": discharge.tolist(),
        "renewables_used_mw": renewables_used.tolist(),
        "curtailment_mw": curtail.tolist(),
        "unmet_load_mw": unmet.tolist(),
        "soc_mwh": soc.tolist(),
        "expected_cost_usd": expected_cost,
        "carbon_kg": carbon,
        "carbon_cost_usd": carbon_cost,
        "status": res.message,
    }
