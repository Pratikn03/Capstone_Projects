from __future__ import annotations

from typing import Dict, Any
import numpy as np


def _as_array(x) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)
    return np.asarray([x], dtype=float)


def _load_cfg(cfg: dict) -> dict:
    cfg = cfg or {}
    battery = cfg.get("battery", {})
    grid = cfg.get("grid", {})
    penalties = cfg.get("penalties", {})
    objective = cfg.get("objective", {})

    return {
        "battery": {
            "capacity": float(battery.get("capacity_mwh", 10.0)),
            "max_charge": float(battery.get("max_charge_mw", battery.get("max_power_mw", 2.0))),
            "max_discharge": float(battery.get("max_discharge_mw", battery.get("max_power_mw", 2.0))),
            "eff": float(battery.get("efficiency", 0.95)),
            "min_soc": float(battery.get("min_soc_mwh", 0.0)),
            "soc0": float(battery.get("initial_soc_mwh", battery.get("capacity_mwh", 10.0) / 2)),
        },
        "grid": {
            "max_import": float(grid.get("max_import_mw", grid.get("max_draw_mw", 50.0))),
            "price": float(grid.get("price_per_mwh", grid.get("price_usd_per_mwh", 50.0))),
            "carbon_cost": float(grid.get("carbon_cost_per_mwh", 0.0)),
            "carbon_kg": float(grid.get("carbon_kg_per_mwh", 0.0)),
        },
        "penalties": {
            "curtail": float(penalties.get("curtailment_per_mw", 500.0)),
            "unmet": float(penalties.get("unmet_load_per_mw", 10000.0)),
        },
        "objective": {
            "cost_weight": float(objective.get("cost_weight", 1.0)),
            "carbon_weight": float(objective.get("carbon_weight", 0.0)),
        },
    }


def grid_only_dispatch(forecast_load, forecast_renewables, cfg: dict, price_series=None) -> Dict[str, Any]:
    load = _as_array(forecast_load)
    ren = _as_array(forecast_renewables)
    if ren.size == 1 and load.size > 1:
        ren = np.full_like(load, float(ren[0]))
    if load.shape != ren.shape:
        raise ValueError("forecast_load and forecast_renewables must have the same length")

    cfg = _load_cfg(cfg)
    H = len(load)

    deficit = load - ren
    grid = np.clip(deficit, 0.0, cfg["grid"]["max_import"])
    unmet = np.clip(deficit - grid, 0.0, None)
    curtail = np.clip(ren - load, 0.0, None)

    if price_series is not None:
        price = _as_array(price_series)
        if price.size == 1 and H > 1:
            price = np.full(H, float(price[0]))
        expected_cost = float(np.sum(grid * price) + np.sum(curtail) * cfg["penalties"]["curtail"] + np.sum(unmet) * cfg["penalties"]["unmet"])
    else:
        expected_cost = float(np.sum(grid) * cfg["grid"]["price"] + np.sum(curtail) * cfg["penalties"]["curtail"] + np.sum(unmet) * cfg["penalties"]["unmet"])
    carbon_kg = float(np.sum(grid) * cfg["grid"]["carbon_kg"])
    carbon_cost = float(np.sum(grid) * cfg["grid"]["carbon_cost"])

    return {
        "grid_mw": grid.tolist(),
        "battery_charge_mw": [0.0] * H,
        "battery_discharge_mw": [0.0] * H,
        "renewables_used_mw": (ren - curtail).tolist(),
        "curtailment_mw": curtail.tolist(),
        "unmet_load_mw": unmet.tolist(),
        "soc_mwh": [cfg["battery"]["soc0"]] * H,
        "expected_cost_usd": expected_cost,
        "carbon_kg": carbon_kg,
        "carbon_cost_usd": carbon_cost,
        "policy": "grid_only",
    }


def naive_battery_dispatch(forecast_load, forecast_renewables, cfg: dict, price_series=None) -> Dict[str, Any]:
    """Simple policy: charge at night (00–05), discharge at evening peak (17–21)."""
    load = _as_array(forecast_load)
    ren = _as_array(forecast_renewables)
    if ren.size == 1 and load.size > 1:
        ren = np.full_like(load, float(ren[0]))
    if load.shape != ren.shape:
        raise ValueError("forecast_load and forecast_renewables must have the same length")

    cfg = _load_cfg(cfg)
    H = len(load)
    soc = cfg["battery"]["soc0"]

    grid = np.zeros(H)
    charge = np.zeros(H)
    discharge = np.zeros(H)
    curtail = np.zeros(H)
    unmet = np.zeros(H)
    soc_series = []

    for t in range(H):
        hour = t % 24
        net = load[t] - ren[t]

        # charge window
        if hour in {0, 1, 2, 3, 4, 5}:
            c = min(cfg["battery"]["max_charge"], cfg["battery"]["capacity"] - soc)
            charge[t] = c
            soc += c * cfg["battery"]["eff"]
        # discharge window
        elif hour in {17, 18, 19, 20, 21}:
            d = min(cfg["battery"]["max_discharge"], soc - cfg["battery"]["min_soc"])
            discharge[t] = d
            soc -= d / cfg["battery"]["eff"]

        # recompute net with battery
        net = load[t] - ren[t] - discharge[t] + charge[t]
        if net > 0:
            grid[t] = min(net, cfg["grid"]["max_import"])
            unmet[t] = max(0.0, net - grid[t])
        else:
            curtail[t] = max(0.0, -net)

        soc = min(max(soc, cfg["battery"]["min_soc"]), cfg["battery"]["capacity"])
        soc_series.append(soc)

    if price_series is not None:
        price = _as_array(price_series)
        if price.size == 1 and H > 1:
            price = np.full(H, float(price[0]))
        expected_cost = float(np.sum(grid * price) + np.sum(curtail) * cfg["penalties"]["curtail"] + np.sum(unmet) * cfg["penalties"]["unmet"])
    else:
        expected_cost = float(np.sum(grid) * cfg["grid"]["price"] + np.sum(curtail) * cfg["penalties"]["curtail"] + np.sum(unmet) * cfg["penalties"]["unmet"])
    carbon_kg = float(np.sum(grid) * cfg["grid"]["carbon_kg"])
    carbon_cost = float(np.sum(grid) * cfg["grid"]["carbon_cost"])

    return {
        "grid_mw": grid.tolist(),
        "battery_charge_mw": charge.tolist(),
        "battery_discharge_mw": discharge.tolist(),
        "renewables_used_mw": (ren - curtail).tolist(),
        "curtailment_mw": curtail.tolist(),
        "unmet_load_mw": unmet.tolist(),
        "soc_mwh": soc_series,
        "expected_cost_usd": expected_cost,
        "carbon_kg": carbon_kg,
        "carbon_cost_usd": carbon_cost,
        "policy": "naive_battery",
    }
