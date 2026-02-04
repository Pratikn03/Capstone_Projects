"""Optimization baselines for dispatch comparisons."""
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def _as_array(x) -> np.ndarray:
    """Convert scalars/lists to a float NumPy array."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)
    return np.asarray([x], dtype=float)


def _load_cfg(cfg: dict) -> dict:
    """Normalize config values and apply defaults."""
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


def _compute_costs(
    grid: np.ndarray,
    curtail: np.ndarray,
    unmet: np.ndarray,
    cfg: dict,
    price_series=None,
    carbon_series=None,
) -> tuple[float, float, float]:
    """Compute cost, carbon mass, and carbon cost for a dispatch plan."""
    H = len(grid)
    if price_series is not None:
        price = _as_array(price_series)
        if price.size == 1 and H > 1:
            price = np.full(H, float(price[0]))
        expected_cost = float(
            np.sum(grid * price)
            + np.sum(curtail) * cfg["penalties"]["curtail"]
            + np.sum(unmet) * cfg["penalties"]["unmet"]
        )
    else:
        expected_cost = float(
            np.sum(grid) * cfg["grid"]["price"]
            + np.sum(curtail) * cfg["penalties"]["curtail"]
            + np.sum(unmet) * cfg["penalties"]["unmet"]
        )

    if carbon_series is not None:
        carbon = _as_array(carbon_series)
        if carbon.size == 1 and H > 1:
            carbon = np.full(H, float(carbon[0]))
        carbon_kg = float(np.sum(grid * carbon))
        cost_per_kg = (cfg["grid"]["carbon_cost"] / cfg["grid"]["carbon_kg"]) if cfg["grid"]["carbon_kg"] > 0 else 0.0
        carbon_cost = float(carbon_kg * cost_per_kg)
    else:
        carbon_kg = float(np.sum(grid) * cfg["grid"]["carbon_kg"])
        carbon_cost = float(np.sum(grid) * cfg["grid"]["carbon_cost"])

    return expected_cost, carbon_kg, carbon_cost


def grid_only_dispatch(
    forecast_load,
    forecast_renewables,
    cfg: dict,
    price_series=None,
    carbon_series=None,
) -> Dict[str, Any]:
    """Baseline: meet net load using grid only (no battery)."""
    load = _as_array(forecast_load)
    ren = _as_array(forecast_renewables)
    if ren.size == 1 and load.size > 1:
        ren = np.full_like(load, float(ren[0]))
    if load.shape != ren.shape:
        raise ValueError("forecast_load and forecast_renewables must have the same length")

    cfg = _load_cfg(cfg)
    H = len(load)

    # Net deficit drives grid import; surplus becomes curtailment.
    deficit = load - ren
    grid = np.clip(deficit, 0.0, cfg["grid"]["max_import"])
    unmet = np.clip(deficit - grid, 0.0, None)
    curtail = np.clip(ren - load, 0.0, None)

    expected_cost, carbon_kg, carbon_cost = _compute_costs(
        grid,
        curtail,
        unmet,
        cfg,
        price_series=price_series,
        carbon_series=carbon_series,
    )

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


def naive_battery_dispatch(
    forecast_load,
    forecast_renewables,
    cfg: dict,
    price_series=None,
    carbon_series=None,
) -> Dict[str, Any]:
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

        # Recompute net with battery applied.
        net = load[t] - ren[t] - discharge[t] + charge[t]
        if net > 0:
            grid[t] = min(net, cfg["grid"]["max_import"])
            unmet[t] = max(0.0, net - grid[t])
        else:
            curtail[t] = max(0.0, -net)

        soc = min(max(soc, cfg["battery"]["min_soc"]), cfg["battery"]["capacity"])
        soc_series.append(soc)

    expected_cost, carbon_kg, carbon_cost = _compute_costs(
        grid,
        curtail,
        unmet,
        cfg,
        price_series=price_series,
        carbon_series=carbon_series,
    )

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


def peak_shaving_dispatch(
    forecast_load,
    forecast_renewables,
    cfg: dict,
    price_series=None,
    carbon_series=None,
    target_quantile: float = 0.8,
) -> Dict[str, Any]:
    """Heuristic peak‑shaving policy: cap net load near a target quantile."""
    load = _as_array(forecast_load)
    ren = _as_array(forecast_renewables)
    if ren.size == 1 and load.size > 1:
        ren = np.full_like(load, float(ren[0]))
    if load.shape != ren.shape:
        raise ValueError("forecast_load and forecast_renewables must have the same length")

    cfg = _load_cfg(cfg)
    H = len(load)
    net = load - ren
    # Use a robust cap based on the net‑load distribution.
    target_cap = float(np.quantile(net, target_quantile)) if H else 0.0
    target_cap = min(target_cap, cfg["grid"]["max_import"])

    soc = cfg["battery"]["soc0"]
    grid = np.zeros(H)
    charge = np.zeros(H)
    discharge = np.zeros(H)
    curtail = np.zeros(H)
    unmet = np.zeros(H)
    soc_series = []

    for t in range(H):
        net_t = net[t]
        # Discharge to keep net load under the cap.
        if net_t > target_cap:
            need = net_t - target_cap
            d = min(need, cfg["battery"]["max_discharge"], soc - cfg["battery"]["min_soc"])
            if d > 0:
                discharge[t] = d
                soc -= d / cfg["battery"]["eff"]
        # Charge when comfortably below cap.
        elif net_t < target_cap - cfg["battery"]["max_charge"] * 0.5:
            c = min(cfg["battery"]["max_charge"], cfg["battery"]["capacity"] - soc)
            if c > 0:
                charge[t] = c
                soc += c * cfg["battery"]["eff"]

        net_adj = net_t - discharge[t] + charge[t]
        if net_adj > 0:
            grid[t] = min(net_adj, cfg["grid"]["max_import"])
            unmet[t] = max(0.0, net_adj - grid[t])
        else:
            curtail[t] = max(0.0, -net_adj)

        soc = min(max(soc, cfg["battery"]["min_soc"]), cfg["battery"]["capacity"])
        soc_series.append(soc)

    expected_cost, carbon_kg, carbon_cost = _compute_costs(
        grid,
        curtail,
        unmet,
        cfg,
        price_series=price_series,
        carbon_series=carbon_series,
    )

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
        "policy": "peak_shaving",
        "target_cap_mw": target_cap,
    }


def greedy_price_dispatch(
    forecast_load,
    forecast_renewables,
    cfg: dict,
    price_series=None,
    carbon_series=None,
    low_quantile: float = 0.3,
    high_quantile: float = 0.7,
) -> Dict[str, Any]:
    """Greedy price‑based dispatch: charge on cheap hours, discharge on expensive hours."""
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

    if price_series is None:
        # Fall back to naive policy if no price signal is available.
        return naive_battery_dispatch(load, ren, cfg, price_series=price_series, carbon_series=carbon_series)

    price = _as_array(price_series)
    if price.size == 1 and H > 1:
        price = np.full(H, float(price[0]))
    low = float(np.quantile(price, low_quantile)) if H else float(price[0])
    high = float(np.quantile(price, high_quantile)) if H else float(price[0])

    for t in range(H):
        net_t = load[t] - ren[t]
        if price[t] <= low:
            c = min(cfg["battery"]["max_charge"], cfg["battery"]["capacity"] - soc)
            if c > 0:
                charge[t] = c
                soc += c * cfg["battery"]["eff"]
        elif price[t] >= high:
            d = min(cfg["battery"]["max_discharge"], soc - cfg["battery"]["min_soc"])
            if d > 0:
                discharge[t] = d
                soc -= d / cfg["battery"]["eff"]

        net_adj = net_t - discharge[t] + charge[t]
        if net_adj > 0:
            grid[t] = min(net_adj, cfg["grid"]["max_import"])
            unmet[t] = max(0.0, net_adj - grid[t])
        else:
            curtail[t] = max(0.0, -net_adj)

        soc = min(max(soc, cfg["battery"]["min_soc"]), cfg["battery"]["capacity"])
        soc_series.append(soc)

    expected_cost, carbon_kg, carbon_cost = _compute_costs(
        grid,
        curtail,
        unmet,
        cfg,
        price_series=price_series,
        carbon_series=carbon_series,
    )

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
        "policy": "price_greedy",
        "low_price": low,
        "high_price": high,
    }
