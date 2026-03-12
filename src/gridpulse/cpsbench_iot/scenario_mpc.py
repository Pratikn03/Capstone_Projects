"""Receding-horizon scenario MPC baseline for CPSBench comparison.

Uses the existing robust / CVaR dispatch stack in a rolling-horizon loop
with sampled load scenarios.  This is the "strong control baseline" needed
for R1 reviewer credibility — it re-solves at every step rather than
committing to a single open-loop plan.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gridpulse.optimizer.robust_dispatch import (
    CVaRDispatchConfig,
    RobustDispatchConfig,
    optimize_cvar_dispatch,
    optimize_robust_dispatch,
)


def _battery_constraints(cfg: Mapping[str, Any]) -> dict[str, float]:
    battery = dict(cfg.get("battery", {}))
    capacity = float(battery.get("capacity_mwh", 100.0))
    max_power = float(battery.get("max_power_mw", 50.0))
    return {
        "capacity_mwh": capacity,
        "max_charge_mw": float(battery.get("max_charge_mw", max_power)),
        "max_discharge_mw": float(battery.get("max_discharge_mw", max_power)),
        "min_soc_mwh": float(battery.get("min_soc_mwh", 0.0)),
        "max_soc_mwh": float(battery.get("max_soc_mwh", capacity)),
        "initial_soc_mwh": float(battery.get("initial_soc_mwh", capacity * 0.5)),
        "efficiency": float(battery.get("efficiency", 0.95)),
        "charge_efficiency": float(battery.get("charge_efficiency", battery.get("efficiency", 0.95))),
        "discharge_efficiency": float(battery.get("discharge_efficiency", battery.get("efficiency", 0.95))),
        "degradation_cost_per_mwh": float(battery.get("degradation_cost_per_mwh", 10.0)),
        "max_grid_import_mw": float(cfg.get("grid", {}).get("max_import_mw", 500.0)),
        "time_step_hours": float(cfg.get("time_step_hours", 1.0)),
    }


def _sample_scenarios(
    forecast: np.ndarray,
    sigma: float,
    n_scenarios: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_scenarios, horizon) Gaussian perturbations of *forecast*."""
    horizon = len(forecast)
    eps = rng.normal(0.0, max(sigma, 1.0), size=(n_scenarios, horizon))
    return np.clip(forecast[np.newaxis, :] + eps, a_min=0.0, a_max=None)


def scenario_mpc_dispatch(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    load_true: np.ndarray,
    price: np.ndarray,
    optimization_cfg: Mapping[str, Any] | None = None,
    seed: int = 0,
    n_scenarios: int = 30,
    replan_interval: int = 1,
    planning_horizon: int | None = None,
    use_cvar: bool = False,
    cvar_beta: float = 0.90,
) -> dict[str, Any]:
    """Receding-horizon MPC that re-solves every *replan_interval* steps.

    At each decision point the controller:
      1. Samples *n_scenarios* load trajectories.
      2. Solves a robust or CVaR dispatch over the remaining window.
      3. Commits only the first *replan_interval* actions.
      4. Advances the SOC and repeats.

    This is strictly stronger than the existing open-loop
    ``scenario_robust_dispatch`` because it incorporates feedback.
    """
    load_fc = np.asarray(load_forecast, dtype=float).ravel()
    renewables = np.asarray(renewables_forecast, dtype=float).ravel()
    load_gt = np.asarray(load_true, dtype=float).ravel()
    prices = np.asarray(price, dtype=float).ravel()
    T = len(load_fc)

    cfg = dict(optimization_cfg or {})
    bc = _battery_constraints(cfg)
    sigma = float(np.std(load_gt - load_fc)) if T > 1 else 50.0
    rng = np.random.default_rng(seed)
    ph = planning_horizon or T

    # Output arrays
    charge_out = np.zeros(T, dtype=float)
    discharge_out = np.zeros(T, dtype=float)
    soc_out = np.zeros(T, dtype=float)
    solver_statuses: list[str] = []
    current_soc = bc["initial_soc_mwh"]

    t = 0
    while t < T:
        window_end = min(t + ph, T)
        window_len = window_end - t
        if window_len <= 0:
            break

        fc_window = load_fc[t:window_end]
        ren_window = renewables[t:window_end]
        pr_window = prices[t:window_end]

        scenarios = _sample_scenarios(fc_window, sigma, n_scenarios, rng)
        lo = scenarios.min(axis=0)
        hi = scenarios.max(axis=0)

        common = dict(
            battery_capacity_mwh=bc["capacity_mwh"],
            battery_max_charge_mw=bc["max_charge_mw"],
            battery_max_discharge_mw=bc["max_discharge_mw"],
            battery_charge_efficiency=bc["charge_efficiency"],
            battery_discharge_efficiency=bc["discharge_efficiency"],
            battery_initial_soc_mwh=current_soc,
            battery_min_soc_mwh=bc["min_soc_mwh"],
            battery_max_soc_mwh=bc["max_soc_mwh"],
            max_grid_import_mw=bc["max_grid_import_mw"],
            default_price_per_mwh=float(np.mean(pr_window)),
            degradation_cost_per_mwh=bc["degradation_cost_per_mwh"],
            time_step_hours=bc["time_step_hours"],
            solver_name=str(cfg.get("solver_name", "appsi_highs")),
        )

        result: dict[str, Any]
        if use_cvar:
            cvar_cfg = CVaRDispatchConfig(**common, beta=cvar_beta, n_scenarios=n_scenarios)
            result = optimize_cvar_dispatch(
                scenarios=scenarios,
                renewables_forecast=ren_window.tolist(),
                price=pr_window.tolist(),
                config=cvar_cfg,
                verbose=False,
            )
        else:
            robust_cfg = RobustDispatchConfig(
                **{k: v for k, v in common.items() if k != "time_step_hours"},
                risk_weight_worst_case=1.0,
            )
            result = optimize_robust_dispatch(
                load_lower_bound=lo.tolist(),
                load_upper_bound=hi.tolist(),
                renewables_forecast=ren_window.tolist(),
                price=pr_window.tolist(),
                config=robust_cfg,
                verbose=False,
            )

        status = result.get("solver_status", "unknown")
        solver_statuses.append(status)

        ch = np.asarray(result.get("battery_charge_mw", np.zeros(window_len)), dtype=float)
        dis = np.asarray(result.get("battery_discharge_mw", np.zeros(window_len)), dtype=float)

        commit = min(replan_interval, window_len)
        eff_c = bc["charge_efficiency"]
        eff_d = max(bc["discharge_efficiency"], 1e-6)
        for k in range(commit):
            c = max(0.0, float(ch[k]))
            d = max(0.0, float(dis[k]))
            c = min(c, max(0.0, (bc["max_soc_mwh"] - current_soc) / eff_c))
            d = min(d, max(0.0, (current_soc - bc["min_soc_mwh"]) * eff_d))
            current_soc = current_soc + eff_c * c - d / eff_d
            current_soc = min(bc["max_soc_mwh"], max(bc["min_soc_mwh"], current_soc))
            charge_out[t + k] = c
            discharge_out[t + k] = d
            soc_out[t + k] = current_soc

        t += commit

    grid_import = np.maximum(0.0, load_fc - renewables - discharge_out + charge_out)
    expected_cost = float(np.sum(prices * grid_import))
    lo_full, hi_full = (
        np.maximum(0.0, load_fc - np.maximum(75.0, 0.08 * np.abs(load_fc))),
        load_fc + np.maximum(75.0, 0.08 * np.abs(load_fc)),
    )

    return {
        "policy": "scenario_mpc",
        "dispatch_plan": {},
        "proposed_charge_mw": charge_out,
        "proposed_discharge_mw": discharge_out,
        "safe_charge_mw": charge_out,
        "safe_discharge_mw": discharge_out,
        "soc_mwh": soc_out,
        "interval_lower": lo_full,
        "interval_upper": hi_full,
        "certificates": [None] * T,
        "constraints": bc,
        "expected_cost_usd": expected_cost,
        "carbon_kg": None,
        "replan_count": len(solver_statuses),
        "solver_statuses": solver_statuses,
    }
