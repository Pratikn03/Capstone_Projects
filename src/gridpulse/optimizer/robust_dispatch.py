"""Pyomo-based robust battery dispatch optimization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


@dataclass
class RobustDispatchConfig:
    """Configuration for robust dispatch optimization."""

    battery_capacity_mwh: float = 100.0
    battery_max_charge_mw: float = 50.0
    battery_max_discharge_mw: float = 50.0
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.95
    battery_initial_soc_mwh: float = 50.0
    battery_min_soc_mwh: float = 10.0
    battery_max_soc_mwh: float = 90.0

    max_grid_import_mw: float = 500.0
    default_price_per_mwh: float = 60.0
    degradation_cost_per_mwh: float = 5.0

    time_step_hours: float = 1.0
    solver_name: str = "appsi_highs"


def _as_array(x: Any, label: str) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
    else:
        arr = np.asarray([x], dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D series")
    if arr.size == 0:
        raise ValueError(f"{label} must be non-empty")
    return arr


def _broadcast(arr: np.ndarray, horizon: int, label: str) -> np.ndarray:
    if arr.size == 1 and horizon > 1:
        return np.full(horizon, float(arr[0]))
    if arr.size != horizon:
        raise ValueError(f"{label} length {arr.size} does not match horizon {horizon}")
    return arr.astype(float)


def _ensure_non_negative(arr: np.ndarray, label: str) -> None:
    if np.any(arr < -1e-9):
        raise ValueError(f"{label} must contain non-negative values")


def _validate_config(cfg: RobustDispatchConfig) -> None:
    if cfg.battery_capacity_mwh <= 0:
        raise ValueError("battery_capacity_mwh must be > 0")
    if cfg.battery_max_charge_mw < 0 or cfg.battery_max_discharge_mw < 0:
        raise ValueError("battery_max_charge_mw and battery_max_discharge_mw must be >= 0")
    if not (0 < cfg.battery_charge_efficiency <= 1):
        raise ValueError("battery_charge_efficiency must be in (0, 1]")
    if not (0 < cfg.battery_discharge_efficiency <= 1):
        raise ValueError("battery_discharge_efficiency must be in (0, 1]")
    if cfg.battery_min_soc_mwh < 0:
        raise ValueError("battery_min_soc_mwh must be >= 0")
    if cfg.battery_max_soc_mwh > cfg.battery_capacity_mwh + 1e-9:
        raise ValueError("battery_max_soc_mwh cannot exceed battery_capacity_mwh")
    if cfg.battery_min_soc_mwh > cfg.battery_max_soc_mwh + 1e-9:
        raise ValueError("battery_min_soc_mwh cannot exceed battery_max_soc_mwh")
    if not (cfg.battery_min_soc_mwh - 1e-9 <= cfg.battery_initial_soc_mwh <= cfg.battery_max_soc_mwh + 1e-9):
        raise ValueError("battery_initial_soc_mwh must lie within [battery_min_soc_mwh, battery_max_soc_mwh]")
    if cfg.max_grid_import_mw < 0:
        raise ValueError("max_grid_import_mw must be >= 0")
    if cfg.default_price_per_mwh < 0:
        raise ValueError("default_price_per_mwh must be >= 0")
    if cfg.degradation_cost_per_mwh < 0:
        raise ValueError("degradation_cost_per_mwh must be >= 0")
    if cfg.time_step_hours <= 0:
        raise ValueError("time_step_hours must be > 0")


def _ensure_highs_solver_available(solver_name: str) -> None:
    solver = pyo.SolverFactory(solver_name)
    if solver is None:
        raise RuntimeError(
            "HiGHS solver is required for robust dispatch. Install pyomo and highspy "
            "and set solver_name='appsi_highs'."
        )
    try:
        available = bool(solver.available(exception_flag=False))
    except Exception as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(
            "HiGHS solver is required for robust dispatch. Install pyomo and highspy "
            "and set solver_name='appsi_highs'."
        ) from exc
    if not available:
        raise RuntimeError(
            "HiGHS solver is required for robust dispatch. Install pyomo and highspy "
            "and set solver_name='appsi_highs'."
        )


def _infeasible_result(horizon: int, status: str, cfg: RobustDispatchConfig) -> dict[str, Any]:
    soc_fallback = np.full(horizon, cfg.battery_initial_soc_mwh, dtype=float)
    zeros = np.zeros(horizon, dtype=float)
    return {
        "battery_charge_mw": zeros.tolist(),
        "battery_discharge_mw": zeros.tolist(),
        "soc_mwh_lower": soc_fallback.tolist(),
        "soc_mwh_upper": soc_fallback.tolist(),
        "grid_import_mw_lower": zeros.tolist(),
        "grid_import_mw_upper": zeros.tolist(),
        "worst_case_cost": None,
        "degradation_cost": None,
        "total_cost": None,
        "scenario_cost_lower": None,
        "scenario_cost_upper": None,
        "feasible": False,
        "solver_status": status,
        "binding_scenario": None,
    }


def optimize_robust_dispatch(
    load_lower_bound,
    load_upper_bound,
    renewables_forecast,
    price=None,
    config: RobustDispatchConfig | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Solve robust battery dispatch using a two-scenario min-max DRO LP."""
    cfg = config or RobustDispatchConfig()
    _validate_config(cfg)

    load_lower = _as_array(load_lower_bound, "load_lower_bound")
    load_upper = _as_array(load_upper_bound, "load_upper_bound")

    horizon = max(load_lower.size, load_upper.size)
    load_lower = _broadcast(load_lower, horizon, "load_lower_bound")
    load_upper = _broadcast(load_upper, horizon, "load_upper_bound")
    renewables = _broadcast(_as_array(renewables_forecast, "renewables_forecast"), horizon, "renewables_forecast")

    if price is None:
        prices = np.full(horizon, cfg.default_price_per_mwh, dtype=float)
    else:
        prices = _broadcast(_as_array(price, "price"), horizon, "price")

    _ensure_non_negative(load_lower, "load_lower_bound")
    _ensure_non_negative(load_upper, "load_upper_bound")
    _ensure_non_negative(renewables, "renewables_forecast")
    _ensure_non_negative(prices, "price")

    if np.any(load_lower - load_upper > 1e-9):
        raise ValueError("load_lower_bound must be <= load_upper_bound at every timestep")

    scenario_loads = {"lower": load_lower, "upper": load_upper}

    _ensure_highs_solver_available(cfg.solver_name)

    model = pyo.ConcreteModel(name="robust_dispatch_dro")
    model.T = pyo.RangeSet(0, horizon - 1)
    model.S = pyo.Set(initialize=("lower", "upper"), ordered=True)

    model.P_ch = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, cfg.battery_max_charge_mw),
    )
    model.P_dis = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, cfg.battery_max_discharge_mw),
    )
    model.G = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
    model.SoC = pyo.Var(
        model.S,
        model.T,
        bounds=(cfg.battery_min_soc_mwh, cfg.battery_max_soc_mwh),
    )
    model.z = pyo.Var(domain=pyo.NonNegativeReals)

    def power_adequacy_rule(m, s, t):
        rhs = float(scenario_loads[s][t] - renewables[t])
        return m.P_dis[t] - m.P_ch[t] + m.G[s, t] >= rhs

    model.power_adequacy = pyo.Constraint(model.S, model.T, rule=power_adequacy_rule)

    def grid_cap_rule(m, s, t):
        return m.G[s, t] <= cfg.max_grid_import_mw

    model.grid_cap = pyo.Constraint(model.S, model.T, rule=grid_cap_rule)

    def soc_initial_rule(m, s):
        return (
            m.SoC[s, 0]
            == cfg.battery_initial_soc_mwh
            + cfg.battery_charge_efficiency * m.P_ch[0]
            - (1.0 / cfg.battery_discharge_efficiency) * m.P_dis[0]
        )

    model.soc_initial = pyo.Constraint(model.S, rule=soc_initial_rule)

    def soc_dynamics_rule(m, s, t):
        if t == 0:
            return pyo.Constraint.Skip
        return (
            m.SoC[s, t]
            == m.SoC[s, t - 1]
            + cfg.battery_charge_efficiency * m.P_ch[t]
            - (1.0 / cfg.battery_discharge_efficiency) * m.P_dis[t]
        )

    model.soc_dynamics = pyo.Constraint(model.S, model.T, rule=soc_dynamics_rule)

    def worst_case_epigraph_rule(m, s):
        return m.z >= sum(prices[t] * m.G[s, t] * cfg.time_step_hours for t in m.T)

    model.worst_case_epigraph = pyo.Constraint(model.S, rule=worst_case_epigraph_rule)

    throughput = sum((model.P_ch[t] + model.P_dis[t]) * cfg.time_step_hours for t in model.T)
    model.obj = pyo.Objective(
        expr=model.z + cfg.degradation_cost_per_mwh * throughput,
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(cfg.solver_name)
    result = solver.solve(model, tee=verbose, load_solutions=False)

    term = result.solver.termination_condition
    status = result.solver.status
    status_str = f"{status}:{term}"

    if term != TerminationCondition.optimal:
        return _infeasible_result(horizon, status_str, cfg)

    if hasattr(solver, "load_vars"):
        solver.load_vars()

    battery_charge = np.asarray([pyo.value(model.P_ch[t]) for t in model.T], dtype=float)
    battery_discharge = np.asarray([pyo.value(model.P_dis[t]) for t in model.T], dtype=float)

    grid_lower = np.asarray([pyo.value(model.G["lower", t]) for t in model.T], dtype=float)
    grid_upper = np.asarray([pyo.value(model.G["upper", t]) for t in model.T], dtype=float)

    soc_lower = np.asarray([pyo.value(model.SoC["lower", t]) for t in model.T], dtype=float)
    soc_upper = np.asarray([pyo.value(model.SoC["upper", t]) for t in model.T], dtype=float)

    scenario_cost_lower = float(np.sum(prices * grid_lower) * cfg.time_step_hours)
    scenario_cost_upper = float(np.sum(prices * grid_upper) * cfg.time_step_hours)
    worst_case_cost = max(scenario_cost_lower, scenario_cost_upper)

    degradation_cost = float(
        cfg.degradation_cost_per_mwh
        * np.sum((battery_charge + battery_discharge) * cfg.time_step_hours)
    )
    total_cost = float(worst_case_cost + degradation_cost)

    binding_scenario = "lower" if scenario_cost_lower >= scenario_cost_upper - 1e-9 else "upper"

    return {
        "battery_charge_mw": battery_charge.tolist(),
        "battery_discharge_mw": battery_discharge.tolist(),
        "soc_mwh_lower": soc_lower.tolist(),
        "soc_mwh_upper": soc_upper.tolist(),
        "grid_import_mw_lower": grid_lower.tolist(),
        "grid_import_mw_upper": grid_upper.tolist(),
        "worst_case_cost": worst_case_cost,
        "degradation_cost": degradation_cost,
        "total_cost": total_cost,
        "scenario_cost_lower": scenario_cost_lower,
        "scenario_cost_upper": scenario_cost_upper,
        "feasible": True,
        "solver_status": status_str,
        "binding_scenario": binding_scenario,
    }


def evaluate_dispatch_robustness(
    load_true,
    renewables_true,
    load_lower_bound,
    load_upper_bound,
    renewables_forecast,
    dispatch_solution: dict[str, Any],
    price=None,
    config: RobustDispatchConfig | None = None,
) -> dict[str, float | int | None]:
    """Evaluate a robust dispatch schedule against realized load and renewables."""
    cfg = config or RobustDispatchConfig()
    _validate_config(cfg)

    load_true_arr = _as_array(load_true, "load_true")
    horizon = load_true_arr.size

    renew_true_arr = _broadcast(_as_array(renewables_true, "renewables_true"), horizon, "renewables_true")
    renew_forecast_arr = _broadcast(
        _as_array(renewables_forecast, "renewables_forecast"),
        horizon,
        "renewables_forecast",
    )
    load_lower_arr = _broadcast(_as_array(load_lower_bound, "load_lower_bound"), horizon, "load_lower_bound")
    load_upper_arr = _broadcast(_as_array(load_upper_bound, "load_upper_bound"), horizon, "load_upper_bound")

    if price is None:
        prices = np.full(horizon, cfg.default_price_per_mwh, dtype=float)
    else:
        prices = _broadcast(_as_array(price, "price"), horizon, "price")

    _ensure_non_negative(load_true_arr, "load_true")
    _ensure_non_negative(renew_true_arr, "renewables_true")
    _ensure_non_negative(renew_forecast_arr, "renewables_forecast")
    _ensure_non_negative(load_lower_arr, "load_lower_bound")
    _ensure_non_negative(load_upper_arr, "load_upper_bound")
    _ensure_non_negative(prices, "price")

    battery_charge = _broadcast(
        _as_array(dispatch_solution.get("battery_charge_mw", []), "dispatch_solution.battery_charge_mw"),
        horizon,
        "dispatch_solution.battery_charge_mw",
    )
    battery_discharge = _broadcast(
        _as_array(dispatch_solution.get("battery_discharge_mw", []), "dispatch_solution.battery_discharge_mw"),
        horizon,
        "dispatch_solution.battery_discharge_mw",
    )

    realized_grid = np.zeros(horizon, dtype=float)
    violations = 0
    grid_violations = 0
    soc_violations = 0

    soc = cfg.battery_initial_soc_mwh
    for t in range(horizon):
        soc = (
            soc
            + cfg.battery_charge_efficiency * battery_charge[t]
            - (1.0 / cfg.battery_discharge_efficiency) * battery_discharge[t]
        )

        if soc < cfg.battery_min_soc_mwh - 1e-6 or soc > cfg.battery_max_soc_mwh + 1e-6:
            violations += 1
            soc_violations += 1

        net_load = load_true_arr[t] - renew_true_arr[t]
        required_import = net_load - battery_discharge[t] + battery_charge[t]
        realized_grid[t] = max(required_import, 0.0)

        if realized_grid[t] > cfg.max_grid_import_mw + 1e-6:
            violations += 1
            grid_violations += 1

    realized_grid_cost = float(np.sum(prices * realized_grid) * cfg.time_step_hours)
    realized_degradation = float(
        cfg.degradation_cost_per_mwh
        * np.sum((battery_charge + battery_discharge) * cfg.time_step_hours)
    )
    realized_cost = float(realized_grid_cost + realized_degradation)

    oracle = optimize_robust_dispatch(
        load_lower_bound=load_true_arr,
        load_upper_bound=load_true_arr,
        renewables_forecast=renew_true_arr,
        price=prices,
        config=cfg,
        verbose=False,
    )
    oracle_cost = oracle.get("total_cost")

    regret = None
    regret_pct = None
    if oracle_cost is not None:
        regret = float(realized_cost - float(oracle_cost))
        regret_pct = float(100.0 * regret / max(float(oracle_cost), 1e-9))

    load_mid = 0.5 * (load_lower_arr + load_upper_arr)
    load_rmse = float(np.sqrt(np.mean((load_true_arr - load_mid) ** 2)))
    renew_rmse = float(np.sqrt(np.mean((renew_true_arr - renew_forecast_arr) ** 2)))

    return {
        "realized_cost": realized_cost,
        "realized_grid_cost": realized_grid_cost,
        "realized_degradation_cost": realized_degradation,
        "oracle_cost": float(oracle_cost) if oracle_cost is not None else None,
        "regret": regret,
        "regret_pct": regret_pct,
        "forecast_error_load": load_rmse,
        "forecast_error_renewables": renew_rmse,
        "constraint_violations": int(violations),
        "grid_import_violations": int(grid_violations),
        "soc_violations": int(soc_violations),
        "violation_rate": float(violations / horizon),
    }


def run_perturbation_analysis(
    load_lower_bound,
    load_upper_bound,
    renewables_forecast,
    load_true,
    renewables_true,
    price=None,
    noise_levels: list[float] | None = None,
    config: RobustDispatchConfig | None = None,
    n_samples: int = 10,
) -> pd.DataFrame:
    """Run Monte Carlo perturbation analysis for robust dispatch."""
    cfg = config or RobustDispatchConfig()

    noise_levels = noise_levels or [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    load_lower_arr = _as_array(load_lower_bound, "load_lower_bound")
    load_upper_arr = _as_array(load_upper_bound, "load_upper_bound")
    horizon = max(load_lower_arr.size, load_upper_arr.size)

    load_lower_arr = _broadcast(load_lower_arr, horizon, "load_lower_bound")
    load_upper_arr = _broadcast(load_upper_arr, horizon, "load_upper_bound")
    renew_forecast_arr = _broadcast(
        _as_array(renewables_forecast, "renewables_forecast"),
        horizon,
        "renewables_forecast",
    )
    load_true_arr = _broadcast(_as_array(load_true, "load_true"), horizon, "load_true")
    renew_true_arr = _broadcast(_as_array(renewables_true, "renewables_true"), horizon, "renewables_true")

    if price is None:
        prices = np.full(horizon, cfg.default_price_per_mwh, dtype=float)
    else:
        prices = _broadcast(_as_array(price, "price"), horizon, "price")

    mid = 0.5 * (load_lower_arr + load_upper_arr)
    half_width = 0.5 * (load_upper_arr - load_lower_arr)
    load_std = float(np.std(mid))

    results: list[dict[str, float | int | bool | None]] = []

    for noise_pct in noise_levels:
        for sample in range(n_samples):
            rng = np.random.default_rng(42 + sample)
            noisy_mid = mid + rng.normal(0.0, noise_pct * load_std, size=horizon)
            noisy_mid = np.maximum(noisy_mid, 0.0)

            noisy_lower = np.maximum(noisy_mid - half_width, 0.0)
            noisy_upper = np.maximum(noisy_mid + half_width, noisy_lower)

            solution = optimize_robust_dispatch(
                load_lower_bound=noisy_lower,
                load_upper_bound=noisy_upper,
                renewables_forecast=renew_forecast_arr,
                price=prices,
                config=cfg,
                verbose=False,
            )

            eval_metrics = evaluate_dispatch_robustness(
                load_true=load_true_arr,
                renewables_true=renew_true_arr,
                load_lower_bound=noisy_lower,
                load_upper_bound=noisy_upper,
                renewables_forecast=renew_forecast_arr,
                dispatch_solution=solution,
                price=prices,
                config=cfg,
            )

            results.append(
                {
                    "noise_level": float(noise_pct),
                    "sample": int(sample),
                    "realized_cost": eval_metrics["realized_cost"],
                    "regret": eval_metrics["regret"],
                    "regret_pct": eval_metrics["regret_pct"],
                    "infeasible_rate": eval_metrics["violation_rate"],
                    "feasible": bool(solution.get("feasible", False)),
                }
            )

    return pd.DataFrame(results)
