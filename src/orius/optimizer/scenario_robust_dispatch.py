"""Pyomo-based multi-scenario robust battery dispatch optimization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from .robust_dispatch import RobustDispatchConfig


def _as_scenarios(x: Any, label: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D array with shape (S, H)")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{label} must have shape (S, H) with S>=1 and H>=1")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain finite values")
    return arr


def _as_array(x: Any, label: str) -> np.ndarray:
    if isinstance(x, list | tuple | np.ndarray):
        arr = np.asarray(x, dtype=float)
    else:
        arr = np.asarray([x], dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D series")
    if arr.size == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain finite values")
    return arr


def _broadcast(arr: np.ndarray, horizon: int, label: str) -> np.ndarray:
    if arr.size == 1 and horizon > 1:
        return np.full(horizon, float(arr[0]), dtype=float)
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
    if not (0.0 <= cfg.risk_weight_worst_case <= 1.0):
        raise ValueError("risk_weight_worst_case must be in [0, 1]")
    if cfg.time_step_hours <= 0:
        raise ValueError("time_step_hours must be > 0")


def _ensure_highs_solver_available(solver_name: str) -> None:
    solver = pyo.SolverFactory(solver_name)
    if solver is None:
        raise RuntimeError(
            "HiGHS solver is required for scenario robust dispatch. Install pyomo and highspy "
            "and set solver_name='appsi_highs'."
        )
    try:
        available = bool(solver.available(exception_flag=False))
    except Exception as exc:
        raise RuntimeError(
            "HiGHS solver is required for scenario robust dispatch. Install pyomo and highspy "
            "and set solver_name='appsi_highs'."
        ) from exc
    if not available:
        raise RuntimeError(
            "HiGHS solver is required for scenario robust dispatch. Install pyomo and highspy "
            "and set solver_name='appsi_highs'."
        )


def _infeasible_result(
    n_scenarios: int, horizon: int, status: str, cfg: RobustDispatchConfig
) -> dict[str, Any]:
    zeros_h = np.zeros(horizon, dtype=float)
    zeros_sh = np.zeros((n_scenarios, horizon), dtype=float)
    return {
        "battery_charge_mw": zeros_h.tolist(),
        "battery_discharge_mw": zeros_h.tolist(),
        "total_cost": None,
        "feasible": False,
        "solver_status": status,
        "scenario_costs": [],
        "binding_scenario": None,
        "worst_case_cost": None,
        "mean_scenario_cost": None,
        "degradation_cost": None,
        "grid_import_mw": zeros_sh.tolist(),
        "soc_mwh": np.full((n_scenarios, horizon), cfg.battery_initial_soc_mwh, dtype=float).tolist(),
        "risk_weight_worst_case": float(cfg.risk_weight_worst_case),
    }


def optimize_scenario_robust_dispatch(
    load_scenarios,
    renewables_forecast,
    price=None,
    config: RobustDispatchConfig | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Solve shared-action dispatch against an explicit set of load scenarios."""
    cfg = config or RobustDispatchConfig()
    _validate_config(cfg)

    scen = _as_scenarios(load_scenarios, "load_scenarios")
    s_count, horizon = scen.shape
    renewables = _broadcast(
        _as_array(renewables_forecast, "renewables_forecast"), horizon, "renewables_forecast"
    )
    if price is None:
        prices = np.full(horizon, cfg.default_price_per_mwh, dtype=float)
    else:
        prices = _broadcast(_as_array(price, "price"), horizon, "price")

    _ensure_non_negative(scen.reshape(-1), "load_scenarios")
    _ensure_non_negative(renewables, "renewables_forecast")
    _ensure_non_negative(prices, "price")

    _ensure_highs_solver_available(cfg.solver_name)

    model = pyo.ConcreteModel(name="scenario_robust_dispatch")
    model.T = pyo.RangeSet(0, horizon - 1)
    model.S = pyo.RangeSet(0, s_count - 1)

    model.P_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, cfg.battery_max_charge_mw))
    model.P_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, cfg.battery_max_discharge_mw))
    model.G = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals, bounds=(0.0, cfg.max_grid_import_mw))
    model.SoC = pyo.Var(model.S, model.T, bounds=(cfg.battery_min_soc_mwh, cfg.battery_max_soc_mwh))
    model.z = pyo.Var(domain=pyo.NonNegativeReals)

    def adequacy_rule(m, s, t):
        rhs = float(scen[s, t] - renewables[t])
        return m.P_dis[t] - m.P_ch[t] + m.G[s, t] >= rhs

    model.adequacy = pyo.Constraint(model.S, model.T, rule=adequacy_rule)

    def soc_initial_rule(m, s):
        return (
            m.SoC[s, 0]
            == cfg.battery_initial_soc_mwh
            + cfg.time_step_hours * cfg.battery_charge_efficiency * m.P_ch[0]
            - cfg.time_step_hours * (1.0 / cfg.battery_discharge_efficiency) * m.P_dis[0]
        )

    model.soc_initial = pyo.Constraint(model.S, rule=soc_initial_rule)

    def soc_dyn_rule(m, s, t):
        if t == 0:
            return pyo.Constraint.Skip
        return (
            m.SoC[s, t]
            == m.SoC[s, t - 1]
            + cfg.time_step_hours * cfg.battery_charge_efficiency * m.P_ch[t]
            - cfg.time_step_hours * (1.0 / cfg.battery_discharge_efficiency) * m.P_dis[t]
        )

    model.soc_dyn = pyo.Constraint(model.S, model.T, rule=soc_dyn_rule)

    def scenario_grid_cost(m, s):
        return sum(prices[t] * m.G[s, t] * cfg.time_step_hours for t in m.T)

    def worst_case_epigraph_rule(m, s):
        return m.z >= scenario_grid_cost(m, s)

    model.worst_case_epigraph = pyo.Constraint(model.S, rule=worst_case_epigraph_rule)

    mean_cost = (1.0 / s_count) * sum(scenario_grid_cost(model, s) for s in model.S)
    throughput = sum((model.P_ch[t] + model.P_dis[t]) * cfg.time_step_hours for t in model.T)
    model.obj = pyo.Objective(
        expr=(
            cfg.risk_weight_worst_case * model.z
            + (1.0 - cfg.risk_weight_worst_case) * mean_cost
            + cfg.degradation_cost_per_mwh * throughput
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(cfg.solver_name)
    result = solver.solve(model, tee=verbose, load_solutions=False)
    term = result.solver.termination_condition
    status = result.solver.status
    status_str = f"{status}:{term}"
    if term != TerminationCondition.optimal:
        return _infeasible_result(s_count, horizon, status_str, cfg)

    if hasattr(solver, "load_vars"):
        solver.load_vars()

    battery_charge = np.asarray([pyo.value(model.P_ch[t]) for t in model.T], dtype=float)
    battery_discharge = np.asarray([pyo.value(model.P_dis[t]) for t in model.T], dtype=float)
    grid_import = np.asarray([[pyo.value(model.G[s, t]) for t in model.T] for s in model.S], dtype=float)
    soc = np.asarray([[pyo.value(model.SoC[s, t]) for t in model.T] for s in model.S], dtype=float)
    scenario_costs = np.asarray(
        [float(np.sum(prices * grid_import[s]) * cfg.time_step_hours) for s in range(s_count)],
        dtype=float,
    )
    worst_case_cost = float(np.max(scenario_costs))
    mean_scenario_cost = float(np.mean(scenario_costs))
    degradation_cost = float(
        cfg.degradation_cost_per_mwh * np.sum((battery_charge + battery_discharge) * cfg.time_step_hours)
    )
    binding_scenario = int(np.argmax(scenario_costs))

    return {
        "battery_charge_mw": battery_charge.tolist(),
        "battery_discharge_mw": battery_discharge.tolist(),
        "total_cost": float(worst_case_cost + degradation_cost),
        "feasible": True,
        "solver_status": status_str,
        "scenario_costs": scenario_costs.tolist(),
        "binding_scenario": binding_scenario,
        "worst_case_cost": worst_case_cost,
        "mean_scenario_cost": mean_scenario_cost,
        "degradation_cost": degradation_cost,
        "grid_import_mw": grid_import.tolist(),
        "soc_mwh": soc.tolist(),
        "risk_weight_worst_case": float(cfg.risk_weight_worst_case),
    }
