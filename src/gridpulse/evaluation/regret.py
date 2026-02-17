"""
Regret analysis for dispatch optimization.

Computes regret as the cost difference between a dispatch policy
and the oracle (perfect foresight) policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


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
        return np.full(horizon, float(arr[0]), dtype=float)
    if arr.size != horizon:
        raise ValueError(f"{label} length {arr.size} does not match horizon {horizon}")
    return arr.astype(float)


def _require_horizon(arr: np.ndarray, horizon: int, label: str) -> np.ndarray:
    if arr.size != horizon:
        raise ValueError(f"{label} length {arr.size} does not match horizon {horizon}")
    return arr.astype(float)


def _ensure_non_negative(arr: np.ndarray, label: str) -> None:
    if np.any(arr < -1e-9):
        raise ValueError(f"{label} must contain non-negative values")


def _resolve_robust_config(robust_config: Any | None) -> Any:
    """Resolve robust config as RobustDispatchConfig instance without hard module import at import-time."""
    if robust_config is not None and not isinstance(robust_config, dict):
        return robust_config

    try:
        from gridpulse.optimizer.robust_dispatch import RobustDispatchConfig
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError(
            "Robust dispatch dependencies are unavailable. Install pyomo/highspy "
            "or pass an already-constructed robust config object."
        ) from exc

    if robust_config is None:
        return RobustDispatchConfig()
    return RobustDispatchConfig(**robust_config)


def _solve_deterministic_dispatch(
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    deterministic_config: dict[str, Any] | None,
    price: np.ndarray,
) -> dict[str, Any]:
    from gridpulse.optimizer.lp_dispatch import optimize_dispatch

    return optimize_dispatch(
        forecast_load=load_forecast,
        forecast_renewables=renewables_forecast,
        config=deterministic_config or {},
        forecast_price=price,
    )


def _solve_robust_dispatch(
    load_lower_bound: np.ndarray,
    load_upper_bound: np.ndarray,
    renewables_forecast: np.ndarray,
    price: np.ndarray,
    robust_config: Any,
) -> dict[str, Any]:
    from gridpulse.optimizer.robust_dispatch import optimize_robust_dispatch

    return optimize_robust_dispatch(
        load_lower_bound=load_lower_bound,
        load_upper_bound=load_upper_bound,
        renewables_forecast=renewables_forecast,
        price=price,
        config=robust_config,
        verbose=False,
    )


def _extract_schedule(
    solution: dict[str, Any],
    horizon: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    charge_raw = solution.get("battery_charge_mw", np.zeros(horizon, dtype=float))
    discharge_raw = solution.get("battery_discharge_mw", np.zeros(horizon, dtype=float))
    charge = _require_horizon(
        _as_array(charge_raw, f"{label}.battery_charge_mw"),
        horizon,
        f"{label}.battery_charge_mw",
    )
    discharge = _require_horizon(
        _as_array(discharge_raw, f"{label}.battery_discharge_mw"),
        horizon,
        f"{label}.battery_discharge_mw",
    )
    return charge, discharge


def _deterministic_operational_params(
    deterministic_config: dict[str, Any] | None,
) -> tuple[float, float, float, float, float, float]:
    cfg = deterministic_config or {}
    grid_cfg = cfg.get("grid", {}) if isinstance(cfg, dict) else {}
    battery_cfg = cfg.get("battery", {}) if isinstance(cfg, dict) else {}

    max_grid_import = float(grid_cfg.get("max_import_mw", grid_cfg.get("max_draw_mw", 50.0)))
    degradation_cost = float(battery_cfg.get("degradation_cost_per_mwh", 10.0))
    time_step_hours = 1.0
    capacity = float(battery_cfg.get("capacity_mwh", 100.0))
    initial_soc = float(battery_cfg.get("initial_soc_mwh", 0.5 * capacity))
    charge_eff = float(battery_cfg.get("efficiency_regime_a", battery_cfg.get("efficiency", 0.95)))
    discharge_eff = float(battery_cfg.get("efficiency_regime_a", battery_cfg.get("efficiency", 0.95)))
    return max_grid_import, degradation_cost, time_step_hours, initial_soc, charge_eff, discharge_eff


def _robust_operational_params(robust_config: Any) -> tuple[float, float, float, float, float, float]:
    max_grid_import = float(getattr(robust_config, "max_grid_import_mw"))
    degradation_cost = float(getattr(robust_config, "degradation_cost_per_mwh"))
    time_step_hours = float(getattr(robust_config, "time_step_hours", 1.0))
    initial_soc = float(getattr(robust_config, "battery_initial_soc_mwh", 50.0))
    charge_eff = float(getattr(robust_config, "battery_charge_efficiency", 0.95))
    discharge_eff = float(getattr(robust_config, "battery_discharge_efficiency", 0.95))
    return max_grid_import, degradation_cost, time_step_hours, initial_soc, charge_eff, discharge_eff


def _resolve_terminal_soc_policy(
    deterministic_config: dict[str, Any] | None,
) -> tuple[float | None, float]:
    cfg = deterministic_config or {}
    if not isinstance(cfg, dict):
        return None, 0.0

    research_cfg = cfg.get("research_operational", {})
    if not isinstance(research_cfg, dict):
        return None, 0.0
    terminal_cfg = research_cfg.get("terminal_soc", {})
    if not isinstance(terminal_cfg, dict):
        return None, 0.0
    if not bool(terminal_cfg.get("enabled", False)):
        return None, 0.0

    penalty = float(terminal_cfg.get("penalty_per_mwh_shortfall", 0.0))
    if penalty <= 0:
        return None, 0.0

    battery_cfg = cfg.get("battery", {}) if isinstance(cfg.get("battery", {}), dict) else {}
    capacity = float(battery_cfg.get("capacity_mwh", 100.0))
    max_soc = float(battery_cfg.get("max_soc_mwh", capacity))
    target_from_config = terminal_cfg.get("resolved_target_mwh")
    if target_from_config is not None:
        target = float(target_from_config)
    else:
        target_frac = float(terminal_cfg.get("target_soc_fraction", 0.15))
        target_abs = float(terminal_cfg.get("target_soc_mwh", 0.0))
        target = max(target_abs, max(0.0, target_frac) * max(0.0, capacity))
    target = min(max_soc, max(0.0, target))
    return target, penalty


def _default_price(
    horizon: int,
    deterministic_config: dict[str, Any] | None,
    robust_config: Any | None,
) -> np.ndarray:
    det_cfg = deterministic_config or {}
    grid_cfg = det_cfg.get("grid", {}) if isinstance(det_cfg, dict) else {}
    if "price_per_mwh" in grid_cfg:
        return np.full(horizon, float(grid_cfg["price_per_mwh"]), dtype=float)
    if "price_usd_per_mwh" in grid_cfg:
        return np.full(horizon, float(grid_cfg["price_usd_per_mwh"]), dtype=float)
    if robust_config is not None and hasattr(robust_config, "default_price_per_mwh"):
        return np.full(horizon, float(getattr(robust_config, "default_price_per_mwh")), dtype=float)
    return np.full(horizon, 50.0, dtype=float)


def _evaluate_realized_cost(
    load_true: np.ndarray,
    renewables_true: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    price: np.ndarray,
    max_grid_import: float,
    degradation_cost_per_mwh: float,
    unmet_load_penalty_per_mwh: float,
    time_step_hours: float,
    initial_soc_mwh: float | None = None,
    charge_efficiency: float = 1.0,
    discharge_efficiency: float = 1.0,
    terminal_soc_target_mwh: float | None = None,
    terminal_soc_penalty_per_mwh: float = 0.0,
) -> dict[str, float]:
    required_import = load_true - renewables_true - discharge + charge
    grid_import = np.clip(required_import, a_min=0.0, a_max=max_grid_import)
    unmet = np.maximum(required_import - max_grid_import, 0.0)

    throughput = np.maximum(charge, 0.0) + np.maximum(discharge, 0.0)

    grid_cost = float(np.sum(price * grid_import) * time_step_hours)
    degradation_cost = float(
        degradation_cost_per_mwh * np.sum(throughput) * time_step_hours
    )
    unmet_penalty_cost = float(
        unmet_load_penalty_per_mwh * np.sum(unmet) * time_step_hours
    )
    terminal_penalty_cost = 0.0
    if (
        initial_soc_mwh is not None
        and terminal_soc_target_mwh is not None
        and terminal_soc_penalty_per_mwh > 0
    ):
        eta_ch = max(float(charge_efficiency), 1e-9)
        eta_dis = max(float(discharge_efficiency), 1e-9)
        soc = float(initial_soc_mwh)
        for ch, dis in zip(charge, discharge):
            soc += (
                eta_ch * max(float(ch), 0.0)
                - (max(float(dis), 0.0) / eta_dis)
            ) * time_step_hours
        shortfall = max(float(terminal_soc_target_mwh) - soc, 0.0)
        terminal_penalty_cost = float(terminal_soc_penalty_per_mwh * shortfall)

    total_cost = float(grid_cost + degradation_cost + unmet_penalty_cost + terminal_penalty_cost)

    return {
        "total_cost": total_cost,
        "grid_cost": grid_cost,
        "degradation_cost": degradation_cost,
        "unmet_penalty_cost": unmet_penalty_cost,
        "terminal_penalty_cost": terminal_penalty_cost,
    }


def compute_regret(
    actual_cost: float,
    oracle_cost: float,
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute regret metrics.
    
    Args:
        actual_cost: Cost achieved by policy with forecasts
        oracle_cost: Cost with perfect foresight (oracle)
        normalize: Whether to compute percentage regret
    
    Returns:
        Dictionary with:
        - absolute_regret: Cost difference (EUR)
        - relative_regret: Percentage increase over oracle
        - oracle_cost: Oracle cost for reference
        - actual_cost: Actual cost for reference
    """
    regret_abs = actual_cost - oracle_cost
    regret_rel = 100 * regret_abs / max(oracle_cost, 1e-6)
    
    return {
        "absolute_regret": float(regret_abs),
        "relative_regret": float(regret_rel),
        "oracle_cost": float(oracle_cost),
        "actual_cost": float(actual_cost),
    }


def compute_multi_scenario_regret(
    scenario_costs: list[float],
    oracle_costs: list[float],
) -> dict[str, Any]:
    """
    Compute regret statistics across multiple scenarios.
    
    Args:
        scenario_costs: Costs for each scenario (n_scenarios,)
        oracle_costs: Oracle costs for each scenario
    
    Returns:
        Dictionary with aggregated regret metrics
    """
    regrets = []
    for actual, oracle in zip(scenario_costs, oracle_costs):
        r = compute_regret(actual, oracle)
        regrets.append(r["relative_regret"])
    
    regrets = np.array(regrets)
    
    return {
        "mean_regret": float(np.mean(regrets)),
        "std_regret": float(np.std(regrets)),
        "median_regret": float(np.median(regrets)),
        "min_regret": float(np.min(regrets)),
        "max_regret": float(np.max(regrets)),
        "n_scenarios": len(regrets),
    }


def calculate_evpi(
    load_true,
    renewables_true,
    load_forecast,
    renewables_forecast,
    load_lower_bound=None,
    load_upper_bound=None,
    price=None,
    deterministic_config: dict[str, Any] | None = None,
    robust_config: Any | None = None,
    unmet_load_penalty_per_mwh: float = 10000.0,
    actual_model: Literal["robust", "deterministic"] = "robust",
) -> dict[str, float | str | int]:
    """
    Calculate EVPI (Expected Value of Perfect Information) for one scenario.

    EVPI = realized_cost(actual policy) - realized_cost(perfect-information policy)
    """
    if actual_model not in {"robust", "deterministic"}:
        raise ValueError("actual_model must be 'robust' or 'deterministic'")
    if unmet_load_penalty_per_mwh < 0:
        raise ValueError("unmet_load_penalty_per_mwh must be >= 0")

    load_true_arr = _as_array(load_true, "load_true")
    horizon = load_true_arr.size

    load_forecast_arr = _require_horizon(
        _as_array(load_forecast, "load_forecast"),
        horizon,
        "load_forecast",
    )
    renewables_forecast_arr = _require_horizon(
        _as_array(renewables_forecast, "renewables_forecast"),
        horizon,
        "renewables_forecast",
    )
    renewables_true_arr = _require_horizon(
        _as_array(renewables_true, "renewables_true"),
        horizon,
        "renewables_true",
    )

    _ensure_non_negative(load_true_arr, "load_true")
    _ensure_non_negative(renewables_true_arr, "renewables_true")
    _ensure_non_negative(load_forecast_arr, "load_forecast")
    _ensure_non_negative(renewables_forecast_arr, "renewables_forecast")

    robust_cfg = _resolve_robust_config(robust_config) if actual_model == "robust" else None

    if price is None:
        price_arr = _default_price(horizon, deterministic_config, robust_cfg)
    else:
        price_arr = _broadcast(_as_array(price, "price"), horizon, "price")
    _ensure_non_negative(price_arr, "price")

    if actual_model == "robust":
        lb_raw = load_forecast_arr if load_lower_bound is None else _as_array(load_lower_bound, "load_lower_bound")
        ub_raw = load_forecast_arr if load_upper_bound is None else _as_array(load_upper_bound, "load_upper_bound")
        load_lower_arr = _require_horizon(lb_raw, horizon, "load_lower_bound")
        load_upper_arr = _require_horizon(ub_raw, horizon, "load_upper_bound")

        _ensure_non_negative(load_lower_arr, "load_lower_bound")
        _ensure_non_negative(load_upper_arr, "load_upper_bound")
        if np.any(load_lower_arr - load_upper_arr > 1e-9):
            raise ValueError("load_lower_bound must be <= load_upper_bound at every timestep")

        actual_solution = _solve_robust_dispatch(
            load_lower_bound=load_lower_arr,
            load_upper_bound=load_upper_arr,
            renewables_forecast=renewables_forecast_arr,
            price=price_arr,
            robust_config=robust_cfg,
        )
        perfect_solution = _solve_robust_dispatch(
            load_lower_bound=load_true_arr,
            load_upper_bound=load_true_arr,
            renewables_forecast=renewables_true_arr,
            price=price_arr,
            robust_config=robust_cfg,
        )

        actual_charge, actual_discharge = _extract_schedule(actual_solution, horizon, "actual_solution")
        perfect_charge, perfect_discharge = _extract_schedule(perfect_solution, horizon, "perfect_solution")
        (
            max_grid_import,
            degradation_cost,
            time_step_hours,
            initial_soc,
            charge_eff,
            discharge_eff,
        ) = _robust_operational_params(robust_cfg)
    else:
        actual_solution = _solve_deterministic_dispatch(
            load_forecast=load_forecast_arr,
            renewables_forecast=renewables_forecast_arr,
            deterministic_config=deterministic_config,
            price=price_arr,
        )
        perfect_solution = _solve_deterministic_dispatch(
            load_forecast=load_true_arr,
            renewables_forecast=renewables_true_arr,
            deterministic_config=deterministic_config,
            price=price_arr,
        )

        actual_charge, actual_discharge = _extract_schedule(actual_solution, horizon, "actual_solution")
        perfect_charge, perfect_discharge = _extract_schedule(perfect_solution, horizon, "perfect_solution")
        (
            max_grid_import,
            degradation_cost,
            time_step_hours,
            initial_soc,
            charge_eff,
            discharge_eff,
        ) = _deterministic_operational_params(deterministic_config)

    terminal_target_soc_mwh, terminal_shortfall_penalty = _resolve_terminal_soc_policy(deterministic_config)

    actual_eval = _evaluate_realized_cost(
        load_true=load_true_arr,
        renewables_true=renewables_true_arr,
        charge=actual_charge,
        discharge=actual_discharge,
        price=price_arr,
        max_grid_import=max_grid_import,
        degradation_cost_per_mwh=degradation_cost,
        unmet_load_penalty_per_mwh=unmet_load_penalty_per_mwh,
        time_step_hours=time_step_hours,
        initial_soc_mwh=initial_soc,
        charge_efficiency=charge_eff,
        discharge_efficiency=discharge_eff,
        terminal_soc_target_mwh=terminal_target_soc_mwh,
        terminal_soc_penalty_per_mwh=terminal_shortfall_penalty,
    )
    perfect_eval = _evaluate_realized_cost(
        load_true=load_true_arr,
        renewables_true=renewables_true_arr,
        charge=perfect_charge,
        discharge=perfect_discharge,
        price=price_arr,
        max_grid_import=max_grid_import,
        degradation_cost_per_mwh=degradation_cost,
        unmet_load_penalty_per_mwh=unmet_load_penalty_per_mwh,
        time_step_hours=time_step_hours,
        initial_soc_mwh=initial_soc,
        charge_efficiency=charge_eff,
        discharge_efficiency=discharge_eff,
        terminal_soc_target_mwh=terminal_target_soc_mwh,
        terminal_soc_penalty_per_mwh=terminal_shortfall_penalty,
    )

    actual_realized_cost = float(actual_eval["total_cost"])
    perfect_info_cost = float(perfect_eval["total_cost"])

    return {
        "evpi": float(actual_realized_cost - perfect_info_cost),
        "actual_realized_cost": actual_realized_cost,
        "perfect_info_cost": perfect_info_cost,
        "actual_model": actual_model,
        "horizon": int(horizon),
    }


def calculate_vss(
    load_true,
    renewables_true,
    load_forecast,
    renewables_forecast,
    load_lower_bound,
    load_upper_bound,
    price=None,
    deterministic_config: dict[str, Any] | None = None,
    robust_config: Any | None = None,
    unmet_load_penalty_per_mwh: float = 10000.0,
) -> dict[str, float | int]:
    """
    Calculate VSS (Value of Stochastic Solution) on realized ground truth.

    VSS = realized_cost(deterministic policy) - realized_cost(robust policy)
    """
    if unmet_load_penalty_per_mwh < 0:
        raise ValueError("unmet_load_penalty_per_mwh must be >= 0")

    load_true_arr = _as_array(load_true, "load_true")
    horizon = load_true_arr.size

    renewables_true_arr = _require_horizon(
        _as_array(renewables_true, "renewables_true"),
        horizon,
        "renewables_true",
    )
    load_forecast_arr = _require_horizon(
        _as_array(load_forecast, "load_forecast"),
        horizon,
        "load_forecast",
    )
    renewables_forecast_arr = _require_horizon(
        _as_array(renewables_forecast, "renewables_forecast"),
        horizon,
        "renewables_forecast",
    )
    load_lower_arr = _require_horizon(_as_array(load_lower_bound, "load_lower_bound"), horizon, "load_lower_bound")
    load_upper_arr = _require_horizon(_as_array(load_upper_bound, "load_upper_bound"), horizon, "load_upper_bound")

    _ensure_non_negative(load_true_arr, "load_true")
    _ensure_non_negative(renewables_true_arr, "renewables_true")
    _ensure_non_negative(load_forecast_arr, "load_forecast")
    _ensure_non_negative(renewables_forecast_arr, "renewables_forecast")
    _ensure_non_negative(load_lower_arr, "load_lower_bound")
    _ensure_non_negative(load_upper_arr, "load_upper_bound")
    if np.any(load_lower_arr - load_upper_arr > 1e-9):
        raise ValueError("load_lower_bound must be <= load_upper_bound at every timestep")

    robust_cfg = _resolve_robust_config(robust_config)
    if price is None:
        price_arr = _default_price(horizon, deterministic_config, robust_cfg)
    else:
        price_arr = _broadcast(_as_array(price, "price"), horizon, "price")
    _ensure_non_negative(price_arr, "price")

    det_solution = _solve_deterministic_dispatch(
        load_forecast=load_forecast_arr,
        renewables_forecast=renewables_forecast_arr,
        deterministic_config=deterministic_config,
        price=price_arr,
    )
    robust_solution = _solve_robust_dispatch(
        load_lower_bound=load_lower_arr,
        load_upper_bound=load_upper_arr,
        renewables_forecast=renewables_forecast_arr,
        price=price_arr,
        robust_config=robust_cfg,
    )

    det_charge, det_discharge = _extract_schedule(det_solution, horizon, "deterministic_solution")
    robust_charge, robust_discharge = _extract_schedule(robust_solution, horizon, "robust_solution")

    (
        det_max_grid,
        det_degradation,
        det_time_step,
        det_initial_soc,
        det_charge_eff,
        det_discharge_eff,
    ) = _deterministic_operational_params(deterministic_config)
    (
        rob_max_grid,
        rob_degradation,
        rob_time_step,
        rob_initial_soc,
        rob_charge_eff,
        rob_discharge_eff,
    ) = _robust_operational_params(robust_cfg)
    terminal_target_soc_mwh, terminal_shortfall_penalty = _resolve_terminal_soc_policy(deterministic_config)

    det_eval = _evaluate_realized_cost(
        load_true=load_true_arr,
        renewables_true=renewables_true_arr,
        charge=det_charge,
        discharge=det_discharge,
        price=price_arr,
        max_grid_import=det_max_grid,
        degradation_cost_per_mwh=det_degradation,
        unmet_load_penalty_per_mwh=unmet_load_penalty_per_mwh,
        time_step_hours=det_time_step,
        initial_soc_mwh=det_initial_soc,
        charge_efficiency=det_charge_eff,
        discharge_efficiency=det_discharge_eff,
        terminal_soc_target_mwh=terminal_target_soc_mwh,
        terminal_soc_penalty_per_mwh=terminal_shortfall_penalty,
    )
    robust_eval = _evaluate_realized_cost(
        load_true=load_true_arr,
        renewables_true=renewables_true_arr,
        charge=robust_charge,
        discharge=robust_discharge,
        price=price_arr,
        max_grid_import=rob_max_grid,
        degradation_cost_per_mwh=rob_degradation,
        unmet_load_penalty_per_mwh=unmet_load_penalty_per_mwh,
        time_step_hours=rob_time_step,
        initial_soc_mwh=rob_initial_soc,
        charge_efficiency=rob_charge_eff,
        discharge_efficiency=rob_discharge_eff,
        terminal_soc_target_mwh=terminal_target_soc_mwh,
        terminal_soc_penalty_per_mwh=terminal_shortfall_penalty,
    )

    deterministic_realized_cost = float(det_eval["total_cost"])
    robust_realized_cost = float(robust_eval["total_cost"])

    return {
        "vss": float(deterministic_realized_cost - robust_realized_cost),
        "deterministic_realized_cost": deterministic_realized_cost,
        "robust_realized_cost": robust_realized_cost,
        "horizon": int(horizon),
    }


def generate_stochastic_metrics_report(
    scenarios: list[dict[str, Any]],
    output_csv: str | Path,
    deterministic_config: dict[str, Any] | None = None,
    robust_config: Any | None = None,
    unmet_load_penalty_per_mwh: float = 10000.0,
) -> pd.DataFrame:
    """
    Generate and save a stochastic-programming metrics CSV with scenario rows + summary row.
    """
    if not scenarios:
        raise ValueError("scenarios must be a non-empty list")

    required_keys = {
        "scenario",
        "load_true",
        "renewables_true",
        "load_forecast",
        "renewables_forecast",
        "load_lower_bound",
        "load_upper_bound",
    }

    rows: list[dict[str, Any]] = []
    for i, scenario in enumerate(scenarios):
        missing = sorted(required_keys - set(scenario.keys()))
        if missing:
            raise ValueError(f"scenario index {i} missing required keys: {missing}")

        robust_evpi = calculate_evpi(
            load_true=scenario["load_true"],
            renewables_true=scenario["renewables_true"],
            load_forecast=scenario["load_forecast"],
            renewables_forecast=scenario["renewables_forecast"],
            load_lower_bound=scenario["load_lower_bound"],
            load_upper_bound=scenario["load_upper_bound"],
            price=scenario.get("price"),
            deterministic_config=deterministic_config,
            robust_config=robust_config,
            unmet_load_penalty_per_mwh=unmet_load_penalty_per_mwh,
            actual_model="robust",
        )
        deterministic_evpi = calculate_evpi(
            load_true=scenario["load_true"],
            renewables_true=scenario["renewables_true"],
            load_forecast=scenario["load_forecast"],
            renewables_forecast=scenario["renewables_forecast"],
            load_lower_bound=scenario["load_lower_bound"],
            load_upper_bound=scenario["load_upper_bound"],
            price=scenario.get("price"),
            deterministic_config=deterministic_config,
            robust_config=robust_config,
            unmet_load_penalty_per_mwh=unmet_load_penalty_per_mwh,
            actual_model="deterministic",
        )
        vss = calculate_vss(
            load_true=scenario["load_true"],
            renewables_true=scenario["renewables_true"],
            load_forecast=scenario["load_forecast"],
            renewables_forecast=scenario["renewables_forecast"],
            load_lower_bound=scenario["load_lower_bound"],
            load_upper_bound=scenario["load_upper_bound"],
            price=scenario.get("price"),
            deterministic_config=deterministic_config,
            robust_config=robust_config,
            unmet_load_penalty_per_mwh=unmet_load_penalty_per_mwh,
        )

        rows.append(
            {
                "row_type": "scenario",
                "scenario": str(scenario["scenario"]),
                "horizon": int(robust_evpi["horizon"]),
                "evpi": float(robust_evpi["evpi"]),
                "evpi_robust": float(robust_evpi["evpi"]),
                "evpi_deterministic": float(deterministic_evpi["evpi"]),
                "vss": float(vss["vss"]),
                "robust_actual_realized_cost": float(robust_evpi["actual_realized_cost"]),
                "robust_perfect_info_cost": float(robust_evpi["perfect_info_cost"]),
                "deterministic_actual_realized_cost": float(deterministic_evpi["actual_realized_cost"]),
                "deterministic_perfect_info_cost": float(deterministic_evpi["perfect_info_cost"]),
                "unmet_load_penalty_per_mwh": float(unmet_load_penalty_per_mwh),
            }
        )

    df = pd.DataFrame(rows)
    summary_numeric_cols = [
        "horizon",
        "evpi",
        "evpi_robust",
        "evpi_deterministic",
        "vss",
        "robust_actual_realized_cost",
        "robust_perfect_info_cost",
        "deterministic_actual_realized_cost",
        "deterministic_perfect_info_cost",
    ]
    summary_row: dict[str, Any] = {
        "row_type": "summary_mean",
        "scenario": "__summary_mean__",
        "unmet_load_penalty_per_mwh": float(unmet_load_penalty_per_mwh),
    }
    for col in summary_numeric_cols:
        summary_row[col] = float(pd.to_numeric(df[col], errors="coerce").mean())

    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
