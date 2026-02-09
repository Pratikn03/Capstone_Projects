"""
Robust dispatch optimization with uncertainty-aware forecasts.

This module extends standard dispatch optimization by incorporating prediction
intervals from conformal prediction. Instead of using point forecasts, it uses
lower/upper bounds to ensure robust operation under forecast uncertainty.

Key features:
- Conformal prediction intervals (90%, 95% coverage)
- Safety-margin approach: use conservative forecasts for dispatch
- Robustness evaluation under forecast perturbations
- Infeasibility detection and reporting
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass
class RobustDispatchConfig:
    """Configuration for robust dispatch optimization."""
    
    # Uncertainty quantification
    conformal_alpha: float = 0.10  # 90% coverage intervals
    safety_margin: float = 0.0  # Additional safety buffer (0-1 = 0-100%)
    
    # Dispatch mode
    mode: Literal["point", "lower", "upper", "conservative"] = "conservative"
    # - point: use point forecast (standard)
    # - lower: use lower bound of prediction interval
    # - upper: use upper bound of prediction interval
    # - conservative: use lower bound for load, upper for renewables
    
    # Robust optimization
    enable_robust: bool = True
    robust_gamma: float = 0.0  # Robustness budget (0 = nominal, >0 = robust)
    
    # Battery constraints
    battery_capacity_mwh: float = 100.0
    battery_power_mw: float = 50.0
    battery_efficiency: float = 0.92
    battery_initial_soc: float = 0.5
    
    # Cost parameters
    grid_buy_price: float = 60.0  # EUR/MWh
    grid_sell_price: float = 50.0  # EUR/MWh
    carbon_penalty: float = 20.0  # EUR/ton CO2
    carbon_intensity: float = 0.4  # ton CO2/MWh
    
    # Constraints
    max_grid_import_mw: float = 500.0
    max_grid_export_mw: float = 300.0


def apply_forecast_uncertainty(
    forecast: np.ndarray,
    lower_bound: np.ndarray | None = None,
    upper_bound: np.ndarray | None = None,
    mode: str = "conservative",
    forecast_type: str = "load",
) -> np.ndarray:
    """
    Apply uncertainty-aware forecast transformation.
    
    Args:
        forecast: Point forecast (n_steps,)
        lower_bound: Lower prediction interval bound
        upper_bound: Upper prediction interval bound
        mode: "point", "lower", "upper", "conservative"
        forecast_type: "load", "wind", "solar" (affects conservative strategy)
    
    Returns:
        Adjusted forecast for robust dispatch
    """
    if mode == "point":
        return forecast
    
    if mode == "lower" and lower_bound is not None:
        return lower_bound
    
    if mode == "upper" and upper_bound is not None:
        return upper_bound
    
    if mode == "conservative":
        # Conservative: prepare for worst-case
        # - Load: use upper bound (prepare for higher demand)
        # - Renewables: use lower bound (prepare for lower generation)
        if forecast_type in ("wind", "solar", "wind_mw", "solar_mw"):
            return lower_bound if lower_bound is not None else forecast
        else:  # load or other demand
            return upper_bound if upper_bound is not None else forecast
    
    return forecast


def optimize_robust_dispatch(
    load_forecast: np.ndarray,
    wind_forecast: np.ndarray,
    solar_forecast: np.ndarray,
    load_lower: np.ndarray | None = None,
    load_upper: np.ndarray | None = None,
    wind_lower: np.ndarray | None = None,
    wind_upper: np.ndarray | None = None,
    solar_lower: np.ndarray | None = None,
    solar_upper: np.ndarray | None = None,
    config: RobustDispatchConfig | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Solve robust dispatch optimization with uncertainty-aware forecasts.
    
    Args:
        load_forecast: Load point forecast (MW, horizon steps)
        wind_forecast: Wind point forecast (MW)
        solar_forecast: Solar point forecast (MW)
        load_lower/upper: Load prediction interval bounds
        wind_lower/upper: Wind prediction interval bounds
        solar_lower/upper: Solar prediction interval bounds
        config: Robust dispatch configuration
        verbose: Print optimization details
    
    Returns:
        Dictionary with:
        - battery_charge: Battery charging schedule (MW)
        - battery_discharge: Battery discharging schedule (MW)
        - battery_soc: State of charge trajectory (MWh)
        - grid_import: Grid import schedule (MW)
        - grid_export: Grid export schedule (MW)
        - total_cost: Total cost (EUR)
        - carbon_emissions: Total emissions (tons CO2)
        - feasible: Whether solution is feasible
        - infeasibility_hours: Number of constraint violations
    """
    if config is None:
        config = RobustDispatchConfig()
    
    # Apply uncertainty-aware forecast transformation
    load_adj = apply_forecast_uncertainty(
        load_forecast, load_lower, load_upper, config.mode, "load"
    )
    wind_adj = apply_forecast_uncertainty(
        wind_forecast, wind_lower, wind_upper, config.mode, "wind"
    )
    solar_adj = apply_forecast_uncertainty(
        solar_forecast, solar_lower, solar_upper, config.mode, "solar"
    )
    
    horizon = len(load_adj)
    
    # Net load after renewables
    net_load = load_adj - wind_adj - solar_adj
    net_load = np.maximum(net_load, 0)  # Cannot be negative (excess absorbed by battery/grid)
    
    # Simple heuristic dispatch (can be replaced with LP solver)
    battery_capacity = config.battery_capacity_mwh
    battery_power = config.battery_power_mw
    eta = config.battery_efficiency
    soc = config.battery_initial_soc * battery_capacity
    
    battery_charge = np.zeros(horizon)
    battery_discharge = np.zeros(horizon)
    battery_soc = np.zeros(horizon)
    grid_import = np.zeros(horizon)
    grid_export = np.zeros(horizon)
    
    infeasible_count = 0
    
    for t in range(horizon):
        # Renewable surplus
        surplus = wind_adj[t] + solar_adj[t] - load_adj[t]
        
        if surplus > 0:
            # Excess generation: charge battery or export
            charge_amount = min(surplus, battery_power, (battery_capacity - soc) / eta)
            battery_charge[t] = charge_amount
            soc += charge_amount * eta
            
            remaining = surplus - charge_amount
            if remaining > 0:
                grid_export[t] = min(remaining, config.max_grid_export_mw)
                if remaining > config.max_grid_export_mw:
                    infeasible_count += 1
        else:
            # Deficit: discharge battery or import
            deficit = abs(surplus)
            
            discharge_amount = min(deficit, battery_power, soc)
            battery_discharge[t] = discharge_amount
            soc -= discharge_amount
            
            remaining_deficit = deficit - discharge_amount
            if remaining_deficit > 0:
                grid_import[t] = min(remaining_deficit, config.max_grid_import_mw)
                if remaining_deficit > config.max_grid_import_mw:
                    infeasible_count += 1
        
        battery_soc[t] = soc
    
    # Cost calculation
    import_cost = np.sum(grid_import) * config.grid_buy_price
    export_revenue = np.sum(grid_export) * config.grid_sell_price
    carbon_cost = np.sum(grid_import) * config.carbon_intensity * config.carbon_penalty
    
    total_cost = import_cost - export_revenue + carbon_cost
    total_emissions = np.sum(grid_import) * config.carbon_intensity
    
    if verbose:
        print(f"Robust Dispatch (mode={config.mode}):")
        print(f"  Total cost: €{total_cost:.2f}")
        print(f"  Grid import: {np.sum(grid_import):.1f} MWh")
        print(f"  Grid export: {np.sum(grid_export):.1f} MWh")
        print(f"  Carbon: {total_emissions:.2f} tons CO2")
        print(f"  Infeasible hours: {infeasible_count}/{horizon}")
    
    return {
        "battery_charge": battery_charge,
        "battery_discharge": battery_discharge,
        "battery_soc": battery_soc,
        "grid_import": grid_import,
        "grid_export": grid_export,
        "total_cost": float(total_cost),
        "carbon_emissions": float(total_emissions),
        "feasible": infeasible_count == 0,
        "infeasibility_hours": int(infeasible_count),
        "mode": config.mode,
    }


def evaluate_dispatch_robustness(
    load_true: np.ndarray,
    wind_true: np.ndarray,
    solar_true: np.ndarray,
    load_forecast: np.ndarray,
    wind_forecast: np.ndarray,
    solar_forecast: np.ndarray,
    dispatch_solution: dict[str, Any],
    config: RobustDispatchConfig | None = None,
) -> dict[str, float]:
    """
    Evaluate dispatch solution against true realizations (ex-post).
    
    Computes actual cost, constraint violations, and regret.
    
    Args:
        *_true: True realizations (MW)
        *_forecast: Forecasts used for dispatch (MW)
        dispatch_solution: Solution from optimize_robust_dispatch
        config: Configuration
    
    Returns:
        Dictionary with realized metrics:
        - realized_cost: Actual cost with dispatch applied to true data
        - forecast_error_load/wind/solar: Forecast errors (RMSE)
        - constraint_violations: Number of hours with violations
        - regret: Cost difference vs oracle (perfect foresight)
    """
    if config is None:
        config = RobustDispatchConfig()
    
    horizon = len(load_true)
    
    # Extract dispatch decisions
    battery_charge = dispatch_solution["battery_charge"]
    battery_discharge = dispatch_solution["battery_discharge"]
    
    # Simulate with true data
    soc = config.battery_initial_soc * config.battery_capacity_mwh
    realized_import = np.zeros(horizon)
    realized_export = np.zeros(horizon)
    violations = 0
    
    for t in range(horizon):
        # Net load with true values
        net_true = load_true[t] - wind_true[t] - solar_true[t]
        
        # Apply scheduled battery actions
        soc += battery_charge[t] * config.battery_efficiency
        soc -= battery_discharge[t]
        soc = np.clip(soc, 0, config.battery_capacity_mwh)
        
        # Residual after battery
        residual = net_true - battery_discharge[t] + battery_charge[t]
        
        if residual > 0:
            realized_import[t] = residual
            if residual > config.max_grid_import_mw:
                violations += 1
        else:
            realized_export[t] = abs(residual)
            if abs(residual) > config.max_grid_export_mw:
                violations += 1
    
    # Realized cost
    import_cost = np.sum(realized_import) * config.grid_buy_price
    export_revenue = np.sum(realized_export) * config.grid_sell_price
    carbon_cost = np.sum(realized_import) * config.carbon_intensity * config.carbon_penalty
    realized_cost = import_cost - export_revenue + carbon_cost
    
    # Forecast errors
    load_rmse = float(np.sqrt(np.mean((load_true - load_forecast) ** 2)))
    wind_rmse = float(np.sqrt(np.mean((wind_true - wind_forecast) ** 2)))
    solar_rmse = float(np.sqrt(np.mean((solar_true - solar_forecast) ** 2)))
    
    # Compute oracle (perfect foresight) cost for regret
    oracle_solution = optimize_robust_dispatch(
        load_true, wind_true, solar_true,
        config=config, verbose=False
    )
    oracle_cost = oracle_solution["total_cost"]
    regret = realized_cost - oracle_cost
    
    return {
        "realized_cost": float(realized_cost),
        "oracle_cost": float(oracle_cost),
        "regret": float(regret),
        "regret_pct": float(100 * regret / max(oracle_cost, 1e-6)),
        "forecast_error_load": load_rmse,
        "forecast_error_wind": wind_rmse,
        "forecast_error_solar": solar_rmse,
        "constraint_violations": int(violations),
        "violation_rate": float(violations / horizon),
    }


def run_perturbation_analysis(
    load_forecast: np.ndarray,
    wind_forecast: np.ndarray,
    solar_forecast: np.ndarray,
    load_true: np.ndarray,
    wind_true: np.ndarray,
    solar_true: np.ndarray,
    noise_levels: list[float] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    config: RobustDispatchConfig | None = None,
    n_samples: int = 10,
) -> pd.DataFrame:
    """
    Evaluate dispatch robustness under forecast perturbations.
    
    Adds Gaussian noise to forecasts and measures impact on cost and feasibility.
    
    Args:
        *_forecast: Base forecasts
        *_true: True realizations
        noise_levels: Noise levels as fraction of forecast std (0-1)
        config: Configuration
        n_samples: Monte Carlo samples per noise level
    
    Returns:
        DataFrame with columns:
        - noise_level: Perturbation level
        - sample: Monte Carlo sample index
        - realized_cost: Cost with perturbed forecast
        - regret: Regret vs oracle
        - infeasible_rate: Fraction of constraint violations
    """
    if config is None:
        config = RobustDispatchConfig()
    
    results = []
    
    # Compute forecast standard deviations for noise scaling
    load_std = np.std(load_forecast)
    wind_std = np.std(wind_forecast)
    solar_std = np.std(solar_forecast)
    
    for noise_pct in noise_levels:
        for sample in range(n_samples):
            # Add Gaussian noise
            np.random.seed(42 + sample)
            load_noisy = load_forecast + np.random.normal(0, noise_pct * load_std, len(load_forecast))
            wind_noisy = wind_forecast + np.random.normal(0, noise_pct * wind_std, len(wind_forecast))
            solar_noisy = solar_forecast + np.random.normal(0, noise_pct * solar_std, len(solar_forecast))
            
            # Ensure non-negative
            load_noisy = np.maximum(load_noisy, 0)
            wind_noisy = np.maximum(wind_noisy, 0)
            solar_noisy = np.maximum(solar_noisy, 0)
            
            # Solve dispatch with noisy forecast
            solution = optimize_robust_dispatch(
                load_noisy, wind_noisy, solar_noisy,
                config=config, verbose=False
            )
            
            # Evaluate against true data
            eval_metrics = evaluate_dispatch_robustness(
                load_true, wind_true, solar_true,
                load_noisy, wind_noisy, solar_noisy,
                solution, config
            )
            
            results.append({
                "noise_level": noise_pct,
                "sample": sample,
                "realized_cost": eval_metrics["realized_cost"],
                "regret": eval_metrics["regret"],
                "regret_pct": eval_metrics["regret_pct"],
                "infeasible_rate": eval_metrics["violation_rate"],
            })
    
    return pd.DataFrame(results)
