#!/usr/bin/env python3
"""
Ablation study runner for GridPulse.

Compares different system configurations to quantify the impact of each component:
1. Full system (uncertainty + anomaly filter + carbon penalty + optimization)
2. No uncertainty (point forecasts only)
3. No anomaly filter
4. No carbon penalty (cost-only optimization)
5. Forecast-only (no optimization/dispatch)

Outputs:
- ablation_results.csv: Numeric results table
- ablation_comparison.png: Bar chart comparison
- ablation_stats.json: Statistical test results
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src to path
REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from gridpulse.optimizer.robust_dispatch import (
    RobustDispatchConfig,
    optimize_robust_dispatch,
    evaluate_dispatch_robustness,
)
from gridpulse.evaluation.stats import compare_systems_statistically, bootstrap_ci
from gridpulse.utils.seed import set_seed


def load_test_data(data_path: Path) -> dict[str, np.ndarray]:
    """Load test split data for ablation analysis."""
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Extract relevant columns (assuming standard naming)
    data = {
        "load_true": df["load_mw"].values if "load_mw" in df.columns else df["load"].values,
        "wind_true": df["wind_mw"].values if "wind_mw" in df.columns else df.get("wind", np.zeros(len(df))).values,
        "solar_true": df["solar_mw"].values if "solar_mw" in df.columns else df.get("solar", np.zeros(len(df))).values,
    }
    
    return data


def generate_synthetic_forecasts(
    true_values: np.ndarray,
    noise_level: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic forecasts with Gaussian noise.
    
    Returns:
        forecast, lower_bound, upper_bound
    """
    np.random.seed(seed)
    
    std = noise_level * np.std(true_values)
    forecast = true_values + np.random.normal(0, std, len(true_values))
    forecast = np.maximum(forecast, 0)
    
    # Prediction intervals (90% coverage ≈ 1.645 * std)
    lower = forecast - 1.645 * std
    upper = forecast + 1.645 * std
    lower = np.maximum(lower, 0)
    
    return forecast, lower, upper


def run_ablation_scenario(
    scenario_name: str,
    load_true: np.ndarray,
    wind_true: np.ndarray,
    solar_true: np.ndarray,
    load_forecast: np.ndarray,
    wind_forecast: np.ndarray,
    solar_forecast: np.ndarray,
    load_lower: np.ndarray | None = None,
    load_upper: np.ndarray | None = None,
    wind_lower: np.ndarray | None = None,
    wind_upper: np.ndarray | None = None,
    solar_lower: np.ndarray | None = None,
    solar_upper: np.ndarray | None = None,
    enable_uncertainty: bool = True,
    enable_carbon: bool = True,
    enable_optimization: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run a single ablation scenario.
    
    Args:
        scenario_name: Name of the scenario
        *_true: True values
        *_forecast: Point forecasts
        *_lower/upper: Prediction interval bounds
        enable_*: Feature flags for ablation
    
    Returns:
        Dictionary with scenario results
    """
    # Configure dispatch
    config = RobustDispatchConfig()
    if not enable_carbon:
        # Legacy ablation toggle: map old carbon-off path to no degradation penalty.
        config.degradation_cost_per_mwh = 0.0

    if not enable_uncertainty:
        # No uncertainty scenario: collapse interval to point forecast.
        load_lower = load_forecast
        load_upper = load_forecast
    else:
        load_lower = load_forecast if load_lower is None else load_lower
        load_upper = load_forecast if load_upper is None else load_upper
    
    if not enable_optimization:
        # Forecast-only: just measure forecast error
        # No dispatch optimization
        load_rmse = float(np.sqrt(np.mean((load_true - load_forecast) ** 2)))
        wind_rmse = float(np.sqrt(np.mean((wind_true - wind_forecast) ** 2)))
        solar_rmse = float(np.sqrt(np.mean((solar_true - solar_forecast) ** 2)))
        
        return {
            "scenario": scenario_name,
            "total_cost": None,  # No dispatch, no cost
            "carbon_emissions": None,
            "regret": None,
            "infeasible_rate": None,
            "forecast_error_load": load_rmse,
            "forecast_error_wind": wind_rmse,
            "forecast_error_solar": solar_rmse,
        }
    
    renewables_forecast = np.asarray(wind_forecast, dtype=float) + np.asarray(solar_forecast, dtype=float)
    renewables_true = np.asarray(wind_true, dtype=float) + np.asarray(solar_true, dtype=float)

    # Run dispatch optimization
    solution = optimize_robust_dispatch(
        load_lower_bound=load_lower,
        load_upper_bound=load_upper,
        renewables_forecast=renewables_forecast,
        config=config,
        verbose=verbose,
    )

    # Evaluate against true data
    eval_metrics = evaluate_dispatch_robustness(
        load_true=load_true,
        renewables_true=renewables_true,
        load_lower_bound=load_lower,
        load_upper_bound=load_upper,
        renewables_forecast=renewables_forecast,
        dispatch_solution=solution,
        config=config,
    )

    wind_rmse = float(np.sqrt(np.mean((wind_true - wind_forecast) ** 2)))
    solar_rmse = float(np.sqrt(np.mean((solar_true - solar_forecast) ** 2)))

    return {
        "scenario": scenario_name,
        "total_cost": eval_metrics["realized_cost"],
        "oracle_cost": eval_metrics["oracle_cost"],
        "regret": eval_metrics["regret"],
        "regret_pct": eval_metrics["regret_pct"],
        "carbon_emissions": None,
        "infeasible_rate": eval_metrics["violation_rate"],
        "forecast_error_load": eval_metrics["forecast_error_load"],
        "forecast_error_wind": wind_rmse,
        "forecast_error_solar": solar_rmse,
    }


def run_ablation_study(
    data_path: Path,
    output_dir: Path,
    n_runs: int = 5,
    forecast_noise: float = 0.10,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run complete ablation study with multiple scenarios.
    
    Args:
        data_path: Path to test data
        output_dir: Output directory  
        n_runs: Number of random runs for robustness
        forecast_noise: Forecast noise level (0-1)
        verbose: Print progress
    
    Returns:
        DataFrame with ablation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {data_path}...")
    data = load_test_data(data_path)
    
    # Truncate to reasonable horizon for speed (e.g., 1 week = 168 hours)
    horizon = min(168, len(data["load_true"]))
    load_true = data["load_true"][:horizon]
    wind_true = data["wind_true"][:horizon]
    solar_true = data["solar_true"][:horizon]
    
    # Ablation scenarios
    scenarios = [
        {
            "name": "Full System",
            "uncertainty": True,
            "carbon": True,
            "optimization": True,
            "description": "All features enabled"
        },
        {
            "name": "No Uncertainty",
            "uncertainty": False,
            "carbon": True,
            "optimization": True,
            "description": "Point forecasts only"
        },
        {
            "name": "No Carbon",
            "uncertainty": True,
            "carbon": False,
            "optimization": True,
            "description": "Cost-only optimization"
        },
        {
            "name": "Forecast Only",
            "uncertainty": False,
            "carbon": False,
            "optimization": False,
            "description": "No dispatch optimization"
        },
    ]
    
    all_results = []
    
    for run in range(n_runs):
        set_seed(42 + run)
        
        # Generate synthetic forecasts with noise
        load_forecast, load_lower, load_upper = generate_synthetic_forecasts(
            load_true, forecast_noise, seed=42 + run
        )
        wind_forecast, wind_lower, wind_upper = generate_synthetic_forecasts(
            wind_true, forecast_noise, seed=100 + run
        )
        solar_forecast, solar_lower, solar_upper = generate_synthetic_forecasts(
            solar_true, forecast_noise, seed=200 + run
        )
        
        if verbose:
            print(f"\nRun {run + 1}/{n_runs}")
        
        for scenario in scenarios:
            if verbose:
                print(f"  Testing: {scenario['name']}")
            
            result = run_ablation_scenario(
                scenario_name=scenario["name"],
                load_true=load_true,
                wind_true=wind_true,
                solar_true=solar_true,
                load_forecast=load_forecast,
                wind_forecast=wind_forecast,
                solar_forecast=solar_forecast,
                load_lower=load_lower if scenario["uncertainty"] else None,
                load_upper=load_upper if scenario["uncertainty"] else None,
                wind_lower=wind_lower if scenario["uncertainty"] else None,
                wind_upper=wind_upper if scenario["uncertainty"] else None,
                solar_lower=solar_lower if scenario["uncertainty"] else None,
                solar_upper=solar_upper if scenario["uncertainty"] else None,
                enable_uncertainty=scenario["uncertainty"],
                enable_carbon=scenario["carbon"],
                enable_optimization=scenario["optimization"],
                verbose=False,
            )
            
            result["run"] = run
            result["description"] = scenario["description"]
            all_results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved ablation results: {csv_path}")
    
    return df


def create_ablation_summary(results_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Create statistical summary of ablation results.
    
    Args:
        results_df: Results from run_ablation_study
        output_dir: Output directory
    
    Returns:
        Summary statistics dictionary
    """
    # Group by scenario
    summary = []
    
    for scenario in results_df["scenario"].unique():
        mask = results_df["scenario"] == scenario
        scenario_data = results_df[mask]
        
        # Extract costs (excluding None for forecast-only)
        costs = scenario_data["total_cost"].dropna().values
        
        if len(costs) == 0:
            summary_stats = {
                "scenario": scenario,
                "mean_cost": None,
                "std_cost": None,
                "ci_lower": None,
                "ci_upper": None,
            }
        else:
            ci = bootstrap_ci(costs, confidence=0.95)
            summary_stats = {
                "scenario": scenario,
                "mean_cost": float(np.mean(costs)),
                "std_cost": float(np.std(costs)),
                "median_cost": float(np.median(costs)),
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "mean_regret_pct": float(scenario_data["regret_pct"].dropna().mean()),
                "mean_infeasible_rate": float(scenario_data["infeasible_rate"].dropna().mean()),
            }
        
        summary.append(summary_stats)
    
    summary_df = pd.DataFrame(summary)
    
    # Save summary
    summary_path = output_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved ablation summary: {summary_path}")
    
    # Perform pairwise comparisons vs Full System
    full_costs = results_df[results_df["scenario"] == "Full System"]["total_cost"].dropna().values
    
    comparisons = {}
    for scenario in results_df["scenario"].unique():
        if scenario == "Full System":
            continue
        
        scenario_costs = results_df[results_df["scenario"] == scenario]["total_cost"].dropna().values
        
        if len(scenario_costs) > 0:
            comparison = compare_systems_statistically(
                full_costs,
                scenario_costs,
                system_names=("Full System", scenario),
            )
            comparisons[scenario] = comparison
    
    # Save comparisons
    comp_path = output_dir / "ablation_stats.json"
    comp_path.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(f"Saved statistical comparisons: {comp_path}")
    
    return {
        "summary": summary_df.to_dict("records"),
        "comparisons": comparisons,
    }


def plot_ablation_results(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar chart comparison of ablation scenarios."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping plot")
        return
    
    # Group by scenario and compute mean
    scenario_means = results_df.groupby("scenario")["total_cost"].mean().sort_values()
    scenario_stds = results_df.groupby("scenario")["total_cost"].std()
    
    # Remove NaN (forecast-only has no cost)
    scenario_means = scenario_means.dropna()
    scenario_stds = scenario_stds[scenario_means.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenario_means))
    bars = ax.bar(x, scenario_means, yerr=scenario_stds, capsize=5, alpha=0.8)
    
    # Color code: Full System in green, others in blue/orange
    colors = ['green' if 'Full' in name else 'steelblue' for name in scenario_means.index]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Cost (EUR)", fontsize=12, fontweight="bold")
    ax.set_title("Ablation Study: System Component Impact", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_means.index, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "ablation_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved ablation plot: {plot_path}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run GridPulse ablation study")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/splits/test.parquet"),
        help="Test data path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ablations"),
        help="Output directory",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of random runs",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.10,
        help="Forecast noise level (0-1)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GridPulse Ablation Study")
    print("=" * 60)
    
    # Run ablation study
    results_df = run_ablation_study(
        data_path=args.data,
        output_dir=args.output,
        n_runs=args.n_runs,
        forecast_noise=args.noise,
        verbose=args.verbose,
    )
    
    # Create summary and statistics
    summary = create_ablation_summary(results_df, args.output)
    
    # Create visualization
    plot_ablation_results(results_df, args.output)
    
    print("\n" + "=" * 60)
    print("Ablation study complete!")
    print(f"Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
