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
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from scipy.stats import wilcoxon

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
from gridpulse.cpsbench_iot.runner import run_suite as run_cpsbench_suite


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


def _load_dc3s_ablation_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _bootstrap_ci_mean(values: np.ndarray, n_bootstrap: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(42)
    idx = rng.integers(0, vals.size, size=(n_bootstrap, vals.size))
    means = vals[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - (alpha / 2.0)))
    return lo, hi


def _bootstrap_ci_relative_reduction(
    baseline: np.ndarray,
    candidate: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    mask = np.isfinite(base) & np.isfinite(cand) & (base > 1e-9)
    base = base[mask]
    cand = cand[mask]
    if base.size == 0:
        return (float("nan"), float("nan"))
    if base.size == 1:
        rel = float((base[0] - cand[0]) / base[0])
        return (rel, rel)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, base.size, size=(n_bootstrap, base.size))
    rel = (base[idx] - cand[idx]) / np.maximum(base[idx], 1e-9)
    rel_mean = rel.mean(axis=1)
    lo = float(np.quantile(rel_mean, alpha / 2.0))
    hi = float(np.quantile(rel_mean, 1.0 - (alpha / 2.0)))
    return lo, hi


def _wilcoxon_safe(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    """Run Wilcoxon signed-rank safely when vectors are identical or too short."""
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if len(xv) < 2:
        return None, None
    if np.allclose(xv - yv, 0.0, rtol=0.0, atol=1e-12):
        return 0.0, 1.0
    stat, p = wilcoxon(xv, yv)
    return float(stat), float(p)


def _compute_metric_bundle(
    *,
    baseline: np.ndarray,
    candidate: np.ndarray,
    metric: str,
    threshold_rel: float,
    p_threshold: float,
    bootstrap_n: int,
) -> dict[str, float | bool | int | None]:
    b = np.asarray(baseline, dtype=float)
    d = np.asarray(candidate, dtype=float)
    mask = np.isfinite(b) & np.isfinite(d)
    b = b[mask]
    d = d[mask]
    if len(b) == 0:
        return {
            "metric": metric,
            "n_pairs": 0,
            "baseline_mean": np.nan,
            "dc3s_mean": np.nan,
            "dc3s_ci_low": np.nan,
            "dc3s_ci_high": np.nan,
            "rel_reduction_mean": np.nan,
            "rel_reduction_ci_low": np.nan,
            "rel_reduction_ci_high": np.nan,
            "wilcoxon_stat": None,
            "wilcoxon_p": None,
            "passes_10pct": False,
            "passes_15pct": False,
            "target_relative_reduction": float(threshold_rel),
            "passes_target": False,
            "passes_p01": False,
            "passes_metric": False,
        }

    rel_reduction = (b - d) / np.maximum(b, 1e-9)
    ci_lo, ci_hi = _bootstrap_ci_mean(d, n_bootstrap=bootstrap_n)
    rel_lo, rel_hi = _bootstrap_ci_relative_reduction(b, d, n_bootstrap=bootstrap_n)
    stat, p = _wilcoxon_safe(pd.Series(b), pd.Series(d))
    mean_rel = float(np.mean(rel_reduction))
    passes_rel = bool(mean_rel >= float(threshold_rel))
    passes_sig = bool((p is not None) and (p < float(p_threshold)))
    return {
        "metric": metric,
        "n_pairs": int(len(b)),
        "baseline_mean": float(np.mean(b)),
        "dc3s_mean": float(np.mean(d)),
        "dc3s_ci_low": float(ci_lo),
        "dc3s_ci_high": float(ci_hi),
        "rel_reduction_mean": float(mean_rel),
        "rel_reduction_ci_low": float(rel_lo),
        "rel_reduction_ci_high": float(rel_hi),
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
        "passes_10pct": bool(mean_rel >= 0.10),
        "passes_15pct": bool(mean_rel >= 0.15),
        "target_relative_reduction": float(threshold_rel),
        "passes_target": passes_rel,
        "passes_p01": passes_sig,
        "passes_metric": bool(passes_rel and passes_sig),
    }


def run_dc3s_ablation_matrix(
    *,
    output_dir: Path,
    seeds: list[int],
    scenario: str = "drift_combo",
    horizon: int = 96,
    cfg_path: Path = Path("configs/dc3s_ablations.yaml"),
    precomputed_main_csv: Path | None = None,
    precomputed_sweep_csv: Path | None = None,
) -> dict:
    cfg = _load_dc3s_ablation_cfg(cfg_path)
    if len(seeds) != 10:
        raise ValueError("Task-2 protocol requires exactly 10 seeds")

    scenarios = cfg.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        scenarios = [str(scenario), "dropout", "drift_combo"]
    scenarios = list(dict.fromkeys(str(s) for s in scenarios))

    output_dir.mkdir(parents=True, exist_ok=True)
    use_precomputed = (
        precomputed_main_csv is not None
        and precomputed_sweep_csv is not None
        and Path(precomputed_main_csv).exists()
        and Path(precomputed_sweep_csv).exists()
    )
    if use_precomputed:
        main_df = pd.read_csv(Path(precomputed_main_csv))
        sweep_df = pd.read_csv(Path(precomputed_sweep_csv))
    else:
        suite_out = Path(tempfile.mkdtemp(prefix="cpsbench_ablation_", dir=str(output_dir)))
        try:
            run_cpsbench_suite(
                scenarios=scenarios,
                seeds=seeds,
                out_dir=suite_out,
                horizon=int(horizon),
            )
            main_df = pd.read_csv(suite_out / "dc3s_main_table.csv")
            sweep_df = pd.read_csv(suite_out / "cpsbench_merged_sweep.csv")
        finally:
            shutil.rmtree(suite_out, ignore_errors=True)

    sev_col = "true_soc_violation_severity_p95_mwh"
    if sev_col not in main_df.columns:
        sev_col = "true_soc_violation_severity_p95"
    if sev_col not in sweep_df.columns:
        sev_col = "true_soc_violation_severity_p95"

    primary_cfg = cfg.get("primary_gate", {}) if isinstance(cfg.get("primary_gate"), dict) else {}
    p_threshold = float(primary_cfg.get("p_threshold", 0.01))
    threshold_rel = float(primary_cfg.get("min_relative_reduction", 0.10))
    primary_baseline = str(primary_cfg.get("baseline", "robust_fixed_interval"))
    candidate = str(primary_cfg.get("candidate", "dc3s_wrapped"))
    faulted_only = bool(primary_cfg.get("faulted_only", True))
    severity_min = float(primary_cfg.get("severity_min", 0.0))
    enforce_primary_gate = bool(cfg.get("enforce_primary_gate", True))
    bootstrap_n = int(cfg.get("bootstrap_n", 10000))

    metrics = ("true_soc_violation_rate", sev_col)
    summary_rows: list[dict[str, object]] = []
    stats_summary: dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat(),
        "scenarios": scenarios,
        "seeds": seeds,
        "wilcoxon_threshold_p": p_threshold,
        "bootstrap_ci": 0.95,
        "min_relative_reduction": threshold_rel,
        "comparisons": {},
    }

    # Primary gate: aggregate faulted sweep points.
    pair_cols = ["fault_dimension", "severity", "scenario", "seed"]
    base_sweep = sweep_df[sweep_df["controller"] == primary_baseline]
    cand_sweep = sweep_df[sweep_df["controller"] == candidate]
    primary_pairs = base_sweep.merge(cand_sweep, on=pair_cols, suffixes=("_baseline", "_dc3s"), how="inner")
    if faulted_only:
        sev_vals = pd.to_numeric(primary_pairs["severity"], errors="coerce")
        primary_pairs = primary_pairs[sev_vals > severity_min].copy()

    primary_metrics: dict[str, dict[str, object]] = {}
    primary_pass = True
    primary_row: dict[str, object] = {
        "analysis_scope": "primary_aggregate_fault_sweep",
        "baseline_role": "primary",
        "scenario": "aggregate_fault_sweep",
        "fault_dimension": "all",
        "baseline_controller": primary_baseline,
        "candidate_controller": candidate,
        "n_pairs": int(len(primary_pairs)),
    }
    for metric in metrics:
        bundle = _compute_metric_bundle(
            baseline=primary_pairs[f"{metric}_baseline"].to_numpy(dtype=float) if len(primary_pairs) else np.asarray([], dtype=float),
            candidate=primary_pairs[f"{metric}_dc3s"].to_numpy(dtype=float) if len(primary_pairs) else np.asarray([], dtype=float),
            metric=metric,
            threshold_rel=threshold_rel,
            p_threshold=p_threshold,
            bootstrap_n=bootstrap_n,
        )
        metric_key = metric.replace("_mwh", "")
        primary_row[f"{metric_key}_baseline_mean"] = bundle["baseline_mean"]
        primary_row[f"{metric_key}_dc3s_mean"] = bundle["dc3s_mean"]
        primary_row[f"{metric_key}_dc3s_ci_low"] = bundle["dc3s_ci_low"]
        primary_row[f"{metric_key}_dc3s_ci_high"] = bundle["dc3s_ci_high"]
        primary_row[f"{metric_key}_rel_reduction"] = bundle["rel_reduction_mean"]
        primary_row[f"{metric_key}_rel_reduction_ci_low"] = bundle["rel_reduction_ci_low"]
        primary_row[f"{metric_key}_rel_reduction_ci_high"] = bundle["rel_reduction_ci_high"]
        primary_row[f"{metric_key}_wilcoxon_stat"] = bundle["wilcoxon_stat"]
        primary_row[f"{metric_key}_wilcoxon_p"] = bundle["wilcoxon_p"]
        primary_row[f"{metric_key}_passes_10pct"] = bundle["passes_10pct"]
        primary_row[f"{metric_key}_passes_15pct"] = bundle["passes_15pct"]
        primary_row[f"{metric_key}_target_relative_reduction"] = bundle["target_relative_reduction"]
        primary_row[f"{metric_key}_passes_target"] = bundle["passes_target"]
        primary_row[f"{metric_key}_passes_p01"] = bundle["passes_p01"]
        primary_pass = bool(primary_pass and bundle["passes_metric"])
        primary_metrics[metric] = dict(bundle)
    primary_row["passes_all_thresholds"] = bool(primary_pass)
    summary_rows.append(primary_row)
    stats_summary["primary_gate"] = {
        "scope": "aggregate_fault_sweep",
        "faulted_only": bool(faulted_only),
        "severity_min": float(severity_min),
        "baseline_controller": primary_baseline,
        "candidate_controller": candidate,
        "n_pairs": int(len(primary_pairs)),
        "passes_all_thresholds": bool(primary_pass),
        "metrics": primary_metrics,
    }

    # Secondary diagnostics: per-fault and per-scenario.
    sec_cfg = cfg.get("secondary_diagnostics", {}) if isinstance(cfg.get("secondary_diagnostics"), dict) else {}
    baselines = sec_cfg.get("baselines", ["robust_fixed_interval", "deterministic_lp"])
    include_per_fault = bool(sec_cfg.get("include_per_fault", True))
    for baseline in [str(b) for b in baselines]:
        base_df = main_df[main_df["controller"] == baseline]
        dc3s_df = main_df[main_df["controller"] == candidate]
        merged = base_df.merge(
            dc3s_df,
            on=["scenario", "seed"],
            suffixes=("_baseline", "_dc3s"),
            how="inner",
        )
        for sc in scenarios:
            sub = merged[merged["scenario"] == sc]
            if sub.empty:
                continue

            row: dict[str, object] = {
                "analysis_scope": "secondary_scenario",
                "baseline_role": "secondary",
                "scenario": sc,
                "fault_dimension": "aggregate",
                "baseline_controller": baseline,
                "candidate_controller": candidate,
                "n_pairs": int(len(sub)),
            }
            stats_summary["comparisons"].setdefault(sc, {})[baseline] = {}
            overall_pass = True

            for metric in metrics:
                bundle = _compute_metric_bundle(
                    baseline=pd.to_numeric(sub[f"{metric}_baseline"], errors="coerce").to_numpy(dtype=float),
                    candidate=pd.to_numeric(sub[f"{metric}_dc3s"], errors="coerce").to_numpy(dtype=float),
                    metric=metric,
                    threshold_rel=threshold_rel,
                    p_threshold=p_threshold,
                    bootstrap_n=bootstrap_n,
                )
                overall_pass = bool(overall_pass and bundle["passes_metric"])
                metric_key = metric.replace("_mwh", "")
                row[f"{metric_key}_baseline_mean"] = bundle["baseline_mean"]
                row[f"{metric_key}_dc3s_mean"] = bundle["dc3s_mean"]
                row[f"{metric_key}_dc3s_ci_low"] = bundle["dc3s_ci_low"]
                row[f"{metric_key}_dc3s_ci_high"] = bundle["dc3s_ci_high"]
                row[f"{metric_key}_rel_reduction"] = bundle["rel_reduction_mean"]
                row[f"{metric_key}_rel_reduction_ci_low"] = bundle["rel_reduction_ci_low"]
                row[f"{metric_key}_rel_reduction_ci_high"] = bundle["rel_reduction_ci_high"]
                row[f"{metric_key}_wilcoxon_stat"] = bundle["wilcoxon_stat"]
                row[f"{metric_key}_wilcoxon_p"] = bundle["wilcoxon_p"]
                row[f"{metric_key}_passes_10pct"] = bundle["passes_10pct"]
                row[f"{metric_key}_passes_15pct"] = bundle["passes_15pct"]
                row[f"{metric_key}_target_relative_reduction"] = bundle["target_relative_reduction"]
                row[f"{metric_key}_passes_target"] = bundle["passes_target"]
                row[f"{metric_key}_passes_p01"] = bundle["passes_p01"]

                stats_summary["comparisons"][sc][baseline][metric] = {
                    "n_pairs": bundle["n_pairs"],
                    "baseline_mean": bundle["baseline_mean"],
                    "dc3s_mean": bundle["dc3s_mean"],
                    "rel_reduction_mean": bundle["rel_reduction_mean"],
                    "rel_reduction_ci_low": bundle["rel_reduction_ci_low"],
                    "rel_reduction_ci_high": bundle["rel_reduction_ci_high"],
                    "wilcoxon_stat": bundle["wilcoxon_stat"],
                    "wilcoxon_p": bundle["wilcoxon_p"],
                    "passes_10pct": bundle["passes_10pct"],
                    "passes_15pct": bundle["passes_15pct"],
                    "target_relative_reduction": bundle["target_relative_reduction"],
                    "passes_target": bundle["passes_target"],
                    "passes_p01": bundle["passes_p01"],
                }

            row["passes_all_thresholds"] = bool(overall_pass)
            summary_rows.append(row)

        if include_per_fault:
            baseline_sweep = sweep_df[sweep_df["controller"] == baseline]
            candidate_sweep = sweep_df[sweep_df["controller"] == candidate]
            merged_fault = baseline_sweep.merge(candidate_sweep, on=pair_cols, suffixes=("_baseline", "_dc3s"), how="inner")
            if faulted_only:
                sev_vals = pd.to_numeric(merged_fault["severity"], errors="coerce")
                merged_fault = merged_fault[sev_vals > severity_min].copy()
            for fault_dim, sub_fault in merged_fault.groupby("fault_dimension", sort=True):
                row_fault: dict[str, object] = {
                    "analysis_scope": "secondary_fault_dimension",
                    "baseline_role": "secondary",
                    "scenario": "aggregate_fault_sweep",
                    "fault_dimension": str(fault_dim),
                    "baseline_controller": baseline,
                    "candidate_controller": candidate,
                    "n_pairs": int(len(sub_fault)),
                }
                fault_pass = True
                for metric in metrics:
                    bundle = _compute_metric_bundle(
                        baseline=sub_fault[f"{metric}_baseline"].to_numpy(dtype=float),
                        candidate=sub_fault[f"{metric}_dc3s"].to_numpy(dtype=float),
                        metric=metric,
                        threshold_rel=threshold_rel,
                        p_threshold=p_threshold,
                        bootstrap_n=bootstrap_n,
                    )
                    metric_key = metric.replace("_mwh", "")
                    row_fault[f"{metric_key}_baseline_mean"] = bundle["baseline_mean"]
                    row_fault[f"{metric_key}_dc3s_mean"] = bundle["dc3s_mean"]
                    row_fault[f"{metric_key}_dc3s_ci_low"] = bundle["dc3s_ci_low"]
                    row_fault[f"{metric_key}_dc3s_ci_high"] = bundle["dc3s_ci_high"]
                    row_fault[f"{metric_key}_rel_reduction"] = bundle["rel_reduction_mean"]
                    row_fault[f"{metric_key}_rel_reduction_ci_low"] = bundle["rel_reduction_ci_low"]
                    row_fault[f"{metric_key}_rel_reduction_ci_high"] = bundle["rel_reduction_ci_high"]
                    row_fault[f"{metric_key}_wilcoxon_stat"] = bundle["wilcoxon_stat"]
                    row_fault[f"{metric_key}_wilcoxon_p"] = bundle["wilcoxon_p"]
                    row_fault[f"{metric_key}_passes_10pct"] = bundle["passes_10pct"]
                    row_fault[f"{metric_key}_passes_15pct"] = bundle["passes_15pct"]
                    row_fault[f"{metric_key}_target_relative_reduction"] = bundle["target_relative_reduction"]
                    row_fault[f"{metric_key}_passes_target"] = bundle["passes_target"]
                    row_fault[f"{metric_key}_passes_p01"] = bundle["passes_p01"]
                    fault_pass = bool(fault_pass and bundle["passes_metric"])
                row_fault["passes_all_thresholds"] = bool(fault_pass)
                summary_rows.append(row_fault)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["baseline_role", "analysis_scope", "scenario", "fault_dimension", "baseline_controller"]
        ).reset_index(drop=True)
    table2_path = output_dir / "table2_ablations.csv"
    summary_df.to_csv(table2_path, index=False, float_format="%.6f")

    stats_path = output_dir / "stats_summary.json"
    stats_summary["table2_ablations"] = str(table2_path)
    stats_summary["overall_pass"] = bool(primary_pass)
    stats_path.write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")

    publication_dir = Path("reports/publication")
    publication_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(publication_dir / "table2_ablations.csv", index=False, float_format="%.6f")
    (publication_dir / "stats_summary.json").write_text(stats_path.read_text(encoding="utf-8"), encoding="utf-8")

    result = {
        "rows": int(len(summary_df)),
        "scenarios": scenarios,
        "table2_path": str(table2_path),
        "stats_path": str(stats_path),
        "overall_pass": bool(primary_pass),
    }
    if enforce_primary_gate and not primary_pass:
        raise RuntimeError(
            f"Primary aggregate robust gate failed: expected p < {p_threshold:.3f} and >="
            f"{100.0 * threshold_rel:.1f}% relative reduction for both violation rate and severity metrics."
        )
    return result


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run GridPulse ablation study")
    parser.add_argument(
        "--dc3s",
        action="store_true",
        help="Run DC3S CPSBench ablation matrix instead of legacy optimizer ablations",
    )
    parser.add_argument("--scenario", type=str, default="drift_combo", help="CPSBench scenario for --dc3s mode")
    parser.add_argument("--horizon", type=int, default=96, help="CPSBench horizon for --dc3s mode")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seed list for --dc3s mode (default: 0..9)",
    )
    parser.add_argument(
        "--dc3s-config",
        type=Path,
        default=Path("configs/dc3s_ablations.yaml"),
        help="DC3S ablation config path",
    )
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

    if args.dc3s:
        seeds = list(args.seeds or list(range(10)))
        summary = run_dc3s_ablation_matrix(
            output_dir=args.output,
            seeds=seeds,
            scenario=str(args.scenario),
            horizon=int(args.horizon),
            cfg_path=args.dc3s_config,
        )
        print(json.dumps(summary, indent=2))
        return
    
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
