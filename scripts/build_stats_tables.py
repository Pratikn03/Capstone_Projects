#!/usr/bin/env python3
"""
Build publication-ready statistical tables from GridPulse results.

Generates LaTeX and CSV tables for:
- Ablation study comparisons
- Cross-region performance
- Robustness under perturbations
- Statistical significance tests
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_ablation_results(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load ablation study results."""
    results_csv = results_dir / "ablation_results.csv"
    stats_json = results_dir / "ablation_stats.json"
    
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing: {results_csv}")
    
    df = pd.read_csv(results_csv)
    
    stats = {}
    if stats_json.exists():
        stats = json.loads(stats_json.read_text())
    
    return df, stats


def create_ablation_latex_table(
    results_df: pd.DataFrame,
    stats: dict,
    output_path: Path,
) -> None:
    """
    Create LaTeX table for ablation study.
    
    Format:
    Scenario | Cost (mean ± std) | 95% CI | Regret (%) | p-value vs Full
    """
    rows = []
    
    # Group by scenario
    for scenario in ["Full System", "No Uncertainty", "No Carbon", "Forecast Only"]:
        mask = results_df["scenario"] == scenario
        scenario_data = results_df[mask]
        
        costs = scenario_data["total_cost"].dropna().values
        
        if len(costs) == 0:
            rows.append({
                "Scenario": scenario.replace("_", " "),
                "Cost (mean ± std)": "—",
                "95\\% CI": "—",
                "Regret (\\%)": "—",
                "p-value": "—",
            })
            continue
        
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        ci_lower = np.percentile(costs, 2.5)
        ci_upper = np.percentile(costs, 97.5)
        mean_regret = scenario_data["regret_pct"].dropna().mean()
        
        # Get p-value from statistical comparison
        p_value = "—"
        if scenario != "Full System" and scenario in stats:
            p_val = stats[scenario]["comparison"]["p_value"]
            if p_val < 0.001:
                p_value = "< 0.001"
            else:
                p_value = f"{p_val:.3f}"
        
        rows.append({
            "Scenario": scenario.replace("_", " "),
            "Cost (mean ± std)": f"€{mean_cost:.1f} ± {std_cost:.1f}",
            "95\\% CI": f"[{ci_lower:.1f}, {ci_upper:.1f}]",
            "Regret (\\%)": f"{mean_regret:.2f}" if not np.isnan(mean_regret) else "—",
            "p-value": p_value,
        })
    
    df_table = pd.DataFrame(rows)
    
    # Save as LaTeX
    latex_str = df_table.to_latex(
        index=False,
        escape=False,
        column_format="lrrrr",
        caption="Ablation study results comparing system configurations.",
        label="tab:ablation",
    )
    
    output_path.write_text(latex_str, encoding="utf-8")
    print(f"Saved LaTeX table: {output_path}")
    
    # Also save as CSV
    csv_path = output_path.with_suffix(".csv")
    df_table.to_csv(csv_path, index=False)
    print(f"Saved CSV table: {csv_path}")


def create_robustness_table(
    perturbation_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create table showing robustness under forecast perturbations.
    
    Format:
    Noise Level | Mean Regret (%) | Std Regret | Infeasible Rate
    """
    if perturbation_results.empty:
        print("No perturbation results available")
        return
    
    # Group by noise level
    grouped = perturbation_results.groupby("noise_level")
    
    rows = []
    for noise, group in grouped:
        rows.append({
            "Noise Level": f"{int(noise * 100)}\\%",
            "Mean Regret (\\%)": f"{group['regret_pct'].mean():.2f}",
            "Std Regret": f"{group['regret_pct'].std():.2f}",
            "Infeasible Rate": f"{group['infeasible_rate'].mean():.3f}",
        })
    
    df_table = pd.DataFrame(rows)
    
    # LaTeX
    latex_str = df_table.to_latex(
        index=False,
        escape=False,
        column_format="lrrr",
        caption="Robustness analysis under forecast noise perturbations.",
        label="tab:robustness",
    )
    
    output_path.write_text(latex_str, encoding="utf-8")
    print(f"Saved robustness table: {output_path}")


def create_stats_summary_table(
    comparison: dict,
    output_path: Path,
) -> None:
    """
    Create summary table for statistical comparison.
    
    Shows mean, CI, p-value, and effect size.
    """
    baseline_name = comparison["system_names"][0]
    treatment_name = comparison["system_names"][1]
    
    rows = []
    
    # Mean ± std
    rows.append({
        "Metric": "Cost (mean ± std)",
        baseline_name: f"€{comparison['baseline']['mean']:.2f} ± {comparison['baseline']['std']:.2f}",
        treatment_name: f"€{comparison['treatment']['mean']:.2f} ± {comparison['treatment']['std']:.2f}",
    })
    
    # 95% CI
    rows.append({
        "Metric": "95\\% CI",
        baseline_name: f"[{comparison['baseline']['ci_lower']:.2f}, {comparison['baseline']['ci_upper']:.2f}]",
        treatment_name: f"[{comparison['treatment']['ci_lower']:.2f}, {comparison['treatment']['ci_upper']:.2f}]",
    })
   # Cost reduction
    rows.append({
        "Metric": "Cost Reduction",
        baseline_name: "—",
        treatment_name: f"{comparison['comparison']['cost_reduction_pct']:.1f}\\%",
    })
    
    # Statistical test
    p_val = comparison['comparison']['p_value']
    p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
    
    rows.append({
        "Metric": "p-value",
        baseline_name: "—",
        treatment_name: p_str,
    })
    
    # Effect size
    rows.append({
        "Metric": "Cohen's d",
        baseline_name: "—",
        treatment_name: f"{comparison['comparison']['cohens_d']:.3f}",
    })
    
    df_table = pd.DataFrame(rows)
    
    # LaTeX
    latex_str = df_table.to_latex(
        index=False,
        escape=False,
        column_format="lrr",
        caption=f"Statistical comparison: {baseline_name} vs {treatment_name}.",
        label="tab:stats_comparison",
    )
    
    output_path.write_text(latex_str, encoding="utf-8")
    print(f"Saved stats comparison table: {output_path}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build statistical tables")
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("reports/ablations"),
        help="Ablation results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/tables"),
        help="Output directory for tables",
    )
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Building publication tables...")
    
    # Ablation tables
    if args.ablation_dir.exists():
        try:
            results_df, stats = load_ablation_results(args.ablation_dir)
            
            create_ablation_latex_table(
                results_df,
                stats,
                args.output_dir / "ablation_table.tex",
            )
        except Exception as e:
            print(f"Could not build ablation table: {e}")
    
    print(f"\nTables saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
