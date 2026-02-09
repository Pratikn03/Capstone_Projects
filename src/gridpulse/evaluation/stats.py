"""
Statistical analysis tools for GridPulse evaluation.

Provides bootstrap confidence intervals, paired statistical tests,
and time-fold cross-validation for robust model comparison.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Sample data (1D array)
        statistic: Function to compute (default: mean)
        confidence: Confidence level (0-1)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with:
        - point_estimate: Statistic on original data
        - ci_lower: Lower CI bound
        - ci_upper: Upper CI bound
        - std_error: Bootstrap standard error
    """
    np.random.seed(seed)
    
    data = np.asarray(data)
    n = len(data)
    
    # Compute point estimate
    point_est = statistic(data)
    
    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(resample)
    
    # Compute CI using percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    # Standard error
    std_err = np.std(bootstrap_stats)
    
    return {
        "point_estimate": float(point_est),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std_error": float(std_err),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def paired_test(
    baseline: np.ndarray,
    treatment: np.ndarray,
    test: str = "wilcoxon",
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """
    Perform paired statistical test between baseline and treatment.
    
    Args:
        baseline: Baseline system performance (e.g., costs)
        treatment: Treatment system performance
        test: "wilcoxon" (non-parametric) or "ttest" (parametric)
        alternative: "two-sided", "less", "greater"
    
    Returns:
        Dictionary with:
        - statistic: Test statistic
        - p_value: P-value
        - significant: Whether p < 0.05
        - effect_size: Mean difference
        - effect_size_pct: Percentage improvement
    """
    baseline = np.asarray(baseline)
    treatment = np.asarray(treatment)
    
    if len(baseline) != len(treatment):
        raise ValueError("Baseline and treatment must have same length")
    
    # Compute differences
    diff = treatment - baseline
    mean_diff = np.mean(diff)
    
    # Effect size as percentage
    baseline_mean = np.mean(baseline)
    effect_pct = 100 * mean_diff / max(abs(baseline_mean), 1e-6)
    
    # Perform test
    if test == "wilcoxon":
        # Non-parametric paired test (Wilcoxon signed-rank)
        try:
            stat, p_value = stats.wilcoxon(diff, alternative=alternative)
        except ValueError:
            # All differences are zero
            stat, p_value = 0.0, 1.0
    elif test == "ttest":
        # Parametric paired t-test
        stat, p_value = stats.ttest_rel(treatment, baseline, alternative=alternative)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        "test": test,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "effect_size": float(mean_diff),
        "effect_size_pct": float(effect_pct),
        "n_pairs": len(baseline),
    }


def time_fold_cv_stats(
    results: list[dict[str, float]],
    metric_name: str = "cost",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Compute statistics across time-fold cross-validation results.
    
    Args:
        results: List of result dictionaries from each fold
        metric_name: Metric to analyze (e.g., "cost", "regret")
        confidence: Confidence level for bootstrap CI
    
    Returns:
        Dictionary with:
        - mean: Mean across folds
        - std: Standard deviation
        - median: Median
        - min/max: Min and max values
        - ci_lower/ci_upper: Bootstrap confidence interval
    """
    values = np.array([r[metric_name] for r in results])
    
    # Basic statistics
    stats_dict = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_folds": len(values),
    }
    
    # Bootstrap CI
    if len(values) >= 3:
        ci = bootstrap_ci(values, confidence=confidence)
        stats_dict.update({
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "std_error": ci["std_error"],
        })
    
    return stats_dict


def cohens_d(baseline: np.ndarray, treatment: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Effect size interpretation:
    - |d| < 0.2: small
    - 0.2 <= |d| < 0.5: medium
    - |d| >= 0.5: large
    
    Args:
        baseline: Baseline measurements
        treatment: Treatment measurements
    
    Returns:
        Cohen's d effect size
    """
    baseline = np.asarray(baseline)
    treatment = np.asarray(treatment)
    
    mean_diff = np.mean(treatment) - np.mean(baseline)
    pooled_std = np.sqrt((np.var(baseline) + np.var(treatment)) / 2)
    
    if pooled_std < 1e-10:
        return 0.0
    
    return float(mean_diff / pooled_std)


def compare_systems_statistically(
    baseline_costs: np.ndarray,
    treatment_costs: np.ndarray,
    system_names: tuple[str, str] = ("Baseline", "Treatment"),
    test: str = "wilcoxon",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Comprehensive statistical comparison between two systems.
    
    Args:
        baseline_costs: Baseline system costs (n_runs,)
        treatment_costs: Treatment system costs (n_runs,)
        system_names: Names for baseline and treatment
        test: Statistical test ("wilcoxon" or "ttest")
        confidence: Confidence level
    
    Returns:
        Dictionary with complete statistical summary
    """
    baseline_costs = np.asarray(baseline_costs)
    treatment_costs = np.asarray(treatment_costs)
    
    # Summary statistics for each system
    baseline_stats = {
        "mean": float(np.mean(baseline_costs)),
        "std": float(np.std(baseline_costs)),
        "median": float(np.median(baseline_costs)),
    }
    
    treatment_stats = {
        "mean": float(np.mean(treatment_costs)),
        "std": float(np.std(treatment_costs)),
        "median": float(np.median(treatment_costs)),
    }
    
    # Paired test
    test_result = paired_test(baseline_costs, treatment_costs, test=test)
    
    # Effect size
    effect_size = cohens_d(baseline_costs, treatment_costs)
    
    # Bootstrap CIs
    baseline_ci = bootstrap_ci(baseline_costs, confidence=confidence)
    treatment_ci = bootstrap_ci(treatment_costs, confidence=confidence)
    
    # Cost reduction
    cost_reduction = baseline_stats["mean"] - treatment_stats["mean"]
    cost_reduction_pct = 100 * cost_reduction / max(baseline_stats["mean"], 1e-6)
    
    return {
        "system_names": system_names,
        "baseline": {
            **baseline_stats,
            "ci_lower": baseline_ci["ci_lower"],
            "ci_upper": baseline_ci["ci_upper"],
        },
        "treatment": {
            **treatment_stats,
            "ci_lower": treatment_ci["ci_lower"],
            "ci_upper": treatment_ci["ci_upper"],
        },
        "comparison": {
            "cost_reduction": float(cost_reduction),
            "cost_reduction_pct": float(cost_reduction_pct),
            "test_name": test_result["test"],
            "statistic": test_result["statistic"],
            "p_value": test_result["p_value"],
            "significant": test_result["significant"],
            "cohens_d": float(effect_size),
        },
        "n_runs": len(baseline_costs),
        "confidence": confidence,
    }


def format_stats_table(comparison: dict[str, Any]) -> pd.DataFrame:
    """
    Format statistical comparison as publication-ready table.
    
    Args:
        comparison: Output from compare_systems_statistically
    
    Returns:
        DataFrame with formatted statistics
    """
    baseline_name, treatment_name = comparison["system_names"]
    
    rows = []
    
    # Mean ± std
    rows.append({
        "Metric": "Cost (mean ± std)",
        baseline_name: f"{comparison['baseline']['mean']:.2f} ± {comparison['baseline']['std']:.2f}",
        treatment_name: f"{comparison['treatment']['mean']:.2f} ± {comparison['treatment']['std']:.2f}",
        "p-value": f"{comparison['comparison']['p_value']:.4f}",
    })
    
    # Median (95% CI)
    rows.append({
        "Metric": "Cost (median)",
        baseline_name: f"{comparison['baseline']['median']:.2f}",
        treatment_name: f"{comparison['treatment']['median']:.2f}",
        "p-value": "",
    })
    
    # 95% CI
    rows.append({
        "Metric": "95% CI",
        baseline_name: f"[{comparison['baseline']['ci_lower']:.2f}, {comparison['baseline']['ci_upper']:.2f}]",
        treatment_name: f"[{comparison['treatment']['ci_lower']:.2f}, {comparison['treatment']['ci_upper']:.2f}]",
        "p-value": "",
    })
    
    # Cost reduction
    rows.append({
        "Metric": "Cost reduction",
        baseline_name: "—",
        treatment_name: f"{comparison['comparison']['cost_reduction_pct']:.1f}%",
        "p-value": "",
    })
    
    # Effect size
    rows.append({
        "Metric": "Cohen's d",
        baseline_name: "—",
        treatment_name: f"{comparison['comparison']['cohens_d']:.3f}",
        "p-value": "",
    })
    
    return pd.DataFrame(rows)
