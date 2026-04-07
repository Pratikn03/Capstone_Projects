"""
Statistical analysis tools for ORIUS evaluation.

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


def bca_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Bias-corrected and accelerated bootstrap interval."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("BCa bootstrap requires a one-dimensional sample of size >= 2")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    n = values.size
    point_est = float(statistic(values))
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot = np.array([float(statistic(values[sample])) for sample in idx], dtype=float)

    less_fraction = np.clip(np.mean(boot < point_est), 1e-9, 1.0 - 1e-9)
    z0 = float(stats.norm.ppf(less_fraction))

    jackknife = np.empty(n, dtype=float)
    for i in range(n):
        jackknife[i] = float(statistic(np.delete(values, i)))
    jack_mean = float(np.mean(jackknife))
    num = np.sum((jack_mean - jackknife) ** 3)
    den = 6.0 * (np.sum((jack_mean - jackknife) ** 2) ** 1.5 + 1e-12)
    acceleration = float(num / den) if den > 0 else 0.0

    alpha = 1.0 - confidence
    z_lo = float(stats.norm.ppf(alpha / 2.0))
    z_hi = float(stats.norm.ppf(1.0 - alpha / 2.0))
    adj_lo = float(
        stats.norm.cdf(z0 + (z0 + z_lo) / max(1.0 - acceleration * (z0 + z_lo), 1e-12))
    )
    adj_hi = float(
        stats.norm.cdf(z0 + (z0 + z_hi) / max(1.0 - acceleration * (z0 + z_hi), 1e-12))
    )
    adj_lo = float(np.clip(adj_lo, 0.0, 1.0))
    adj_hi = float(np.clip(adj_hi, 0.0, 1.0))
    ci_lower = float(np.quantile(boot, min(adj_lo, adj_hi)))
    ci_upper = float(np.quantile(boot, max(adj_lo, adj_hi)))

    return {
        "point_estimate": point_est,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "bias_correction_z0": z0,
        "acceleration": acceleration,
    }


def paired_bootstrap(
    baseline: np.ndarray,
    treatment: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for paired differences."""
    base = np.asarray(baseline, dtype=float)
    treat = np.asarray(treatment, dtype=float)
    if base.shape != treat.shape:
        raise ValueError("baseline and treatment must have matching shape")
    if base.ndim != 1 or base.size < 2:
        raise ValueError("paired bootstrap requires one-dimensional paired samples of size >= 2")

    diffs = treat - base
    ci = bootstrap_ci(diffs, statistic=statistic, confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)
    ci["effect_size"] = float(statistic(diffs))
    return ci


def wilcoxon_signed_rank(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float | bool]:
    """Convenience wrapper around the paired Wilcoxon signed-rank test."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("sample_a and sample_b must have matching shape")
    if np.allclose(a, b):
        stat = 0.0
        p_value = 1.0
    else:
        stat, p_value = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "n_pairs": int(a.size),
    }


def mcnemar_test(confusion_matrix: np.ndarray, exact: bool | None = None) -> dict[str, float | bool | str]:
    """McNemar test for paired binary outcomes.

    Expects a 2x2 table:
        [[both_correct, a_only],
         [b_only, both_wrong]]
    """
    table = np.asarray(confusion_matrix, dtype=float)
    if table.shape != (2, 2):
        raise ValueError("confusion_matrix must be 2x2")
    b = float(table[0, 1])
    c = float(table[1, 0])
    n = b + c
    use_exact = bool(n < 25) if exact is None else bool(exact)
    if n == 0:
        statistic = 0.0
        p_value = 1.0
        method = "degenerate"
    elif use_exact:
        statistic = min(b, c)
        p_value = float(2.0 * stats.binomtest(int(min(b, c)), int(n), p=0.5).pvalue)
        p_value = min(p_value, 1.0)
        method = "exact"
    else:
        statistic = ((abs(b - c) - 1.0) ** 2) / n
        p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))
        method = "chi2_cc"
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "method": method,
    }


def benjamini_hochberg(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Benjamini-Hochberg false-discovery-rate correction."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    thresholds = alpha * (np.arange(1, n + 1) / max(n, 1))
    passed = ranked <= thresholds
    k = int(np.max(np.nonzero(passed)[0])) + 1 if np.any(passed) else 0
    reject_ranked = np.zeros(n, dtype=bool)
    if k > 0:
        reject_ranked[:k] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_ranked

    adjusted_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return {
        "alpha": alpha,
        "rejected": reject.tolist(),
        "adjusted_p_values": adjusted.tolist(),
        "n_rejected": int(np.sum(reject)),
    }


def bonferroni(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Bonferroni family-wise error-rate correction."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    adjusted = np.clip(p * max(len(p), 1), 0.0, 1.0)
    rejected = adjusted <= alpha
    return {
        "alpha": alpha,
        "rejected": rejected.tolist(),
        "adjusted_p_values": adjusted.tolist(),
        "n_rejected": int(np.sum(rejected)),
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
