"""Robustness evaluation module."""

from __future__ import annotations

from .regret import (
    calculate_evpi,
    calculate_vss,
    compute_multi_scenario_regret,
    compute_regret,
    generate_stochastic_metrics_report,
)
from .stats import (
    bca_bootstrap,
    benjamini_hochberg,
    bonferroni,
    bootstrap_ci,
    cohens_d,
    compare_systems_statistically,
    format_stats_table,
    mcnemar_test,
    paired_bootstrap,
    paired_test,
    time_fold_cv_stats,
    wilcoxon_signed_rank,
)

__all__ = [
    "bca_bootstrap",
    "benjamini_hochberg",
    "bonferroni",
    "bootstrap_ci",
    "calculate_evpi",
    "calculate_vss",
    "cohens_d",
    "compare_systems_statistically",
    "compute_multi_scenario_regret",
    "compute_regret",
    "format_stats_table",
    "generate_stochastic_metrics_report",
    "mcnemar_test",
    "paired_bootstrap",
    "paired_test",
    "time_fold_cv_stats",
    "wilcoxon_signed_rank",
]
