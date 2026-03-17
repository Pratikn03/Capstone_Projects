#!/usr/bin/env python3
"""Sync report figures to paper assets. Maps reports/ figures to paper figure tokens.

Run before build to ensure all PaperFigure references resolve. Uses fallbacks
when primary source is missing.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_FIG = REPO / "paper" / "assets" / "figures"
PUB = REPO / "reports" / "publication"
REP_FIG = REPO / "reports" / "figures"
EIA930 = REPO / "reports" / "eia930" / "figures"

# Paper token -> list of candidate source paths (first existing wins)
FIGURE_MAP: dict[str, list[Path]] = {
    "FIG01_ARCHITECTURE": [PAPER_FIG / "fig01_architecture.png", REP_FIG / "architecture.png"],
    "FIG02_DC3S_STEP": [PUB / "figures" / "fig02_dc3s_step.png", REP_FIG / "dispatch_compare.png"],
    "FIG03_TRUE_SOC_VIOLATION": [PUB / "fig_true_soc_violation_vs_dropout.png", PAPER_FIG / "fig03_true_soc_violation_vs_dropout.png"],
    "FIG04_TRUE_SOC_SEVERITY": [PUB / "fig_true_soc_severity_p95_vs_dropout.png", PAPER_FIG / "fig04_true_soc_severity_p95_vs_dropout.png"],
    "FIG05_CQR_GROUP_COVERAGE": [PUB / "fig_cqr_group_coverage.png", PUB / "fig_coverage_width.png", PAPER_FIG / "fig05_cqr_group_coverage.png"],
    "FIG06_TRANSFER_COVERAGE": [PUB / "fig_transfer_coverage.png", PAPER_FIG / "fig06_transfer_coverage.png"],
    "FIG07_COST_SAFETY_FRONTIER": [PUB / "fig_cost_safety_pareto.png", PAPER_FIG / "fig07_cost_safety_frontier.png"],
    "FIG08_RAC_SENSITIVITY_WIDTH": [PUB / "fig_rac_sensitivity_vs_width.png", PUB / "fig_coverage_width_tradeoff.png", PAPER_FIG / "fig08_rac_sensitivity_vs_width.png"],
    "FIG09_REGION_DATASET_CARDS": [PUB / "fig_region_dataset_cards.png", PAPER_FIG / "fig09_region_dataset_cards.png"],
    "FIG10_CALIBRATION_TRADEOFF": [PUB / "fig_calibration_tradeoff.png", PAPER_FIG / "fig10_calibration_tradeoff.png"],
    "FIG11_TRANSFER_GENERALIZATION": [PUB / "fig_transfer_generalization.png", PAPER_FIG / "fig11_transfer_generalization.png"],
    "FIG12_GEOGRAPHIC_SCOPE": [PAPER_FIG / "fig12_geographic_scope.png"],
    "FIG13_LOAD_RENEWABLE_PROFILES": [PAPER_FIG / "fig13_load_renewable_profiles.png"],
    "FIG14_FORECAST_VS_ACTUAL": [REP_FIG / "forecast_sample.png", REP_FIG / "forecast_vs_actual_load.png"],
    "FIG15_ROLLING_BACKTEST_RMSE": [REP_FIG / "rolling_backtest_rmse_by_week.png"],
    "FIG16_ERROR_SEASONALITY": [REP_FIG / "seasonality_error_heatmap.png"],
    "FIG17_CONFORMAL_INTERVALS": [REP_FIG / "prediction_intervals_load.png", PUB / "fig_coverage_width.png"],
    "FIG18_COVERAGE_VS_HORIZON": [REP_FIG / "coverage_by_horizon.png"],
    "FIG19_ANOMALY_TIMELINE": [REP_FIG / "anomaly_timeline.png"],
    "FIG20_DISPATCH_COMPARISON": [REP_FIG / "dispatch_compare.png", EIA930 / "dispatch_compare.png"],
    "FIG21_SOC_TRAJECTORY": [REP_FIG / "soc_trajectory.png"],
    "FIG22_COST_CARBON_TRADEOFF": [REP_FIG / "cost_vs_carbon_tradeoff.png"],
    "FIG23_SAVINGS_SENSITIVITY": [REP_FIG / "impact_savings.png"],
    "FIG24_REGRET_PERTURBATION": [REP_FIG / "robustness_regret_boxplot.png"],
    "FIG25_DATA_DRIFT": [REP_FIG / "data_drift_ks_over_time.png"],
    "FIG26_ABLATION_SENSITIVITY": [PUB / "ablation_sensitivity.png"],
    "FIG27_VIOLATION_VS_COST": [PUB / "sweeps" / "runs" / "alpha0=0.1__ph_lambda=5__kpen=0.5" / "violation_vs_cost_curve.png", REP_FIG / "impact_savings.png"],
    "FIG28_DISTRIBUTIONAL_VS_CQR": [PUB / "fig_coverage_width.png"],
    "FIG29_CROSS_REGION_TRANSFER": [PUB / "cross_region_transfer.png", REP_FIG / "fig_transfer_coverage.png"],
    "FIG30_VIOLATION_RATE": [PUB / "fig_true_soc_violation_vs_dropout.png"],
    "FIG31_VIOLATION_SEVERITY_P95": [PUB / "fig_true_soc_severity_p95_vs_dropout.png"],
    "FIG32_SHAP_SUMMARY_LOAD": [REP_FIG / "shap_summary_load_mw.png"],
    "FIG33_SHAP_SUMMARY_WIND": [REP_FIG / "shap_summary_wind_mw.png"],
    "FIG34_SHAP_SUMMARY_SOLAR": [REP_FIG / "shap_summary_solar_mw.png"],
    "FIG35_FORECAST_VS_ACTUAL_LOAD": [REP_FIG / "forecast_vs_actual_load.png"],
    "FIG36_FORECAST_VS_ACTUAL_WIND": [REP_FIG / "forecast_vs_actual_wind.png"],
    "FIG37_FORECAST_VS_ACTUAL_SOLAR": [REP_FIG / "forecast_vs_actual_solar.png"],
    "FIG38_MODEL_COMPARISON": [REP_FIG / "model_comparison.png", EIA930 / "model_comparison.png"],
    "FIG39_DISPATCH_COMPARE": [REP_FIG / "dispatch_compare.png"],
    "FIG40_ARBITRAGE_OPTIMIZATION": [REP_FIG / "arbitrage_optimization.png"],
    "FIG41_SOC_TRAJECTORY_DETAIL": [REP_FIG / "soc_trajectory.png"],
    "FIG42_PREDICTION_INTERVALS_LOAD": [REP_FIG / "prediction_intervals_load.png"],
    "FIG43_INTERVAL_WIDTH_BY_HORIZON": [REP_FIG / "interval_width_by_horizon.png"],
    "FIG44_DATA_DRIFT_KS": [REP_FIG / "data_drift_ks_over_time.png"],
    "FIG45_MODEL_DRIFT_METRIC": [REP_FIG / "model_drift_metric_over_time.png"],
    "FIG46_RESIDUAL_ZSCORE": [REP_FIG / "residual_zscore_timeline.png"],
    "FIG47_ERROR_DISTRIBUTION": [REP_FIG / "error_distribution_residuals.png"],
    "FIG48_SEASONALITY_ERROR_HEATMAP": [REP_FIG / "seasonality_error_heatmap.png"],
    "FIG49_ROBUSTNESS_REGRET_BOXPLOT": [REP_FIG / "robustness_regret_boxplot.png"],
    "FIG50_ROBUSTNESS_INFEASIBLE_RATE": [REP_FIG / "robustness_infeasible_rate.png"],
    "FIG51_COVERAGE_BY_HORIZON": [REP_FIG / "coverage_by_horizon.png"],
    "FIG52_COST_CARBON_DETAIL": [REP_FIG / "cost_vs_carbon_tradeoff.png"],
    "FIG53_CASE_STUDY_DISPATCH": [REP_FIG / "case_study_dispatch.png", PAPER_FIG / "fig_48h_trace.png"],
    "FIG54_CASE_STUDY_WEEK": [REP_FIG / "case_study_week.png", EIA930 / "case_study_week.png"],
    "FIG55_MULTI_HORIZON_BACKTEST": [REP_FIG / "multi_horizon_backtest.png", EIA930 / "multi_horizon_backtest.png"],
    "FIG56_USA_DATA_QUALITY": [REP_FIG / "usa_data_quality_deepdive.png"],
    "FIG57_GAP_HOURLY_PROFILES": [REP_FIG / "gap_hourly_profiles.png"],
    "FIG58_GAP_DISTRIBUTION": [PAPER_FIG / "fig58_gap_distribution.png"],
    "FIG_UNIVERSAL_FRAMEWORK": [PAPER_FIG / "fig_universal_framework.png"],
}


def _paper_path(token: str) -> Path:
    """Resolve paper asset path from token. Uses manifest naming convention."""
    name_map = {
        "FIG01_ARCHITECTURE": "fig01_architecture.png",
        "FIG02_DC3S_STEP": "fig02_dc3s_step.png",
        "FIG03_TRUE_SOC_VIOLATION": "fig03_true_soc_violation_vs_dropout.png",
        "FIG04_TRUE_SOC_SEVERITY": "fig04_true_soc_severity_p95_vs_dropout.png",
        "FIG05_CQR_GROUP_COVERAGE": "fig05_cqr_group_coverage.png",
        "FIG06_TRANSFER_COVERAGE": "fig06_transfer_coverage.png",
        "FIG07_COST_SAFETY_FRONTIER": "fig07_cost_safety_frontier.png",
        "FIG08_RAC_SENSITIVITY_WIDTH": "fig08_rac_sensitivity_vs_width.png",
        "FIG09_REGION_DATASET_CARDS": "fig09_region_dataset_cards.png",
        "FIG10_CALIBRATION_TRADEOFF": "fig10_calibration_tradeoff.png",
        "FIG11_TRANSFER_GENERALIZATION": "fig11_transfer_generalization.png",
        "FIG12_GEOGRAPHIC_SCOPE": "fig12_geographic_scope.png",
        "FIG13_LOAD_RENEWABLE_PROFILES": "fig13_load_renewable_profiles.png",
        "FIG14_FORECAST_VS_ACTUAL": "fig14_forecast_vs_actual.png",
        "FIG15_ROLLING_BACKTEST_RMSE": "fig15_rolling_backtest_rmse.png",
        "FIG16_ERROR_SEASONALITY": "fig16_error_seasonality.png",
        "FIG17_CONFORMAL_INTERVALS": "fig17_conformal_intervals.png",
        "FIG18_COVERAGE_VS_HORIZON": "fig18_coverage_vs_horizon.png",
        "FIG19_ANOMALY_TIMELINE": "fig19_anomaly_timeline.png",
        "FIG20_DISPATCH_COMPARISON": "fig20_dispatch_comparison.png",
        "FIG21_SOC_TRAJECTORY": "fig21_soc_trajectory.png",
        "FIG22_COST_CARBON_TRADEOFF": "fig22_cost_carbon_tradeoff.png",
        "FIG23_SAVINGS_SENSITIVITY": "fig23_savings_sensitivity.png",
        "FIG24_REGRET_PERTURBATION": "fig24_regret_perturbation.png",
        "FIG25_DATA_DRIFT": "fig25_data_drift.png",
        "FIG26_ABLATION_SENSITIVITY": "fig26_ablation_sensitivity.png",
        "FIG27_VIOLATION_VS_COST": "fig27_violation_vs_cost.png",
        "FIG28_DISTRIBUTIONAL_VS_CQR": "fig28_distributional_vs_cqr.png",
        "FIG29_CROSS_REGION_TRANSFER": "fig29_cross_region_transfer.png",
        "FIG30_VIOLATION_RATE": "fig30_violation_rate.png",
        "FIG31_VIOLATION_SEVERITY_P95": "fig31_violation_severity_p95.png",
        "FIG32_SHAP_SUMMARY_LOAD": "fig32_shap_summary_load.png",
        "FIG33_SHAP_SUMMARY_WIND": "fig33_shap_summary_wind.png",
        "FIG34_SHAP_SUMMARY_SOLAR": "fig34_shap_summary_solar.png",
        "FIG35_FORECAST_VS_ACTUAL_LOAD": "fig35_forecast_vs_actual_load.png",
        "FIG36_FORECAST_VS_ACTUAL_WIND": "fig36_forecast_vs_actual_wind.png",
        "FIG37_FORECAST_VS_ACTUAL_SOLAR": "fig37_forecast_vs_actual_solar.png",
        "FIG38_MODEL_COMPARISON": "fig38_model_comparison.png",
        "FIG39_DISPATCH_COMPARE": "fig39_dispatch_compare.png",
        "FIG40_ARBITRAGE_OPTIMIZATION": "fig40_arbitrage_optimization.png",
        "FIG41_SOC_TRAJECTORY_DETAIL": "fig41_soc_trajectory_detail.png",
        "FIG42_PREDICTION_INTERVALS_LOAD": "fig42_prediction_intervals_load.png",
        "FIG43_INTERVAL_WIDTH_BY_HORIZON": "fig43_interval_width_by_horizon.png",
        "FIG44_DATA_DRIFT_KS": "fig44_data_drift_ks.png",
        "FIG45_MODEL_DRIFT_METRIC": "fig45_model_drift_metric.png",
        "FIG46_RESIDUAL_ZSCORE": "fig46_residual_zscore.png",
        "FIG47_ERROR_DISTRIBUTION": "fig47_error_distribution.png",
        "FIG48_SEASONALITY_ERROR_HEATMAP": "fig48_seasonality_error_heatmap.png",
        "FIG49_ROBUSTNESS_REGRET_BOXPLOT": "fig49_robustness_regret_boxplot.png",
        "FIG50_ROBUSTNESS_INFEASIBLE_RATE": "fig50_robustness_infeasible_rate.png",
        "FIG51_COVERAGE_BY_HORIZON": "fig51_coverage_by_horizon.png",
        "FIG52_COST_CARBON_DETAIL": "fig52_cost_carbon_detail.png",
        "FIG53_CASE_STUDY_DISPATCH": "fig53_case_study_dispatch.png",
        "FIG54_CASE_STUDY_WEEK": "fig54_case_study_week.png",
        "FIG55_MULTI_HORIZON_BACKTEST": "fig55_multi_horizon_backtest.png",
        "FIG56_USA_DATA_QUALITY": "fig56_usa_data_quality.png",
        "FIG57_GAP_HOURLY_PROFILES": "fig57_gap_hourly_profiles.png",
        "FIG58_GAP_DISTRIBUTION": "fig58_gap_distribution.png",
        "FIG_UNIVERSAL_FRAMEWORK": "fig_universal_framework.png",
    }
    return PAPER_FIG / name_map.get(token, f"{token.lower()}.png")


def main() -> int:
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    synced = 0
    missing = []

    for token, candidates in FIGURE_MAP.items():
        dst = _paper_path(token)
        src = None
        for c in candidates:
            if c.exists():
                src = c
                break
        if src and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            synced += 1
        elif not dst.exists():
            missing.append(token)

    print(f"Synced {synced} figures to paper/assets/figures/")
    if missing:
        print(f"Missing ({len(missing)}): {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
