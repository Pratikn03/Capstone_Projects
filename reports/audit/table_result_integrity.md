# Table/Result Integrity Audit

- Generated: `2026-05-13T01:40:06.213568+00:00`
- Passes: `True`
- Blocking findings: `0`
- Warning findings: `222`
- Scanned: `{'csv': 281, 'json': 193, 'tex': 87, 'duckdb': 0}`

## Top Warnings
- `paper/assets/tables/tbl02_ablations.csv` `true_soc_violation_rate_wilcoxon_stat`: all_zero_metric_review (9 numeric values are all zero.)
- `paper/assets/tables/tbl02_ablations.csv` `true_soc_violation_severity_p95_wilcoxon_stat`: all_zero_metric_review (9 numeric values are all zero.)
- `paper/assets/tables/tbl01_main_results.csv` `true_soc_violation_rate_ci_low`: all_zero_metric_review (5 numeric values are all zero.)
- `paper/assets/tables/tbl01_main_results.csv` `true_soc_violation_severity_p95_ci_low`: all_zero_metric_review (5 numeric values are all zero.)
- `paper/assets/tables/tbl01_main_results.csv` `intervention_rate_ci_low`: all_zero_metric_review (5 numeric values are all zero.)
- `paper/assets/tables/generated/tbl_battery_deep_oqe_summary.tex` `not_applicable`: rendered_placeholder_cell (Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `paper/assets/tables/generated/tbl_intervention_tradeoff.tex` `not_applicable`: rendered_placeholder_cell (Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `paper/assets/tables/generated/tbl_vehicle_leaderboard.tex` `not_applicable`: rendered_placeholder_cell (Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `paper/assets/tables/generated/tbl_multi_domain_evidence_gate.tex` `not_applicable`: rendered_placeholder_cell (Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `paper/assets/tables/generated/tbl_per_domain_controller_comparison.tex` `not_applicable`: rendered_placeholder_cell (Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `paper/assets/tables/generated/tbl_healthcare_leaderboard.tex` `not_applicable`: rendered_placeholder_cell (Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `paper/assets/tables/generated/tbl_constraint_satisfaction.tex` `not_applicable`: rendered_placeholder_cell (Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/cross_region_transfer.csv` `intervention_rate`: all_zero_metric_review (96 numeric values are all zero. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/cross_region_transfer.csv` `grid_import_violation_rate`: all_zero_metric_review (96 numeric values are all zero. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/runtime_summary.csv` `recovery_latency`: all_zero_metric_review (4 numeric values are all zero. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/fault_family_coverage.csv` `coverage`: all_zero_metric_review (12 numeric values are all zero. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/certificate_half_life_blackout.csv` `violations`: all_zero_metric_review (4 numeric values are all zero. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/domain_runtime_contract_witnesses.csv` `failed_obligations`: placeholder_or_blank_cell (50/75 rows contain missing/placeholder values. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/domain_runtime_contract_witnesses.csv` `failure_reason`: placeholder_or_blank_cell (50/75 rows contain missing/placeholder values. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/active_theorem_audit.csv` `assumptions_used`: placeholder_or_blank_cell (19/29 rows contain missing/placeholder values. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/active_theorem_audit.csv` `typed_obligations`: placeholder_or_blank_cell (8/29 rows contain missing/placeholder values. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/active_theorem_audit.csv` `unresolved_assumptions`: placeholder_or_blank_cell (29/29 rows contain missing/placeholder values. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/battery_deep_oqe_summary.csv` `heuristic`: placeholder_or_blank_cell (2/5 rows contain missing/placeholder values. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/battery_deep_oqe_safety_metrics.csv` `true_soc_violation_rate`: all_zero_metric_review (12 numeric values are all zero. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
- `reports/publication/battery_deep_oqe_safety_metrics.csv` `recovery_latency`: all_zero_metric_review (12 numeric values are all zero. Non-current historical/generated surface; audited for visibility but excluded from the strict current-publication gate.)
