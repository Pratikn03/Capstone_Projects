# Shift-Aware Uncertainty in ORIUS

## Problem 2 Definition
Problem 2 is uncertainty validity under distribution shift: intervals can remain calibrated marginally while failing on hard subgroups. ORIUS now tracks subgroup failures, online adaptive quantile behavior, and drift-linked validity states at runtime.

## What changed vs legacy RAC-Cert
- **Legacy path preserved**: reliability + drift inflation still works unchanged when shift-aware mode is disabled.
- **Shift-aware path added**: runtime interval decisions include validity score/status, conditional coverage gap, adaptive quantile state, and policy label.

## Runtime objects
- `validity_score` in `[0,1]` and `validity_status` in `{nominal, watch, degraded, invalid}`.
- `adaptive_quantile` tracked online from hit/miss feedback.
- `coverage_group_key` identifies reliability/volatility/fault/time/custom subgroup.

## Scripts
- `python scripts/run_shift_aware_coverage_audit.py --input <csv>`
- `python scripts/run_adaptive_quantile_replay.py --input <csv>`
- `python scripts/benchmark_shift_aware_runtime.py --steps 10000`

## Artifacts
Written under `reports/publication/`:
- `reliability_group_coverage.csv`
- `volatility_group_coverage.csv`
- `fault_group_coverage.csv`
- `shift_validity_trace.csv`
- `adaptive_quantile_trace.csv`

## Interpretation
- **Watch**: early warning, keep operating with moderate widening.
- **Degraded**: stronger widening and alerting.
- **Invalid**: uncertainty trust is poor; runtime shield gets explicit invalid status.
