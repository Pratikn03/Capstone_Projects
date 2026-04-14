# Shift-Aware Uncertainty (Problem 2) in ORIUS

## Implementation map (audit summary)
- Existing conformal calibration and adaptive alpha logic exists in `src/orius/forecasting/uncertainty/conformal.py`.
- Existing observation reliability + drift-aware widening exists in DC3S calibration/pipeline (`src/orius/dc3s/calibration.py`, `src/orius/dc3s/pipeline.py`).
- Existing conditional coverage theorem support exists in `src/orius/dc3s/coverage_theorem.py`.
- Missing before this upgrade: online subgroup tracker, runtime validity score, adaptive quantile trace artifacts, and explicit DC3S certificate fields for shift-aware validity.
- Backward compatibility retained by default (`ShiftAwareConfig.enabled: false`), preserving legacy RAC-Cert behavior.

## What Problem 2 adds
Problem 2 treats **uncertainty validity under shift** as a runtime object:
- online calibration failure detection,
- subgroup under-coverage tracking,
- adaptive quantile updates,
- drift-aware validity scoring,
- runtime policy control and audit artifacts.

## Difference vs legacy RAC-Cert
Legacy RAC-Cert widens uncertainty mainly from reliability and drift. The shift-aware subsystem additionally tracks whether the uncertainty model remains valid and conditioned across subgroups.

## How to run
- Coverage audit:
  - `python scripts/run_shift_aware_coverage_audit.py --input <csv> --out-dir reports/publication`
- Adaptive replay:
  - `python scripts/run_adaptive_quantile_replay.py --input <csv> --out-dir reports/publication`
- Runtime benchmark:
  - `python scripts/benchmark_shift_aware_runtime.py --steps 5000`

## Interpreting outputs
- `validity_score`: scalar `[0,1]` where lower is worse.
- `validity_status`: `nominal | watch | degraded | invalid`.
- `adaptive_quantile_trace.csv`: online effective alpha/quantile evolution.
- subgroup coverage CSVs: per-group empirical coverage and under-coverage gap vs target.
