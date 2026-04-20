# ORIUS — Executive Summary

> One-page submission overview for committee, reviewers, and ML-safety readers.

## Flagship Claim

ORIUS (Observation–Reality Integrity for Universal Safety) is a typed runtime safety layer for Physical AI systems operating under degraded observation. Its single flagship novelty sentence is:

> ORIUS identifies OASG as the degraded-observation release hazard and provides a reliability-aware runtime safety layer across Battery, AV, and Healthcare.

The defended ML center is grouped calibration and runtime safety under degraded observation through the five-stage kernel **Detect → Calibrate → Constrain → Shield → Certify**.

## What The Active 3-Domain Lane Defends

| Domain | Tier | Current defended result | Canonical evidence |
|--------|------|-------------------------|--------------------|
| **Battery (BESS)** | Witness row | TSVR `0.0393 → 0.0000` on the locked publication-nominal surface | `reports/publication/three_domain_ml_benchmark.csv` + `reports/battery_av/battery/runtime_summary.csv` |
| **Autonomous Vehicles** | Bounded defended row | TSVR `0.1250 → 0.0417` on the promoted longitudinal validation harness | `reports/publication/three_domain_ml_benchmark.csv` + `reports/orius_av/full_corpus/runtime_summary.csv` |
| **Medical and Healthcare Monitoring** | Bounded defended row | TSVR `0.2917 → 0.0417` on the promoted MIMIC-backed monitoring row | `reports/publication/three_domain_ml_benchmark.csv` + `data/healthcare/mimic3/processed/mimic3_manifest.json` |

- The strict theorem authority is the active theorem audit, not a flat theorem count.
- The flagship defended theorem core is carried by `T1`, `T2`, `T5`, `T6`, and `T_trajectory_PAC`.
- The flagship ML bundle is:
  - `reports/publication/three_domain_ml_benchmark.csv`
  - `reports/publication/three_domain_reliability_calibration.csv`
  - `reports/publication/three_domain_grouped_coverage.csv`
  - `reports/publication/novelty_separation_matrix.csv`
  - `reports/publication/what_orius_is_not_matrix.csv`

## What The Submission Does Not Claim

- ORIUS is **not** a new universal controller.
- ORIUS is **not** a new conditional-coverage theorem.
- ORIUS is **not** better forecasting by default.
- The AV row is **not** full autonomous-driving closure; it remains bounded to the TTC plus predictive-entry-barrier contract.
- The Healthcare row is **not** clinical deployment readiness; it remains bounded to monitoring-and-alert semantics on the promoted MIMIC surface.

## Evidence Architecture

```text
reports/battery_av_healthcare/overall/
  ├── release_summary.json
  ├── publication_closure_override.json
  ├── domain_summary.csv
  └── lane_status.json

reports/publication/
  ├── three_domain_ml_benchmark.csv
  ├── three_domain_reliability_calibration.csv
  ├── novelty_separation_matrix.csv
  └── what_orius_is_not_matrix.csv
```

## Reproducibility

- Canonical asset build: `PYTHONPATH=src .venv/bin/python scripts/build_orius_monograph_assets.py`
- Canonical review build: `make review-compile`
- Cold-start guide: `ORIUS_REPRODUCIBILITY.md`
