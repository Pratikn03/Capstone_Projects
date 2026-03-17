# Multi-Domain Training Pipeline

This document describes the training pipeline for all ORIUS domains: **Energy**, **AV**, **Industrial**, **Healthcare**, and **Aerospace**.

## Quick Start

```bash
# Download all multi-domain datasets (AV, Industrial, Healthcare, Aerospace)
make multi-domain-datasets

# Build features for all non-energy domains
make multi-domain-build

# Train a specific domain
make train-dataset DATASET=AV
make train-dataset DATASET=INDUSTRIAL
make train-dataset DATASET=HEALTHCARE
make train-dataset DATASET=AEROSPACE

# Or use the Python script directly
.venv/bin/python scripts/train_dataset.py --dataset AV
```

## Domain Overview

| Domain      | Target(s)   | Features Path                          | Config                          |
|-------------|-------------|----------------------------------------|---------------------------------|
| Energy (DE) | load_mw, wind_mw, solar_mw | data/processed/features.parquet       | configs/train_forecast.yaml     |
| AV          | speed_mps   | data/av/processed/features.parquet     | configs/train_forecast_av.yaml  |
| Industrial  | power_mw    | data/industrial/processed/features.parquet | configs/train_forecast_industrial.yaml |
| Healthcare  | hr_bpm      | data/healthcare/processed/features.parquet  | configs/train_forecast_healthcare.yaml |
| Aerospace   | airspeed_kt | data/aerospace/processed/features.parquet   | configs/train_forecast_aerospace.yaml |

## Pipeline Steps

1. **Download** – Fetch or generate domain-specific data
2. **Build features** – Run domain feature module (lag features, etc.)
3. **Create splits** – Train/calibration/val/test time-series splits
4. **Validate schema** – Check required columns (domain-aware)
5. **Train models** – GBM (and optionally DL) per target
6. **Generate reports** – Metrics, figures, model cards
7. **Verify** – Artifact checks (conformal skipped for multi-domain)

## Key Files

| Purpose              | Path |
|----------------------|------|
| AV features          | `src/orius/data_pipeline/build_features_av.py` |
| Industrial features  | `src/orius/data_pipeline/build_features_industrial.py` |
| Healthcare features  | `src/orius/data_pipeline/build_features_healthcare.py` |
| Aerospace features   | `src/orius/data_pipeline/build_features_aerospace.py` |
| Multi-domain build   | `scripts/build_features_multi_domain.py` |
| Schema validation    | `src/orius/data_pipeline/validate_schema.py` (supports `--required-cols`) |
| Dataset registry     | `scripts/_dataset_registry.py` |
| Train script         | `scripts/train_dataset.py` |

## Multi-Domain Differences

- **Schema validation**: Uses `--required-cols` (e.g. `speed_mps`) instead of energy defaults
- **Data manifest**: Built for all domains including multi-domain (AV, Industrial, Healthcare, Aerospace)
- **Conformal/uncertainty**: Uses dataset targets for conformal prediction; horizon clamped to ≥1 for sub-hourly configs
- **Reports**: Pass `--targets` to `build_reports.py`; energy-only steps (impact, dispatch, case study) are skipped
- **Verification**: Full verification including conformal artifacts for all domains
- **Feature selection**: `make_xy` excludes non-numeric columns (e.g. `ts_utc`) and uses `_report_targets` for domain targets

## Makefile Targets

```makefile
av-datasets          # Download AV data
industrial-datasets  # Download Industrial data
healthcare-datasets  # Download Healthcare data
aerospace-datasets   # Generate Aerospace synthetic data
multi-domain-datasets # All of the above
multi-domain-build   # Build features for AV, Industrial, Healthcare, Aerospace
train-dataset DATASET=AV  # Train AV
```
