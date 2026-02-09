# GridPulse Training Pipeline

This document describes the **production-ready** training pipeline implemented in GridPulse. All features ensure **leakage-safe**, **reproducible**, and **statistically rigorous** model training suitable for academic publication and production deployment.

## Overview

The training pipeline extends the base training with:

1. **Leakage-safe preprocessing** with sklearn Pipeline
2. **Time-series cross-validation** (5-fold forward-chaining)
3. **Per-horizon evaluation metrics** (RMSE/MAE/R²/sMAPE)
4. **Conformal prediction intervals** with per-horizon PICP+MPIW
5. **Deterministic seeding + run manifests** (git hash, pip freeze, config snapshot)
6. **Model bundles** with scalers, features, and metrics
7. **Automated verification** of training outputs

---

## 1. Leakage-Safe Preprocessing

### Problem
Standard implementation can leak test data into training:
```python
# ❌ WRONG: scaler sees all data
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
X_train, X_test = split(X_all_scaled)  # Test contamination!
```

### Solution
**Option A: Manual train-only scaling (current DL models)**
```python
# ✓ Correct: fit on train only
scaler = StandardScaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Option B: sklearn Pipeline (Production-Grade GBM)**
```bash
# Enable Pipeline for GBM (ensures leakage-safe preprocessing)
python -m gridpulse.forecasting.train --config configs/train_forecast.yaml --use-pipeline
```

Implementation:
```python
from gridpulse.forecasting.ml_gbm import train_gbm

model_kind, pipeline = train_gbm(
    X_train, y_train,
    params=gbm_params,
    use_pipeline=True,
    preprocessing="standard"  # StandardScaler built into Pipeline
)

# Pipeline handles fit/transform correctly
y_pred = pipeline.predict(X_test)  # Safe!
```

---

## 2. Time-Series Cross-Validation

### Standard Holdout (Always Used)
```
Train (70%) | Val (15%) | Test (15%)
---|---|---
Fit models | Tune hyperparams | Final evaluation
```

### 5-Fold Forward-Chaining CV (Optional)

Enable with `--enable-cv`:
```bash
python -m gridpulse.forecasting.train --config configs/train_forecast.yaml --enable-cv
```

**How it works:**
```
Fold 1: [Train          ] [Val]
Fold 2: [Train              ] [Val]
Fold 3: [Train                  ] [Val]
Fold 4: [Train                      ] [Val]
Fold 5: [Train                          ] [Val]
```

**Output:** `reports/{target}_gbm_cv_results.json`
```json
{
  "n_splits": 5,
  "target": "load_mw",
  "fold_metrics": [
    {"fold": 1, "rmse": 142.3, "r2": 0.9995, ...},
    {"fold": 2, "rmse": 138.7, "r2": 0.9996, ...},
    ...
  ],
  "aggregated": {
    "rmse_mean": 140.5,
    "rmse_std": 2.1,
    "r2_mean": 0.9995,
    "r2_std": 0.0001
  }
}
```

**API Usage:**
```python
from gridpulse.forecasting.evaluate import time_series_cv_score

cv_results = time_series_cv_score(
    X=X_train,
    y=y_train,
    train_fn=lambda X, y: train_gbm(X, y, params)[1],
    predict_fn=predict_gbm,
    n_splits=5,
    target="load_mw"
)

print(f"CV RMSE: {cv_results['aggregated']['rmse_mean']:.2f} ± {cv_results['aggregated']['rmse_std']:.2f}")
```

---

## 3. Per-Horizon Metrics

Standard metrics aggregate over all forecast steps. Per-horizon decomposition shows error growth over time.

### Walk-Forward Horizon Metrics

Automatically computed during training:
```json
{
  "per_horizon": {
    "1": {"rmse": 125.3, "mae": 89.2, "r2": 0.9997},
    "2": {"rmse": 131.7, "mae": 94.1, "r2": 0.9996},
    ...
    "24": {"rmse": 198.4, "mae": 142.3, "r2": 0.9989}
  }
}
```

**Use case:** Identify when forecast quality degrades (e.g., hour 12+).

---

## 4. Conformal Prediction Intervals

### Global Metrics (Always Computed)
```python
from gridpulse.forecasting.uncertainty.conformal import ConformalInterval, ConformalConfig

cfg = ConformalConfig(alpha=0.10, horizon_wise=True)
ci = ConformalInterval(cfg)
ci.fit_calibration(y_true_cal, y_pred_cal)

coverage = ci.coverage(y_true_test, y_pred_test)  # Should be ~0.90
width = ci.mean_width(y_pred_test)
```

### **NEW: Per-Horizon PICP & MPIW**

```python
metrics = ci.evaluate_intervals(y_true_test, y_pred_test, per_horizon=True)

# Output:
{
  "global_coverage": 0.902,
  "global_mean_width": 245.7,
  "per_horizon_picp": {
    "h1": 0.915,   # Hour 1 coverage (91.5%)
    "h2": 0.908,
    ...
    "h24": 0.887   # Hour 24 coverage (88.7%)
  },
  "per_horizon_mpiw": {
    "h1": 198.3,   # Hour 1 width (198 MW)
    "h2": 211.5,
    ...
    "h24": 289.4   # Hour 24 width (289 MW)
  }
}
```

**Interpretation:**
- **PICP** (Prediction Interval Coverage Probability): Fraction of true values inside prediction interval
  - Target: `1 - alpha = 0.90` (90% coverage for alpha=0.10)
  - Per-horizon PICP shows if coverage degrades at longer horizons
  
- **MPIW** (Mean Prediction Interval Width): Average interval width
  - Lower is better (tighter intervals = more confident predictions)
  - Per-horizon MPIW shows uncertainty growth over forecast horizon

**Saved to:** `artifacts/uncertainty/{target}_conformal.json`

---

## 5. Run Manifests (Reproducibility)

Every training run creates a manifest capturing:
- **Git commit hash** and branch
- **Python version** and platform
- **Installed packages** (`pip freeze`)
- **Config snapshot** (YAML copy)
- **Timestamp** and run ID

### Auto-Generated During Training

```bash
python -m gridpulse.forecasting.train --config configs/train_forecast.yaml
# Creates: artifacts/manifest_20260208_143527.json
```

### Manifest Structure

```json
{
  "run_id": "20260208_143527",
  "timestamp": "2026-02-08T14:35:27Z",
  "git": {
    "commit": "a3f8d9c2...",
    "branch": "main",
    "uncommitted_changes": false
  },
  "system": {
    "python_version": "3.11.7",
    "platform": "macOS-14.2-arm64"
  },
  "packages": [
    "numpy==1.26.4",
    "torch==2.8.0",
    ...
  ],
  "config": {
    "path": "configs/train_forecast.yaml",
    "snapshot": "artifacts/models/config_20260208_143527.yaml",
    "content": "task:\n  horizon_hours: 24\n..."
  },
  "metadata": {
    "enable_cv": true,
    "use_pipeline": false,
    "seed": 42
  }
}
```

### API Usage

```python
from gridpulse.utils.manifest import create_run_manifest, compare_manifests

# Create manifest
manifest = create_run_manifest(
    config_path="configs/train_forecast.yaml",
    output_dir="artifacts/models",
    extra_metadata={"notes": "Tuned hyperparameters"}
)

# Compare two runs
diff = compare_manifests(
    "artifacts/manifest_run1.json",
    "artifacts/manifest_run2.json"
)
print(diff["git_match"])  # Did both runs use same commit?
```

---

## 6. Model Bundles

All models saved with complete metadata for deployment:

### GBM Bundle (.pkl)
```python
{
  "model_type": "gbm",
  "model": <LGBMRegressor>,
  "feature_cols": ["load_mw_lag_1", "hour", ...],
  "target": "load_mw",
  "quantiles": [0.1, 0.5, 0.9],
  "residual_quantiles": {"0.1": -205.3, "0.9": 198.7},
  "quantile_models": {"0.1": <Model>, "0.5": <Model>, "0.9": <Model>},
  "tuned_params": {"n_estimators": 567, ...}
}
```

### DL Bundle (.pt)
```python
{
  "model_type": "lstm",
  "state_dict": <OrderedDict>,
  "feature_cols": [...],
  "target": "load_mw",
  "lookback": 168,
  "horizon": 24,
  "x_scaler": {"mean": [...], "std": [...]},
  "y_scaler": {"mean": [...], "std": [...]},
  "model_params": {"hidden_size": 128, ...}
}
```

---

## 7. Automated Verification

Script to assert all artifacts exist after training:

```bash
# Verify all training outputs
python scripts/verify_training_outputs.py --verbose

# Check specific targets
python scripts/verify_training_outputs.py --targets load_mw wind_mw --check-cv

# Or via Makefile
make verify-training
```

### Checks Performed

✅ **Model artifacts** (.pkl, .pt files)  
✅ **Model bundle completeness** (scalers, features, params)  
✅ **Metrics JSONs** (week2_metrics.json, walk_forward_report.json)  
✅ **CV results** (if `--check-cv`)  
✅ **Conformal artifacts** (JSON + NPZ calibration/test data)  
✅ **Run manifests** (git hash, pip freeze)  
✅ **Figure directory** (PNG/SVG counts)  

**Exit codes:**
- `0`: All checks passed ✓
- `1`: Missing or invalid artifacts ✗

---

## Usage Examples

### Basic Training (Standard)
```bash
python -m gridpulse.forecasting.train --config configs/train_forecast.yaml
```

### Production-Grade Training (Full Suite)
```bash
# Enable all Production-Grade features
python -m gridpulse.forecasting.train \
  --config configs/train_forecast.yaml \
  --enable-cv \
  --use-pipeline

# Verify outputs
python scripts/verify_training_outputs.py --verbose --check-cv
```

### Germany (OPSD) Pipeline
```bash
# Full Production-Grade training
python -m gridpulse.forecasting.train \
  --config configs/train_forecast.yaml \
  --enable-cv --use-pipeline

# Verify
make verify-training
```

### USA (EIA-930) Pipeline
```bash
# Full Production-Grade training
python -m gridpulse.forecasting.train \
  --config configs/train_forecast_eia930.yaml \
  --enable-cv --use-pipeline

# Verify
python scripts/verify_training_outputs.py \
  --models-dir artifacts/models_eia930 \
  --reports-dir reports/eia930 \
  --artifacts-dir artifacts \
  --targets load_mw wind_mw solar_mw
```

---

## Makefile Targets

```makefile
# Run time-series cross-validation
make cv-eval

# Verify all training artifacts
make verify-training

# Train Germany models (OPSD)
make train

# Train USA models (EIA-930)
make train-us

# Generate reports
make reports
make reports-us

# Run full production pipeline
make production
```

---

## Configuration Schema Updates

### `configs/train_forecast.yaml`

```yaml
task:
  horizon_hours: 24
  lookback_hours: 168
  quantiles: [0.1, 0.5, 0.9]
  targets: [load_mw, wind_mw, solar_mw, price_eur_mwh]

seed: 42

# Production-Grade options (CLI flags):
# --enable-cv: 5-fold time-series cross-validation
# --use-pipeline: sklearn Pipeline for GBM preprocessing

data:
  processed_path: data/processed/features.parquet
  timestamp_col: timestamp

models:
  baseline_gbm:
    enabled: true
    params:
      n_estimators: 600
      learning_rate: 0.05
      max_depth: 10
      ...
  quantile_gbm:
    enabled: true
    params:
      n_estimators: 400
      ...
  dl_lstm:
    enabled: true
    params:
      hidden_size: 128
      epochs: 50
      ...
  dl_tcn:
    enabled: true
    params:
      num_channels: [64, 64, 64]
      epochs: 50
      ...

backtest:
  enabled: true
  out_path: reports/walk_forward_report.json

tuning:
  enabled: true
  engine: optuna
  n_trials: 20
```

### `configs/uncertainty.yaml`

```yaml
enabled: true
targets: all  # or [load_mw, wind_mw]

conformal:
  alpha: 0.10           # 90% coverage intervals
  horizon_wise: true    # Per-horizon calibration
  rolling: true         # Rolling window updates
  rolling_window: 720

calibration_split: val
calibration_npz: artifacts/backtests/{target}_calibration.npz
test_npz: artifacts/backtests/{target}_test.npz
artifacts_dir: artifacts/uncertainty
```

---

## File Structure

```
gridpulse/
├── src/gridpulse/forecasting/
│   ├── train.py                      # Main training script (Production-Grade enhanced)
│   ├── evaluate.py                   # NEW: TimeSeriesSplit CV
│   ├── ml_gbm.py                     # NEW: Pipeline support
│   └── uncertainty/
│       └── conformal.py              # NEW: Per-horizon PICP/MPIW
├── src/gridpulse/utils/
│   ├── manifest.py                   # NEW: Run manifest tracking
│   └── scaler.py                     # StandardScaler (train-only fit)
├── scripts/
│   ├── verify_training_outputs.py    # NEW: Artifact validation
│   └── generate_publication_figures.py
├── configs/
│   ├── train_forecast.yaml           # Germany config
│   ├── train_forecast_eia930.yaml    # USA config
│   └── uncertainty.yaml              # Conformal config
└── artifacts/
    ├── models/                       # Model bundles
    ├── uncertainty/                  # Conformal JSONs
    ├── backtests/                    # Calibration/test NPZ
    └── manifest_*.json               # Run manifests
```

---

## Academic Paper Checklist

When writing your paper, you can now cite:

✅ **Leakage prevention**: "Train-only preprocessing with sklearn Pipeline"  
✅ **Robust evaluation**: "5-fold forward-chaining time-series cross-validation (TimeSeriesSplit)"  
✅ **Per-horizon analysis**: "RMSE/MAE/R² decomposed by forecast horizon (1-24h)"  
✅ **Uncertainty quantification**: "Conformal prediction with per-horizon PICP and MPIW"  
✅ **Reproducibility**: "Deterministic seeding (seed=42) with git hash and pip freeze manifests"  
✅ **Model bundles**: "All models saved with scalers, feature lists, and hyperparameters"  
✅ **Automated validation**: "Artifact verification script ensures completeness"  

---

## References

- **TimeSeriesSplit**: `sklearn.model_selection.TimeSeriesSplit`
- **Conformal Prediction**: Vovk et al. (2005), "Algorithmic Learning in a Random World"
- **Pipeline**: `sklearn.pipeline.Pipeline` for leakage-safe preprocessing
- **Reproducibility**: DVC, MLflow best practices

---

## Support

For issues or questions:
- Check `scripts/verify_training_outputs.py` output
- Review `artifacts/manifest_*.json` for run details
- Compare manifests: `gridpulse.utils.manifest.compare_manifests()`

**All Production-Grade features are production-ready and tested.**
