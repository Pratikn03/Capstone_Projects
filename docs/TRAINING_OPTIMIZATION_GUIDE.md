# Model Training Optimization Guide

This document provides specific, actionable recommendations for improving GridPulse model quality based on the current training configuration and observed performance gaps.

---

## Current Performance Summary

| Model | Target | R² (Current) | R² (Target) | Gap |
|-------|--------|--------------|-------------|-----|
| GBM | load_mw | 0.9991 | 0.9995 | Minimal |
| LSTM | load_mw | 0.931 (DE) / 0.762 (US) | 0.990 | **Severe** |
| TCN | load_mw | 0.857 (DE) / 0.685 (US) | 0.990 | **Severe** |

The GBM models perform excellently. Deep learning models significantly underperform.

---

## 1. Deep Learning Hyperparameter Recommendations

### LSTM: Current vs Recommended

```yaml
# CURRENT (configs/train_forecast.yaml)
dl_lstm:
  params:
    hidden_size: 256
    num_layers: 3
    dropout: 0.3
    epochs: 100
    batch_size: 128
    learning_rate: 0.001
    weight_decay: 1e-5

# RECOMMENDED
dl_lstm:
  params:
    hidden_size: 512          # Increase capacity
    num_layers: 2             # Reduce depth (gradient vanishing)
    dropout: 0.2              # Lower dropout for more learning
    epochs: 200               # Longer training
    batch_size: 64            # Smaller batches for better generalization
    learning_rate: 0.0005     # Lower LR for stability
    weight_decay: 1e-6        # Less regularization
    scheduler: cosine
    warmup_epochs: 10         # Longer warmup
    early_stopping:
      patience: 30            # More patience
      min_delta: 0.00001      # Finer threshold
    gradient_clip: 0.5        # Tighter clipping
    bidirectional: true       # Add bidirectional processing
    attention: true           # Add attention mechanism
```

**Rationale:**
- 3-layer LSTM suffers from gradient vanishing on long sequences
- Higher hidden_size with fewer layers often works better
- Smaller batch sizes improve gradient estimation
- ower learning rate prevents overshooting

### TCN: Current vs Recommended

```yaml
# CURRENT
dl_tcn:
  params:
    num_channels: [128, 128, 128, 64]
    kernel_size: 5
    dropout: 0.3
    epochs: 100

# RECOMMENDED
dl_tcn:
  params:
    num_channels: [64, 128, 256, 256, 128]  # Encoder-decoder pattern
    kernel_size: 7                           # Larger receptive field
    dropout: 0.15                            # Lower dropout
    epochs: 300                              # Much longer training
    batch_size: 32                           # Smaller batches
    learning_rate: 0.0003                    # Lower LR
    dilation_base: 2                         # Exponential dilation
    use_skip_connections: true               # ResNet-style skips
    use_layer_norm: true                     # Normalize activations
    early_stopping:
      patience: 40
```

**Rationale:**
- TCN receptive field = (kernel_size - 1) * Σ(dilation) + 1
- For 168-hour lookback, need larger receptive field
- Skip connections prevent gradient degradation

---

## 2. Learning Rate Scheduling

Current configuration uses basic cosine annealing. Consider:

### Option A: One-Cycle Policy (Recommended)
```yaml
scheduler:
  type: one_cycle
  max_lr: 0.003           # Peak learning rate
  pct_start: 0.3          # 30% warmup
  div_factor: 25          # Initial LR = max_lr / 25
  final_div_factor: 1000  # Final LR = max_lr / 1000
```

### Option B: Reduce on Plateau
```yaml
scheduler:
  type: reduce_on_plateau
  mode: min
  factor: 0.5
  patience: 10
  min_lr: 1e-7
```

---

## 3. Sequence Length and Lookback

Current: `lookback_hours: 168` (1 week)

**Analysis:**
- Load patterns have daily (24h) and weekly (168h) seasonality
- 168-hour lookback is appropriate
- However, consider multi-scale inputs:

```yaml
# Multi-resolution input
lookback:
  short_term: 24    # Past day (fine detail)
  medium_term: 168  # Past week (weekly pattern)
  long_term: 720    # Past month (trend) - subsampled to daily
```

---

## 4. Data Augmentation for Time Series

Add noise injection and temporal jittering:

```python
def augment_time_series(X, y, noise_std=0.01, shift_range=2):
    """
    Augment training data for better generalization.
    
    Args:
        X: Input features (batch, seq_len, features)
        y: Targets
        noise_std: Gaussian noise standard deviation
        shift_range: Max temporal shift (hours)
    """
    # Gaussian noise
    X_noisy = X + np.random.normal(0, noise_std * X.std(), X.shape)
    
    # Temporal shift (simulate clock drift)
    shift = np.random.randint(-shift_range, shift_range + 1)
    X_shifted = np.roll(X, shift, axis=1)
    
    return X_noisy, X_shifted, y
```

---

## 5. Seed and Reproducibility

Current seeds:
```yaml
seed: 42
seeds: [42, 123, 456, 789, 2024]  # For ensemble
```

**Recommendation:** Train with multiple seeds and ensemble:

```bash
# Train with 5 different seeds
for seed in 42 123 456 789 2024; do
    python -m gridpulse.forecasting.train \
        --config configs/train_forecast.yaml \
        --seed $seed \
        --output-suffix _seed${seed}
done

# Ensemble predictions
python scripts/ensemble_models.py --seeds 42,123,456,789,2024 --method mean
```

Ensemble typically improves performance by 5-15%.

---

## 6. Time Budget Recommendations

### For Better Deep Learning Results

| Model | Current Epochs | Recommended | Training Time Estimate |
|-------|---------------|-------------|----------------------|
| LSTM | 100 | 200-300 | 2-4 hours (GPU) |
| TCN | 100 | 300-400 | 3-5 hours (GPU) |

**With Early Stopping:**
- Set patience=40 to allow sufficient exploration
- Use min_delta=1e-5 for sensitive stopping

### For Hyperparameter Tuning (Optuna)

Current:
```yaml
tuning:
  n_trials: 50
  n_jobs: 4
```

Recommended:
```yaml
tuning:
  n_trials: 200       # More exploration
  n_jobs: -1          # Use all cores
  timeout: 14400      # 4 hours max
  pruner:
    type: hyperband
    min_resource: 20   # Min epochs before pruning
    reduction_factor: 3
```

---

## 7. Feature Engineering Improvements

### Add More Lag Features
```python
# Current lags
lag_features = [1, 2, 24, 48, 168]

# Recommended lags (capture more patterns)
lag_features = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]
```

### Add Fourier Features
```python
def add_fourier_features(df, period=24, n_harmonics=3):
    """Add Fourier terms to capture cyclical patterns."""
    for k in range(1, n_harmonics + 1):
        df[f'sin_{period}h_{k}'] = np.sin(2 * np.pi * k * df['hour'] / period)
        df[f'cos_{period}h_{k}'] = np.cos(2 * np.pi * k * df['hour'] / period)
    return df
```

### External Regressors
Consider adding:
- Weather forecasts (not just actuals)
- Public holiday flags (already present)
- Day-ahead price forecasts
- Grid frequency deviations (if available)

---

## 8. Specific Fixes for USA Dataset

The USA (EIA-930 MISO) models show worse performance. Potential causes:

### Problem 1: Different Data Characteristics
- MISO has higher intermittency than German grid
- More variable renewable penetration

**Fix:** Increase uncertainty bounds for USA:
```yaml
# configs/train_forecast_eia930.yaml
conformal:
  alpha: 0.05  # 95% intervals instead of 90%
```

### Problem 2: Feature Mismatch
- USA dataset has different features than Germany
- Price proxy may not be realistic

**Fix:** Verify USA feature alignment:
```bash
python -c "
import pandas as pd
df_de = pd.read_parquet('data/processed/features.parquet')
df_us = pd.read_parquet('data/processed/us_eia930/features.parquet')
print('DE features:', len(df_de.columns))
print('US features:', len(df_us.columns))
print('Missing in US:', set(df_de.columns) - set(df_us.columns))
"
```

### Problem 3: Optimization Configuration
USA VSS = 0.00 suggests:
- Price signal may be too flat
- Battery configuration may not match USA grid

**Fix:** Adjust optimization parameters for USA:
```yaml
# configs/optimization_eia930.yaml
battery:
  capacity_mwh: 200.0      # Larger battery
  max_charge_mw: 75.0
  max_discharge_mw: 75.0
grid:
  price_per_mwh: 45.0      # Lower base price (USD)
  peak_premium_factor: 1.5  # Add peak pricing
```

---

## 9. Training Command Examples

### Quick Training (Testing)
```bash
python -m gridpulse.forecasting.train \
    --config configs/train_forecast.yaml \
    --epochs 10 \
    --quick
```

### Full Training (Production)
```bash
python -m gridpulse.forecasting.train \
    --config configs/train_forecast.yaml \
    --enable-cv \
    --use-pipeline \
    --n-seeds 5
```

### Hyperparameter Search
```bash
python -m gridpulse.forecasting.train \
    --config configs/train_forecast.yaml \
    --tune \
    --n-trials 200 \
    --timeout 14400
```

### Deep Learning Only (Focus)
```bash
python -m gridpulse.forecasting.train \
    --config configs/train_dl.yaml \
    --target load_mw \
    --epochs 300 \
    --batch-size 32 \
    --lr 0.0003
```

---

## 10. Validation Strategy

### Current: Single Train/Val/Test Split
- 70/15/15 time-based split
- Risk: May not generalize across seasons

### Recommended: Walk-Forward Validation
```
Year 1-2: Train → Year 3: Val → Year 4: Test
Year 1-3: Train → Year 4: Val → Year 5: Test
...
```

Enable with:
```yaml
cross_validation:
  enabled: true
  strategy: walk_forward
  n_folds: 5
  test_size_hours: 720  # 1 month test per fold
```

---

## Summary: Priority Actions

1. **Immediate (1 day):**
   - Lower LSTM to 2 layers, increase hidden_size to 512
   - Increase training epochs to 200+
   - Lower learning rate to 0.0003-0.0005

2. **Short-term (1 week):**
   - Implement one-cycle learning rate
   - Add bidirectional LSTM option
   - Run 5-seed ensemble

3. **Medium-term (2 weeks):**
   - Full hyperparameter search (200+ trials)
   - Add Fourier features
   - Investigate USA data quality

4. **Long-term (1 month):**
   - Consider Transformer architecture
   - Implement probabilistic forecasting (Gaussian outputs)
   - External weather forecast integration
