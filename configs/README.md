# GridPulse Configuration Files

YAML configuration files for all system components.

## Configuration Files

| File | Purpose |
|------|---------|
| `data.yaml` | Data paths and preprocessing settings |
| `forecast.yaml` | LightGBM model hyperparameters |
| `train_dl.yaml` | Deep learning (LSTM/TCN) settings |
| `train_forecast.yaml` | Full training pipeline config |
| `train_forecast_eia930.yaml` | US EIA-930 dataset config |
| `uncertainty.yaml` | Conformal prediction settings |
| `anomaly.yaml` | Anomaly detection thresholds |
| `optimization.yaml` | Battery dispatch parameters |
| `monitoring.yaml` | Drift detection thresholds |
| `streaming.yaml` | Kafka streaming settings |
| `serving.yaml` | API service configuration |
| `carbon_factors.yaml` | Carbon intensity factors |

## Structure

Configurations follow a hierarchical structure:

```yaml
# Example: forecast.yaml
model:
  type: lightgbm
  hyperparameters:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 8
    
features:
  lag_hours: [1, 2, 3, 24, 168]
  rolling_windows: [24, 168]
  
training:
  validation_size: 0.15
  early_stopping_rounds: 50
```

## Environment Variables

Configs support environment variable expansion:

```yaml
api:
  key: ${GRIDPULSE_API_KEY}
  host: ${API_HOST:-localhost}
```

## Validation

Run configuration validation:

```bash
python scripts/validate_configs.py
```

## Best Practices

1. **Version control**: All configs are tracked in git
2. **Environment-specific**: Use env vars for secrets
3. **Documentation**: Comment non-obvious settings
4. **Defaults**: Provide sensible defaults where possible
