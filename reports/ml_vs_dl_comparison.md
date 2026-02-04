# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw, price_eur_mwh**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 311.765 | 0.004 |
| lstm | 3150.873 | 0.052 |
| tcn | 3252.646 | 0.050 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 254.695 | 0.036 |
| lstm | 5546.509 | 0.839 |
| tcn | 6785.419 | 1.125 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 276.266 | 88836234.167 |
| lstm | 3033.816 | 3734611.901 |
| tcn | 2446.106 | 2609031.895 |

## Target: price_eur_mwh

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 4.918 | 1.519 |
| lstm | 11.711 | 4.981 |
| tcn | 11.872 | 5.612 |

