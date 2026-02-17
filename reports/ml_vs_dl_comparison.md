# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw, price_eur_mwh**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 267.561 | 0.004 |
| lstm | 3474.242 | 0.054 |
| tcn | 2668.493 | 0.039 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 183.966 | 0.026 |
| lstm | 6735.502 | 1.410 |
| tcn | 9197.628 | 1.654 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 250.911 | 45908522.270 |
| lstm | 4079.357 | 5214638.557 |
| tcn | 2702.105 | 1011682.386 |

## Target: price_eur_mwh

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 4.902 | 1.245 |
| lstm | 15.745 | 3.389 |
| tcn | 15.048 | 3.273 |

