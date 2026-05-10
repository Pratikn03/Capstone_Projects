# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw, price_eur_mwh**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 254.470 | 0.003 |
| lstm | 7540.306 | 0.137 |
| tcn | 7465.937 | 0.136 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 163.489 | 0.023 |
| lstm | 7206.161 | 1.421 |
| tcn | 7146.837 | 1.266 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 237.642 | 22344464.599 |
| lstm | 2465.764 | 228517.617 |
| tcn | 2875.997 | 349435.384 |

## Target: price_eur_mwh

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 4.971 | 1.361 |
| lstm | 11.500 | 4.234 |
| tcn | 12.703 | 4.584 |

