# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw, price_eur_mwh**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 270.297 | 0.003 |
| lstm | 2356.004 | 0.034 |
| tcn | 3394.782 | 0.052 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 127.100 | 0.023 |
| lstm | 6025.591 | 0.855 |
| tcn | 7170.804 | 1.051 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 269.477 | 107763526.155 |
| lstm | 2536.569 | 1052534.297 |
| tcn | 3006.648 | 2912918.574 |

## Target: price_eur_mwh

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 4.904 | 1.206 |
| lstm | 10.819 | 3.996 |
| tcn | 14.395 | 3.711 |

