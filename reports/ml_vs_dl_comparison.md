# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw, price_eur_mwh**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 256.000 | 0.003 |
| lstm | 4314.498 | 0.070 |
| tcn | 1708.092 | 0.026 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 169.486 | 0.023 |
| lstm | 6213.294 | 1.100 |
| tcn | 7907.129 | 1.737 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 242.677 | 52765596.918 |
| lstm | 3392.007 | 2391024.371 |
| tcn | 2612.759 | 546397.091 |

## Target: price_eur_mwh

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 4.943 | 1.291 |
| lstm | 12.560 | 3.613 |
| tcn | 12.366 | 3.274 |

