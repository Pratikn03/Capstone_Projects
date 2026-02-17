# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 162.889 | 0.002 |
| lstm | 4767.140 | 0.051 |
| tcn | 3850.488 | 0.038 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 269.230 | 463928300.713 |
| lstm | 6301.907 | 18240.288 |
| tcn | 7187.543 | 24566.312 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 208.916 | 9407839.966 |
| lstm | 1781.969 | 389364.155 |
| tcn | 1743.665 | 308397.896 |

