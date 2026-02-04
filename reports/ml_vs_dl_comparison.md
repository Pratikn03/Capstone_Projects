# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw, price_eur_mwh**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 268.496 | 0.003 |
| lstm | 4975.453 | 0.069 |
| tcn | 6158.062 | 0.099 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 123.975 | 0.023 |
| lstm | 6088.626 | 1.083 |
| tcn | 7087.383 | 0.959 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 250.810 | 32622768.277 |
| lstm | 4630.057 | 3655461.608 |
| tcn | 2768.761 | 6170556.629 |

## Target: price_eur_mwh

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 4.886 | 1.205 |
| lstm | 12.443 | 4.483 |
| tcn | 15.162 | 3.827 |

