# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw, price_eur_mwh**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 261.950 | 0.003 |
| lstm | 3653.348 | 0.059 |
| tcn | 4961.157 | 0.086 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 130.161 | 0.023 |
| lstm | 5680.368 | 0.774 |
| tcn | 7001.289 | 0.915 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 238.782 | 56917430.782 |
| lstm | 3820.715 | 2614345.288 |
| tcn | 2515.105 | 1944900.367 |

## Target: price_eur_mwh

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 4.813 | 1.202 |
| lstm | 11.737 | 3.265 |
| tcn | 13.123 | 4.111 |

