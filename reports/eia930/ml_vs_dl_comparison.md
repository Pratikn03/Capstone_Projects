# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 182.227 | 0.002 |
| lstm | 4788.957 | 0.049 |
| tcn | 5173.258 | 0.053 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 334.280 | 788697459.210 |
| lstm | 6414.972 | 42938.183 |
| tcn | 6929.794 | 32682.861 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 211.661 | 15132125.076 |
| lstm | 1752.054 | 498542.105 |
| tcn | 1636.726 | 20754.286 |

