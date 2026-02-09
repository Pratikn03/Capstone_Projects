# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 139.561 | 0.001 |
| lstm | 3684.726 | 0.038 |
| tcn | 4235.372 | 0.043 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 244.550 | 421902758.289 |
| lstm | 6732.315 | 12754.994 |
| tcn | 6969.003 | 16490.174 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 214.226 | 15705665.163 |
| lstm | 1607.664 | 393034.061 |
| tcn | 1398.138 | 451197.810 |

