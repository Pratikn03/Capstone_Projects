# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 139.800 | 0.001 |
| lstm | 3684.726 | 0.038 |
| tcn | 4235.372 | 0.043 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 239.615 | 438372316.591 |
| lstm | 6732.315 | 12754.994 |
| tcn | 6969.003 | 16490.174 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 212.885 | 14429469.557 |
| lstm | 1607.664 | 393034.061 |
| tcn | 1398.138 | 451197.810 |

