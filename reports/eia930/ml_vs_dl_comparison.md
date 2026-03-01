# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 183.090 | 0.002 |
| lstm | 4413.571 | 0.044 |
| tcn | 6219.553 | 0.061 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 333.223 | 788198352.042 |
| lstm | 6699.709 | 34185.951 |
| tcn | 6934.903 | 32363.889 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 229.603 | 6560725.884 |
| lstm | 1700.873 | 617506.370 |
| tcn | 1444.489 | 191735.966 |

