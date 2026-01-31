# ML vs DL Comparison

## Setup
- Targets: **load_mw, wind_mw, solar_mw**
- Device: **cpu**
- Quantiles: **[0.1, 0.5, 0.9]**

## Target: load_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 210.058 | 0.003 |
| lstm | 53976.906 | 0.997 |
| tcn | 6643.542 | 0.107 |

## Target: wind_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 457.336 | 0.009 |
| lstm | 18422.062 | 0.974 |
| tcn | 8307.022 | 0.872 |

## Target: solar_mw

| Model | RMSE | MAPE |
|---|---:|---:|
| gbm_lightgbm | 442.729 | 42096950.320 |
| lstm | 10155.833 | 6234659840.000 |
| tcn | 2949.849 | 34253938688.000 |

## Model Comparison Chart (mean across targets)

![Model Comparison](figures/model_comparison.png)
