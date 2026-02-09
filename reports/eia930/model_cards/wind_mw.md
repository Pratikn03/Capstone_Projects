# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 244.5496955143561 | 111.44366558423876 | 0.014679243382987895 | 421902758.28858805 |
| lstm | 6732.315044593082 | 5393.442130282138 | 0.4528177330216119 | 12754.993780015144 |
| tcn | 6969.003001818571 | 5863.4087310658515 | 0.4753333608293543 | 16490.17437712878 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
