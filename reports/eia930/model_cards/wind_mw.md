# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 310.2093820736406 | 182.0326613105617 | 0.02123504076366505 | 527148926.7790369 |
| lstm | 6732.315044593082 | 5393.442130282138 | 0.4528177330216119 | 12754.993780015144 |
| tcn | 6969.003001818571 | 5863.4087310658515 | 0.4753333608293543 | 16490.17437712878 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
