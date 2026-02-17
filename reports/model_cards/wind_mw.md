# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 183.9305042033171 | 118.18776499596973 | 0.022895253969732902 | 0.02599257803801949 |
| lstm | 6735.07009751006 | 5511.226734264865 | 0.6084608518120028 | 1.4094474308126195 |
| tcn | 9196.156140506526 | 7167.4122668564605 | 0.7667156554333923 | 1.6537467556241543 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
