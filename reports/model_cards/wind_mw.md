# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 457.3359549872597 | n/a | n/a | 0.009233442289595661 | n/a |
| lstm | 18422.0625 | n/a | n/a | 0.9743436574935913 | n/a |
| tcn | 8307.0224609375 | n/a | n/a | 0.8721930980682373 | n/a |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
