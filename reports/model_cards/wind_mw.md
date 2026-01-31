# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 457.3359549872597 | 154.55677808860673 | 0.009106595579560053 | 0.009233442289595661 |
| lstm | 18422.0625 | 14707.0791015625 | 1.9046194553375244 | 0.9743436574935913 |
| tcn | 8307.0224609375 | 6476.7421875 | 0.5190566182136536 | 0.8721930980682373 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
