# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 261.9013026862777 | 153.72911980681192 | 0.0032671543645541898 | 0.003275942934649663 |
| lstm | 3653.033580802336 | 2943.7813178683728 | 0.06014463366141743 | 0.059251385023769206 |
| tcn | 4962.79349858157 | 3989.2522238461424 | 0.08067873718563466 | 0.08602321931475024 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
