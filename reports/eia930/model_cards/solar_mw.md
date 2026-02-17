# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 208.9155178581897 | 74.92639127730418 | 0.45450404712509446 | 9407839.965532975 | 0.8231692256919554 |
| lstm | 1781.9688755793472 | 1055.338939519795 | 1.3692435056156054 | 389364.1547393924 | 77.2306877272726 |
| tcn | 1743.6645818725526 | 965.0847461366311 | 1.3149781435492511 | 308397.89643664926 | 71.58159943902696 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
