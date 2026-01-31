# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 442.7285394414581 | n/a | n/a | 42096950.319630034 | n/a |
| lstm | 10155.8330078125 | n/a | n/a | 6234659840.0 | n/a |
| tcn | 2949.849365234375 | n/a | n/a | 34253938688.0 | n/a |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
