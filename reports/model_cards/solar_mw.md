# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 442.7285394414581 | 142.53011716798076 | 0.8684740203856156 | 42096950.319630034 | 0.09650620540074602 |
| lstm | 10155.8330078125 | 5828.0888671875 | 1.8478258848190308 | 6234659840.0 | 5.227997303009033 |
| tcn | 2949.849365234375 | 1985.85498046875 | 1.215532660484314 | 34253938688.0 | 29.72636604309082 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
