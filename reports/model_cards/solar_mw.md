# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 240.35320950771336 | 117.3193371819155 | 0.6923604514515829 | 32602460.145666324 | 0.08864876944143797 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
