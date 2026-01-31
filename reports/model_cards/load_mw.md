# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 210.0583597847119 | n/a | n/a | 0.0025260273299586872 | n/a |
| lstm | 53976.90625 | n/a | n/a | 0.9965304136276245 | n/a |
| tcn | 6643.54248046875 | n/a | n/a | 0.1069224402308464 | n/a |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
