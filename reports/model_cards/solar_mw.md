# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 251.43813800279966 | 121.22254576412709 | 0.6927383785850142 | 45890919.30939417 | 0.09792229506310599 |
| lstm | 4079.379640875171 | 2835.7400333226856 | 1.0944293937695597 | 5215103.425959513 | 5215103.425959513 |
| tcn | 2702.340459129023 | 1583.0471632632803 | 0.937588029476364 | 1011461.2194118587 | 1011461.2194118587 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
