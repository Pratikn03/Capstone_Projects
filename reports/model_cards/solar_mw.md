# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 254.2868291404199 | 140.05516446679871 | 0.7568718245715603 | 590419852.4626968 | 0.6362662597892045 |
| lstm | 4079.379640875171 | 2835.7400333226856 | 1.0944293937695597 | 5215103.425959513 | 5215103.425959513 |
| tcn | 2702.340459129023 | 1583.0471632632803 | 0.937588029476364 | 1011461.2194118587 | 1011461.2194118587 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
