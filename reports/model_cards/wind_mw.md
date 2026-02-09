# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 127.08497812200575 | 87.33009845331962 | 0.019834754096708387 | 0.022747657908375494 |
| lstm | 6025.136806059045 | 4304.789359172982 | 0.5205029171351211 | 0.8550586818046001 |
| tcn | 7169.663581806657 | 5185.96133242178 | 0.6047623966086424 | 1.0503505655693721 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
