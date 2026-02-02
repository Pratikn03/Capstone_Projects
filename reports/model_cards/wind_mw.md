# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 18583.963613377047 | 14917.003772059814 | 1.9999999999968563 | 1.0 |
| lstm | 18551.547585777156 | 14868.944211844477 | 1.9999955567755237 | 1.0000009399094407 |
| tcn | 18403.257204187536 | 14855.032618050021 | 1.892701527748823 | 1.037937590694127 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
