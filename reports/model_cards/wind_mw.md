# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 457.3359549872597 | 154.55677808860673 | 0.009106595579560053 | 0.009233442289595661 |
| lstm | 18551.547585777156 | 14868.944211844477 | 1.9999955567755237 | 1.0000009399094407 |
| tcn | 18403.257204187536 | 14855.032618050021 | 1.892701527748823 | 1.037937590694127 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
