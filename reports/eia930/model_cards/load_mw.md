# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 183.369246744089 | 140.64221070422795 | 0.0018225504269295894 | 0.0018218246510229466 |
| lstm | 4413.570886076817 | 3416.167851102955 | 0.04456974238131669 | 0.04410213310317136 |
| tcn | 6219.553186950962 | 4734.462425598746 | 0.06285607979492414 | 0.06060904100770707 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
