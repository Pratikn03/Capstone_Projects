# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 254.6457131201481 | 173.27652305638426 | 0.03233788142797395 | 0.03604640667786144 |
| lstm | 5545.723678944462 | 4065.8456184524216 | 0.492966269752505 | 0.8389672730164671 |
| tcn | 6784.753312539991 | 5141.186704237131 | 0.6011567438651633 | 1.1245434376104424 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
