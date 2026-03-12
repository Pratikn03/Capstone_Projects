# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 357.09618359110743 | 210.72024983372535 | 0.02649683215003194 | 796670137.431063 |
| lstm | 6699.708802166455 | 5729.828730333515 | 0.4619438785216474 | 34185.95123977535 |
| tcn | 6934.903180692785 | 5963.546587811382 | 0.47302189708212306 | 32363.88864425602 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
