# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 269.5542942974431 | 129.53972566184072 | 0.7041557205607407 | 107722205.784544 | 0.1404350400943314 |
| lstm | 2536.1146372469066 | 1536.0016177466066 | 0.9657990565299844 | 1052589.3212084842 | 1052589.3212084842 |
| tcn | 3006.7451233891106 | 2009.5950846992325 | 0.9721751549480074 | 2912810.8970445017 | 2912810.8970445017 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
