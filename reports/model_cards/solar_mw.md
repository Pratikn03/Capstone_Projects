# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 10122.507403779273 | 5727.020342179711 | 1.1776909601703751 | 0.5888454802640442 | 1.0 |
| lstm | 10242.553489606169 | 5840.8395899559755 | 1.9991410136551238 | 972791.2319070074 | 0.9999319494500374 |
| tcn | 10141.583763747207 | 5877.463442260229 | 1.9003787874511744 | 9422616164.30947 | 6.424446696664806 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
