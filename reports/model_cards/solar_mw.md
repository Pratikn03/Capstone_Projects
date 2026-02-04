# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 251.23783758717815 | 120.51588469946148 | 0.6744813958879793 | 32610259.54727739 | 0.04861290806548497 |
| lstm | 4629.5762336389425 | 3219.9212915547964 | 1.1738874543958908 | 3654259.587926467 | 3654259.587926467 |
| tcn | 2768.8137274165933 | 2114.2282694061646 | 0.9969703950115268 | 6170236.228625847 | 6170236.228625847 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
