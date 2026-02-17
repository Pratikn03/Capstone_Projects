# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 162.889098360766 | 123.22699049377624 | 0.0016711108353770399 | 0.0016708040250974364 |
| lstm | 4767.13981471182 | 3835.3213548638823 | 0.052446208271787996 | 0.05144989867645077 |
| tcn | 3850.48798204528 | 2877.3076092397896 | 0.03821778287424641 | 0.03773362151594056 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
