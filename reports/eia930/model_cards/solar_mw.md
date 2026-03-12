# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 229.98768085846916 | 82.14335776365684 | 0.5309049987178593 | 10611026.85282825 | 0.6680680076052432 |
| lstm | 1700.8725151892208 | 1161.313448293208 | 1.3782643440636377 | 617506.3702363383 | 159.18462406495743 |
| tcn | 1444.48926066807 | 754.7591873337503 | 1.2436807834927754 | 191735.96556505052 | 47.4721348447582 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
