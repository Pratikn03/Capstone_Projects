# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 4760.943700726715 | 2829.769375090201 | 1.8610189014589982 | 0.9305094530235243 | 1.0 |
| lstm | 4792.066895724467 | 2867.4839629486974 | 1.9865236159807007 | 162219.74691763284 | 1.0000471658770034 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
