# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 267.50996312219684 | 167.49140642626713 | 0.0035023723719414864 | 0.0035116406790806354 |
| lstm | 3474.776266889287 | 2722.500057873343 | 0.05375660029301429 | 0.05412533668377964 |
| tcn | 2668.052692101338 | 2031.1189882503816 | 0.039438962013668165 | 0.038934521011744765 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
