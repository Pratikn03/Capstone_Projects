# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 257.30103653279747 | 162.87037842496227 | 0.0034154553617824477 | 0.0034244515594101547 |
| lstm | 3474.776266889287 | 2722.500057873343 | 0.05375660029301429 | 0.05412533668377964 |
| tcn | 2668.052692101338 | 2031.1189882503816 | 0.039438962013668165 | 0.038934521011744765 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
