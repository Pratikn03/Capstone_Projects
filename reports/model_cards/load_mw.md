# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 309.97523464409807 | 188.93052865930494 | 0.003925441439473219 | 0.003932067586089694 |
| lstm | 3474.776266889287 | 2722.500057873343 | 0.05375660029301429 | 0.05412533668377964 |
| tcn | 2668.052692101338 | 2031.1189882503816 | 0.039438962013668165 | 0.038934521011744765 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
