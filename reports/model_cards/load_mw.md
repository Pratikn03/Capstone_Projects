# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 5300.541419296405 | 3216.2583121960242 | 0.06882321344387979 | 0.0765628003446947 |
| lstm | 21238.768668648045 | 18371.910515963948 | 0.3058125232434614 | 0.3913652104474483 |
| tcn | 17818616.794414494 | 13758153.861561498 | 1.973746453487942 | 259.46997135462726 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
