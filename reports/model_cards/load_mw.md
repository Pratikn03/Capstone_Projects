# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 311.72480845071334 | 201.5231471613508 | 0.004206509660543428 | 0.004218388597407557 |
| lstm | 3150.8857113651957 | 2531.3296529479308 | 0.05032593909273349 | 0.05200173489268551 |
| tcn | 3253.0385867209866 | 2489.810464814017 | 0.04891888069365588 | 0.04955627721657126 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
