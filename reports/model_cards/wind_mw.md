# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 219.1075784147325 | 129.3622766969851 | 0.024099812028293276 | 0.027884827851954033 |
| nbeats | 7571.148082326869 | 6232.122707659645 | 0.6776443360312809 | 1.6735142389130495 |
| tft | 7312.885898839694 | 5540.71355039805 | 0.63802250253213 | 1.147260565436826 |
| patchtst | 7383.74255383858 | 5974.24413670895 | 0.6652359681965024 | 1.5482969554967791 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
