# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 269.22959007922213 | 144.65876286602762 | 0.017458976355303666 | 463928300.7134795 |
| lstm | 6301.907170540824 | 5234.314534441705 | 0.4259245611501527 | 18240.28812852139 |
| tcn | 7187.542595933752 | 5930.744731074514 | 0.476476746555839 | 24566.311973066917 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
