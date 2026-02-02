# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 210.0583597847119 | 125.83129029468961 | 0.0025181018323605663 | 0.0025260273299586872 |
| lstm | 3324.7741898278246 | 2427.2819398445417 | 0.046478936281032765 | 0.048398699978655056 |
| tcn | 3336.2361574586353 | 2458.659096972831 | 0.04798872277537601 | 0.04999548592999707 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
