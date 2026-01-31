# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 210.0583597847119 | 125.83129029468961 | 0.0025181018323605663 | 0.0025260273299586872 |
| lstm | 53976.90625 | 53074.75 | 1.9861712455749512 | 0.9965304136276245 |
| tcn | 6643.54248046875 | 5330.8486328125 | 0.10105045884847641 | 0.1069224402308464 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
