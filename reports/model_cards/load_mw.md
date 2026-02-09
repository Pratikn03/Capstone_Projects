# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 298.6000157184173 | 185.93743041795145 | 0.003911473903321219 | 0.003922968131321532 |
| lstm | 2355.9737707052263 | 1732.0841498345 | 0.033589623391714125 | 0.03384176636351295 |
| tcn | 3394.193713865421 | 2613.5046818291494 | 0.0534685897978956 | 0.051515772878522645 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
