# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 130.1893791929321 | 91.25972294820731 | 0.020061795705913564 | 0.02280878448806715 |
| lstm | 5679.480870237086 | 4071.257203924327 | 0.5083383266432826 | 0.7740500115709378 |
| tcn | 7000.6263935145225 | 5123.859590274129 | 0.6195663481721845 | 0.9154126142363062 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
