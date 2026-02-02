# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 211.1099686007024 | 111.44578763515571 | 0.0013837949950884533 | 0.001383005581060905 |
| lstm | 13411.260866566116 | 10238.471109038774 | 0.1346894485583618 | 0.1290998477615506 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
