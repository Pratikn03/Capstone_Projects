# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 12411.628450075548 | 10782.008009813826 | 1.9673834608137406 | 0.9836917304084284 |
| lstm | 12492.957293500665 | 10925.599039104149 | 1.999989700380644 | 622.4913997410354 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
