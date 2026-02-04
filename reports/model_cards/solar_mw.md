# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 277.2354565494422 | 137.40184591312502 | 0.7033863301649446 | 88802171.19391449 | 0.17588067168074012 |
| lstm | 3033.734595419477 | 2208.320710398724 | 1.0265053780522089 | 3734370.787846106 | 3734370.787846106 |
| tcn | 2445.89895524222 | 1581.4788292001956 | 0.9566574762464122 | 2608843.4612632683 | 2608843.4612632683 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
