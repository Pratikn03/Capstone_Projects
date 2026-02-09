# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 214.22632462226662 | 75.4025034139851 | 0.3802039884459303 | 15705665.16278832 | 0.5363139423770488 |
| lstm | 1607.664446880507 | 985.0155340662252 | 1.3414418125495307 | 393034.06120136444 | 84.37122030851546 |
| tcn | 1398.1378133547073 | 835.0464982482388 | 1.302093275861052 | 451197.8101677131 | 92.79426025536415 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
