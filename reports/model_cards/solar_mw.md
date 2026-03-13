# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 240.35320950771336 | 117.3193371819155 | 0.6923604514515829 | 32602460.145666324 | 0.08864876944143797 |
| nbeats | 3441.4465020394996 | 2192.564640844372 | 1.04797523091076 | 2473086.952341414 | 2473086.952341414 |
| tft | 9067.28453351178 | 8041.595454347984 | 1.2870201512143018 | 34046116.259454444 | 34046116.259454444 |
| patchtst | 3136.9355579539033 | 2030.2183818457377 | 1.0223736653780242 | 3157307.047928045 | 3157307.047928045 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
