# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 240.35320950771336 | 117.3193371819155 | 0.6923604514515829 | 32602460.145666324 | 0.08864876944143797 |
| nbeats | 2920.2461104784193 | 1815.5492276656976 | 0.982141984592179 | 2206931.1444454975 | 2206931.1444454975 |
| tft | 9020.788896708711 | 7911.670528465601 | 1.3014320127051373 | 31616053.520476427 | 31616053.520476427 |
| patchtst | 3365.943939522834 | 2183.189572123709 | 1.0909832808645556 | 2692679.389838296 | 2692679.389838296 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
