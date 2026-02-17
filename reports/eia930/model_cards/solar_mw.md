# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 203.81657283172586 | 74.67275747690604 | 0.46211458014678414 | 10338752.34764313 | 0.7277483588390152 |
| lstm | 1607.664446880507 | 985.0155340662252 | 1.3414418125495307 | 393034.06120136444 | 84.37122030851546 |
| tcn | 1398.1378133547073 | 835.0464982482388 | 1.302093275861052 | 451197.8101677131 | 92.79426025536415 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
