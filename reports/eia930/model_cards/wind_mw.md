# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 334.2804531884752 | 183.72711738733585 | 0.02283585708985751 | 788697459.2102427 |
| lstm | 6414.971520794806 | 5466.412275687497 | 0.45004542804416253 | 42938.183303959035 |
| tcn | 6929.7936124424195 | 5963.700701900386 | 0.47784558211812683 | 32682.86128741423 |
| nbeats | 8059.1680063909225 | 6764.5939241240785 | 0.5572118929330281 | 27211.92704758095 |
| tft | 7034.164466589556 | 6032.029942678739 | 0.4731624165330012 | 31310.19587458604 |
| patchtst | 7106.909323020893 | 6041.998645600134 | 0.48117080617072633 | 33297.56406468478 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
