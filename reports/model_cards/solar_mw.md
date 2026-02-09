# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 247.82742595723835 | 118.72247785473017 | 0.6887528768499864 | 32060327.266559295 | 0.07243011528850685 |
| lstm | 4079.379640875171 | 2835.7400333226856 | 1.0944293937695597 | 5215103.425959513 | 5215103.425959513 |
| tcn | 3006.7451233891106 | 2009.5950846992325 | 0.9721751549480074 | 2912810.8970445017 | 2912810.8970445017 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
