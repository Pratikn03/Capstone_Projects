# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 305.1209624375309 | 187.86941461742373 | 0.003903913775710714 | 0.003911572495132256 |
| nbeats | 3646.2813440775776 | 2897.931830401579 | 0.05842906163946666 | 0.05611493093836337 |
| tft | 8996.048465327724 | 7892.515856717142 | 0.15522967587094355 | 0.16064577377806163 |
| patchtst | 4087.856999164547 | 3142.774383116903 | 0.062241690325248525 | 0.059458884017828506 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
