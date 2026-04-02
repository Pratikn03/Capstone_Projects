# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 305.1209624375309 | 187.86941461742373 | 0.003903913775710714 | 0.003911572495132256 |
| nbeats | 3834.9989289671275 | 3141.9828470834927 | 0.06334658722851647 | 0.06073790731652637 |
| tft | 8986.547287121532 | 7939.828262523263 | 0.1561407871747522 | 0.1604818069539302 |
| patchtst | 5660.764357567655 | 4702.48419347189 | 0.09297393711770467 | 0.08755996888561209 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
