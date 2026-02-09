# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 139.56084110639836 | 103.80694988079468 | 0.001410804367798613 | 0.0014106444790883707 |
| lstm | 3684.7258041228997 | 2826.066128177769 | 0.03795016661539154 | 0.03753817637505856 |
| tcn | 4235.372144626466 | 3251.1130439674826 | 0.04341264408873344 | 0.04256222082515162 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
