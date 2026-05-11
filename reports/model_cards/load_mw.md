# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 254.46962225713838 | 161.00867465708384 | 0.0033757756825217385 | 0.0033841481905837485 |
| lstm | 7540.30598859217 | 6753.144443999601 | 0.12604181605591336 | 0.13659431241102682 |
| tcn | 7465.936919521333 | 6774.402330836834 | 0.1255544214783542 | 0.13603635233553443 |
| nbeats | 5080.1941530211425 | 4101.365194206555 | 0.07928563621237648 | 0.08256197147205376 |
| tft | 8958.754500685904 | 7724.4369415523415 | 0.15194873874151052 | 0.1512829275001665 |
| patchtst | 3937.978903108021 | 2993.4955676169666 | 0.05840820839884364 | 0.05676929800696147 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
