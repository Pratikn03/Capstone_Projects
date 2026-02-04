# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 123.95640107405151 | 85.11575104376448 | 0.01971040792375047 | 0.022651132863360717 |
| lstm | 6088.681917184256 | 4712.231697154939 | 0.5649990812670935 | 1.082712566266003 |
| tcn | 7086.184863970565 | 5011.337688603397 | 0.597803473619078 | 0.959135282851123 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
