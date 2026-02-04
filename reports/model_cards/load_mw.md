# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 268.45054774692863 | 160.11265279305687 | 0.0033930200373756044 | 0.0034026741258969223 |
| lstm | 4975.172724385955 | 3633.784332033787 | 0.07096101031748166 | 0.0689612324628587 |
| tcn | 6157.075359090004 | 5172.092866547439 | 0.106351766897947 | 0.09933457774538754 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
