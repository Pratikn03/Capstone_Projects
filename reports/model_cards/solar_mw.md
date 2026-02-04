# Model Card — solar_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 239.01965918961375 | 116.19207100407529 | 0.6909447680075763 | 56895606.614095934 | 0.09616268073085278 |
| lstm | 3821.059423253299 | 2586.159735480109 | 1.0953148448816592 | 2613868.237764853 | 2613868.237764853 |
| tcn | 2515.3953881901975 | 1574.5808193708394 | 0.9570789863073993 | 1944896.6379839387 | 1944896.6379839387 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
