# Model Card — load_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 182.22731815260244 | 134.31147362296937 | 0.001741370809780408 | 0.0017407105312542592 |
| lstm | 4788.956614525568 | 3773.3433424556556 | 0.04978454259318315 | 0.04878685225645898 |
| tcn | 5173.258144192 | 4062.3268569308816 | 0.05242975139080958 | 0.05317831180970469 |
| nbeats | 13112.253424427425 | 10669.134537339056 | 0.1486458453303889 | 0.13439067394675916 |
| tft | 10239.691422784752 | 8607.71188837616 | 0.11581858036875912 | 0.10803389727816028 |
| patchtst | 6653.491262062936 | 5202.627487353067 | 0.06859182485603912 | 0.0666978826307173 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
