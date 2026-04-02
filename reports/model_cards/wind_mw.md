# Model Card — wind_mw
## Overview
Forecasting model trained on OPSD Germany time‑series.

## Metrics (test split)
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 169.48598273452913 | 105.94887393173605 | 0.02021658941497981 | 0.023023661198786474 |
| lstm | 6213.293650541069 | 4878.160565741942 | 0.5534699659088912 | 1.0999898691312087 |
| tcn | 7907.128999149803 | 6524.174044693018 | 0.688401623241121 | 1.7365566370048584 |
| nbeats | 7481.904703751449 | 6142.425161888723 | 0.6728316687711736 | 1.6538168922166725 |
| tft | 7320.3438961429 | 5958.948133400825 | 0.66080558320702 | 1.5197525078976903 |
| patchtst | 7417.620654403452 | 6108.6819578473 | 0.673291991827862 | 1.666638998412205 |

## Intended Use
Day‑ahead forecasting for grid planning and dispatch optimization.

## Limitations
Performance depends on feature availability and data quality; retraining is required as grid conditions shift.
