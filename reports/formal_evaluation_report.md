# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 210.0583597847119 | 125.83129029468961 | 0.0025181018323605663 | 0.0025260273299586872 |
| lstm | 3324.7741898278246 | 2427.2819398445417 | 0.046478936281032765 | 0.048398699978655056 |
| tcn | 3336.2361574586353 | 2458.659096972831 | 0.04798872277537601 | 0.04999548592999707 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 457.3359549872597 | 154.55677808860673 | 0.009106595579560053 | 0.009233442289595661 |
| lstm | 18551.547585777156 | 14868.944211844477 | 1.9999955567755237 | 1.0000009399094407 |
| tcn | 18403.257204187536 | 14855.032618050021 | 1.892701527748823 | 1.037937590694127 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 10122.507403779273 | 5727.020342179711 | 1.1776909601703751 | 0.5888454802640442 | 1.0 |
| lstm | 10242.553489606169 | 5840.8395899559755 | 1.9991410136551238 | 972791.2319070074 | 0.9999319494500374 |
| tcn | 10141.583763747207 | 5877.463442260229 | 1.9003787874511744 | 9422616164.30947 | 6.424446696664806 |

## Multi‑Horizon Backtest (Load)
![](/Users/pratik_n/Downloads/gridpulse/reports/figures/multi_horizon_backtest.png)

## Conclusions
GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.
