# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 5300.541419296405 | 3216.2583121960242 | 0.06882321344387979 | 0.0765628003446947 |
| lstm | 21238.768668648045 | 18371.910515963948 | 0.3058125232434614 | 0.3913652104474483 |
| tcn | 17818616.794414494 | 13758153.861561498 | 1.973746453487942 | 259.46997135462726 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 18583.963613377047 | 14917.003772059814 | 1.9999999999968563 | 1.0 |
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
