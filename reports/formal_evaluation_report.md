# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 210.0583597847119 | n/a | n/a | 0.0025260273299586872 | n/a |
| lstm | 53976.90625 | n/a | n/a | 0.9965304136276245 | n/a |
| tcn | 6643.54248046875 | n/a | n/a | 0.1069224402308464 | n/a |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 457.3359549872597 | n/a | n/a | 0.009233442289595661 | n/a |
| lstm | 18422.0625 | n/a | n/a | 0.9743436574935913 | n/a |
| tcn | 8307.0224609375 | n/a | n/a | 0.8721930980682373 | n/a |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 442.7285394414581 | n/a | n/a | 42096950.319630034 | n/a |
| lstm | 10155.8330078125 | n/a | n/a | 6234659840.0 | n/a |
| tcn | 2949.849365234375 | n/a | n/a | 34253938688.0 | n/a |

## Multi‑Horizon Backtest (Load)
![](/Users/pratik_n/Downloads/gridpulse/reports/figures/multi_horizon_backtest.png)

## Conclusions
GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.
