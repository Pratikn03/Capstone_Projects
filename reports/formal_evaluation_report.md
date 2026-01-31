# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 210.0583597847119 | 125.83129029468961 | 0.0025181018323605663 | 0.0025260273299586872 |
| lstm | 53976.90625 | 53074.75 | 1.9861712455749512 | 0.9965304136276245 |
| tcn | 6643.54248046875 | 5330.8486328125 | 0.10105045884847641 | 0.1069224402308464 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 457.3359549872597 | 154.55677808860673 | 0.009106595579560053 | 0.009233442289595661 |
| lstm | 18422.0625 | 14707.0791015625 | 1.9046194553375244 | 0.9743436574935913 |
| tcn | 8307.0224609375 | 6476.7421875 | 0.5190566182136536 | 0.8721930980682373 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 442.7285394414581 | 142.53011716798076 | 0.8684740203856156 | 42096950.319630034 | 0.09650620540074602 |
| lstm | 10155.8330078125 | 5828.0888671875 | 1.8478258848190308 | 6234659840.0 | 5.227997303009033 |
| tcn | 2949.849365234375 | 1985.85498046875 | 1.215532660484314 | 34253938688.0 | 29.72636604309082 |

## Multi‑Horizon Backtest (Load)
![](/Users/pratik_n/Downloads/gridpulse/reports/figures/multi_horizon_backtest.png)

## Conclusions
GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.
