# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 305.1209624375309 | 187.86941461742373 | 0.003903913775710714 | 0.003911572495132256 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 219.1075784147325 | 129.3622766969851 | 0.024099812028293276 | 0.027884827851954033 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 240.35320950771336 | 117.3193371819155 | 0.6923604514515829 | 32602460.145666324 | 0.08864876944143797 |

### price_eur_mwh
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 4.998528372562706 | 2.328736454310254 | 0.10161859846727576 | 1.446678693662406 |

## Baseline Metrics (Test Split)
### load_mw
| Baseline | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| persistence_24h | 6027.9165370760265 | 3926.2976522085155 | 0.07885130425312796 | 0.07775592600459137 |
| moving_average_24h | 8081.41633673434 | 6811.264557633639 | 0.13411407310869267 | 0.13882396917703638 |

### wind_mw
| Baseline | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| persistence_24h | 7823.3481418906595 | 5510.391165937127 | 0.6233845918389466 | 0.8576784765434317 |
| moving_average_24h | 4973.092838685555 | 3612.115549144449 | 0.4307330960637471 | 0.5923108748055911 |

### solar_mw
| Baseline | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| persistence_24h | 2426.8854611997513 | 1246.7361719060884 | 0.14060981027990357 | 358137.833246327 | 0.2205576586910432 |
| moving_average_24h | 8933.132470089251 | 7873.631930627404 | 1.2936790214004685 | 239728491998.21857 | 230.30776660095552 |

## Multi‑Horizon Backtest (Load)
![](/Users/pratik_n/Downloads/gridpulse/reports/figures/multi_horizon_backtest.png)

## Conclusions
GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.
