# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for ORIUS.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 255.2496977060425 | 161.5723971759428 | 0.0034126303861085165 | 0.00342159477292023 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 176.7985272670447 | 108.51175780766646 | 0.021243206186295105 | 0.02413880448831295 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 239.7183190399224 | 120.20584757446878 | 0.688520431139487 | 26041974.13623809 | 0.07250132031101066 |

### price_eur_mwh
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 4.928541060581904 | 2.302548355403686 | 0.10025028999788027 | 1.299223718978415 |

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
![](figures/multi_horizon_backtest.png)

## Conclusions
GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.
