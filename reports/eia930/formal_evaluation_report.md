# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 162.889098360766 | 123.22699049377624 | 0.0016711108353770399 | 0.0016708040250974364 |
| lstm | 4767.13981471182 | 3835.3213548638823 | 0.052446208271787996 | 0.05144989867645077 |
| tcn | 3850.48798204528 | 2877.3076092397896 | 0.03821778287424641 | 0.03773362151594056 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 269.22959007922213 | 144.65876286602762 | 0.017458976355303666 | 463928300.7134795 |
| lstm | 6301.907170540824 | 5234.314534441705 | 0.4259245611501527 | 18240.28812852139 |
| tcn | 7187.542595933752 | 5930.744731074514 | 0.476476746555839 | 24566.311973066917 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 208.9155178581897 | 74.92639127730418 | 0.45450404712509446 | 9407839.965532975 | 0.8231692256919554 |
| lstm | 1781.9688755793472 | 1055.338939519795 | 1.3692435056156054 | 389364.1547393924 | 77.2306877272726 |
| tcn | 1743.6645818725526 | 965.0847461366311 | 1.3149781435492511 | 308397.89643664926 | 71.58159943902696 |

## Baseline Metrics (Test Split)
### load_mw
| Baseline | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| persistence_24h | 4417.818129710728 | 3273.0529179030664 | 0.0439340909523807 | 0.04359091590429231 |
| moving_average_24h | 4701.656233454611 | 3871.5704129574683 | 0.0531521136404887 | 0.05353481724135891 |

### wind_mw
| Baseline | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| persistence_24h | 8226.004438441914 | 6714.227002967359 | 0.5486600993912171 | 7000346192.6289215 |
| moving_average_24h | 5283.128350341692 | 4282.488006923838 | 0.36009115059588626 | 5092672272.135249 |

### solar_mw
| Baseline | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| persistence_24h | 1765.0475861026314 | 837.7438180019782 | 0.6593876229374671 | 15825916.523272963 | 1.6580877716664961 |
| moving_average_24h | 3343.943461756516 | 2755.7884108143753 | 1.5488311735160305 | 7169916964.417997 | 655.33050541318 |

## Multi‑Horizon Backtest (Load)
![](/Users/pratik_n/Downloads/gridpulse/reports/eia930/figures/multi_horizon_backtest.png)

## Conclusions
GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.
