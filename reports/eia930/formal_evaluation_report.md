# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 165.23422675880101 | 125.747170145024 | 0.0017031494376180197 | 0.0017027020867110027 |
| lstm | 3684.7258041228997 | 2826.066128177769 | 0.03795016661539154 | 0.03753817637505856 |
| tcn | 4235.372144626466 | 3251.1130439674826 | 0.04341264408873344 | 0.04256222082515162 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 310.2093820736406 | 182.0326613105617 | 0.02123504076366505 | 527148926.7790369 |
| lstm | 6732.315044593082 | 5393.442130282138 | 0.4528177330216119 | 12754.993780015144 |
| tcn | 6969.003001818571 | 5863.4087310658515 | 0.4753333608293543 | 16490.17437712878 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 203.81657283172586 | 74.67275747690604 | 0.46211458014678414 | 10338752.34764313 | 0.7277483588390152 |
| lstm | 1607.664446880507 | 985.0155340662252 | 1.3414418125495307 | 393034.06120136444 | 84.37122030851546 |
| tcn | 1398.1378133547073 | 835.0464982482388 | 1.302093275861052 | 451197.8101677131 | 92.79426025536415 |

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
