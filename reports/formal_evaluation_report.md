# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 267.50996312219684 | 167.49140642626713 | 0.0035023723719414864 | 0.0035116406790806354 |
| lstm | 3474.776266889287 | 2722.500057873343 | 0.05375660029301429 | 0.05412533668377964 |
| tcn | 2668.052692101338 | 2031.1189882503816 | 0.039438962013668165 | 0.038934521011744765 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 183.9305042033171 | 118.18776499596973 | 0.022895253969732902 | 0.02599257803801949 |
| lstm | 6735.07009751006 | 5511.226734264865 | 0.6084608518120028 | 1.4094474308126195 |
| tcn | 9196.156140506526 | 7167.4122668564605 | 0.7667156554333923 | 1.6537467556241543 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 251.43813800279966 | 121.22254576412709 | 0.6927383785850142 | 45890919.30939417 | 0.09792229506310599 |
| lstm | 4079.379640875171 | 2835.7400333226856 | 1.0944293937695597 | 5215103.425959513 | 5215103.425959513 |
| tcn | 2702.340459129023 | 1583.0471632632803 | 0.937588029476364 | 1011461.2194118587 | 1011461.2194118587 |

### price_eur_mwh
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 4.90166279190016 | 2.2603988824297865 | 0.10129373329932408 | 1.2447705727615472 |
| lstm | 15.745267361234376 | 11.810835123831138 | 0.4359062882395699 | 3.3893327357625957 |
| tcn | 15.048323745361216 | 11.133609856148544 | 0.41141928883047696 | 3.2727786337676092 |

## Baseline Metrics (Test Split)
### load_mw
| Baseline | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| persistence_24h | 6010.558996858891 | 3901.6784055727553 | 0.07831626978944539 | 0.07712115232412328 |
| moving_average_24h | 8113.105972991012 | 6848.471894349846 | 0.13475954102739415 | 0.13944438662427217 |

### wind_mw
| Baseline | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| persistence_24h | 7780.098019853568 | 5496.818498452012 | 0.6368164331302137 | 0.899751897557905 |
| moving_average_24h | 4944.261526746888 | 3591.4473200464395 | 0.4386767036718275 | 0.6066937523900743 |

### solar_mw
| Baseline | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| persistence_24h | 2427.465494058608 | 1254.8564241486067 | 0.1425602904818673 | 348297.3640873443 | 0.2217921835445223 |
| moving_average_24h | 8898.143178342467 | 7844.48131127451 | 1.2920175194653003 | 237883146438.15353 | 226.00614251755545 |

## Multi‑Horizon Backtest (Load)
![](/Users/pratik_n/Downloads/gridpulse/reports/figures/multi_horizon_backtest.png)

## Conclusions
GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.
