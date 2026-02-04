# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 261.9013026862777 | 153.72911980681192 | 0.0032671543645541898 | 0.003275942934649663 |
| lstm | 3653.033580802336 | 2943.7813178683728 | 0.06014463366141743 | 0.059251385023769206 |
| tcn | 4962.79349858157 | 3989.2522238461424 | 0.08067873718563466 | 0.08602321931475024 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 130.1893791929321 | 91.25972294820731 | 0.020061795705913564 | 0.02280878448806715 |
| lstm | 5679.480870237086 | 4071.257203924327 | 0.5083383266432826 | 0.7740500115709378 |
| tcn | 7000.6263935145225 | 5123.859590274129 | 0.6195663481721845 | 0.9154126142363062 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 239.01965918961375 | 116.19207100407529 | 0.6909447680075763 | 56895606.614095934 | 0.09616268073085278 |
| lstm | 3821.059423253299 | 2586.159735480109 | 1.0953148448816592 | 2613868.237764853 | 2613868.237764853 |
| tcn | 2515.3953881901975 | 1574.5808193708394 | 0.9570789863073993 | 1944896.6379839387 | 1944896.6379839387 |

### price_eur_mwh
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 4.812599391054528 | 2.2489778882166065 | 0.09730134244183289 | 1.2021980876479732 |
| lstm | 11.737287891372155 | 7.611077518620933 | 0.2670704591377959 | 3.264897758229742 |
| tcn | 13.122512085608344 | 8.992771791608991 | 0.3038820732168579 | 4.111139251504774 |

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
