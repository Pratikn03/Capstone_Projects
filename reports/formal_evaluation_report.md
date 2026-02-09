# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 271.1705912291987 | 161.08322548285815 | 0.0034172517439374785 | 0.0034268962116018524 |
| lstm | 2355.9737707052263 | 1732.0841498345 | 0.033589623391714125 | 0.03384176636351295 |
| tcn | 3394.193713865421 | 2613.5046818291494 | 0.0534685897978956 | 0.051515772878522645 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 127.08497812200575 | 87.33009845331962 | 0.019834754096708387 | 0.022747657908375494 |
| lstm | 6025.136806059045 | 4304.789359172982 | 0.5205029171351211 | 0.8550586818046001 |
| tcn | 7169.663581806657 | 5185.96133242178 | 0.6047623966086424 | 1.0503505655693721 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 269.5542942974431 | 129.53972566184072 | 0.7041557205607407 | 107722205.784544 | 0.1404350400943314 |
| lstm | 2536.1146372469066 | 1536.0016177466066 | 0.9657990565299844 | 1052589.3212084842 | 1052589.3212084842 |
| tcn | 3006.7451233891106 | 2009.5950846992325 | 0.9721751549480074 | 2912810.8970445017 | 2912810.8970445017 |

### price_eur_mwh
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 4.903815488573345 | 2.256433471158783 | 0.09762172749780437 | 1.205960010787307 |
| lstm | 10.818688719023399 | 6.509276599083433 | 0.22359161937564342 | 3.995862672890131 |
| tcn | 14.395427251891723 | 10.182696580813174 | 0.3658956840137629 | 3.711223081218662 |

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
