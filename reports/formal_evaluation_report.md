# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 311.72480845071334 | 201.5231471613508 | 0.004206509660543428 | 0.004218388597407557 |
| lstm | 3150.8857113651957 | 2531.3296529479308 | 0.05032593909273349 | 0.05200173489268551 |
| tcn | 3253.0385867209866 | 2489.810464814017 | 0.04891888069365588 | 0.04955627721657126 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 254.6457131201481 | 173.27652305638426 | 0.03233788142797395 | 0.03604640667786144 |
| lstm | 5545.723678944462 | 4065.8456184524216 | 0.492966269752505 | 0.8389672730164671 |
| tcn | 6784.753312539991 | 5141.186704237131 | 0.6011567438651633 | 1.1245434376104424 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 277.2354565494422 | 137.40184591312502 | 0.7033863301649446 | 88802171.19391449 | 0.17588067168074012 |
| lstm | 3033.734595419477 | 2208.320710398724 | 1.0265053780522089 | 3734370.787846106 | 3734370.787846106 |
| tcn | 2445.89895524222 | 1581.4788292001956 | 0.9566574762464122 | 2608843.4612632683 | 2608843.4612632683 |

### price_eur_mwh
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 4.917888404934221 | 2.320058018981045 | 0.10333153749453926 | 1.5186917143986867 |
| lstm | 11.710671927383126 | 7.253959331977834 | 0.2358981109866176 | 4.9806048700758785 |
| tcn | 11.872452199284453 | 7.301264289636419 | 0.23310885286929614 | 5.612377660047014 |

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
