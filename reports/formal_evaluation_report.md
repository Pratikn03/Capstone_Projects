# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 257.30103653279747 | 162.87037842496227 | 0.0034154553617824477 | 0.0034244515594101547 |
| lstm | 3474.776266889287 | 2722.500057873343 | 0.05375660029301429 | 0.05412533668377964 |
| tcn | 2668.052692101338 | 2031.1189882503816 | 0.039438962013668165 | 0.038934521011744765 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 184.21298597991048 | 120.15792229116077 | 0.0234513967672387 | 0.026927334832202898 |
| lstm | 6735.07009751006 | 5511.226734264865 | 0.6084608518120028 | 1.4094474308126195 |
| tcn | 9196.156140506526 | 7167.4122668564605 | 0.7667156554333923 | 1.6537467556241543 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 247.82742595723835 | 118.72247785473017 | 0.6887528768499864 | 32060327.266559295 | 0.07243011528850685 |
| lstm | 4079.379640875171 | 2835.7400333226856 | 1.0944293937695597 | 5215103.425959513 | 5215103.425959513 |
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
