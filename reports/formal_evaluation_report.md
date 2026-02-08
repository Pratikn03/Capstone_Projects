# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 298.6000157184173 | 185.93743041795145 | 0.003911473903321219 | 0.003922968131321532 |
| lstm | 4975.172724385955 | 3633.784332033787 | 0.07096101031748166 | 0.0689612324628587 |
| tcn | 6157.075359090004 | 5172.092866547439 | 0.106351766897947 | 0.09933457774538754 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 125.59034623881816 | 84.84647259377734 | 0.01948417381524536 | 0.0225560790273501 |
| lstm | 6088.681917184256 | 4712.231697154939 | 0.5649990812670935 | 1.082712566266003 |
| tcn | 7086.184863970565 | 5011.337688603397 | 0.597803473619078 | 0.959135282851123 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 263.92894737160117 | 123.88871160744712 | 0.6955383652478094 | 56557498.433411874 | 0.0876112802795579 |
| lstm | 4629.5762336389425 | 3219.9212915547964 | 1.1738874543958908 | 3654259.587926467 | 3654259.587926467 |
| tcn | 2768.8137274165933 | 2114.2282694061646 | 0.9969703950115268 | 6170236.228625847 | 6170236.228625847 |

### price_eur_mwh
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 4.985200733310823 | 2.305719713965528 | 0.09932392016733044 | 1.3343300999708962 |
| lstm | 12.443363072235742 | 8.026678932514463 | 0.2614733758907562 | 4.482997862532782 |
| tcn | 15.162372330706507 | 11.124470604269806 | 0.3932856926555346 | 3.8265930747892924 |

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
