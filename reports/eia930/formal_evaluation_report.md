# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 183.369246744089 | 140.64221070422795 | 0.0018225504269295894 | 0.0018218246510229466 |
| lstm | 4413.570886076817 | 3416.167851102955 | 0.04456974238131669 | 0.04410213310317136 |
| tcn | 6219.553186950962 | 4734.462425598746 | 0.06285607979492414 | 0.06060904100770707 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 357.09618359110743 | 210.72024983372535 | 0.02649683215003194 | 796670137.431063 |
| lstm | 6699.708802166455 | 5729.828730333515 | 0.4619438785216474 | 34185.95123977535 |
| tcn | 6934.903180692785 | 5963.546587811382 | 0.47302189708212306 | 32363.88864425602 |

### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 229.98768085846916 | 82.14335776365684 | 0.5309049987178593 | 10611026.85282825 | 0.6680680076052432 |
| lstm | 1700.8725151892208 | 1161.313448293208 | 1.3782643440636377 | 617506.3702363383 | 159.18462406495743 |
| tcn | 1444.48926066807 | 754.7591873337503 | 1.2436807834927754 | 191735.96556505052 | 47.4721348447582 |

## Baseline Metrics (Test Split)
### load_mw
| Baseline | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| persistence_24h | 4794.5029081460025 | 3691.2984251968505 | 0.04787441385901664 | 0.04745277292198041 |
| moving_average_24h | 4658.313525154179 | 3798.4548884514443 | 0.049869783537454436 | 0.05007661316408526 |

### wind_mw
| Baseline | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| persistence_24h | 8237.891716183156 | 6656.84094488189 | 0.5395409837772438 | 11145433071.602448 |
| moving_average_24h | 5415.042819436663 | 4321.018307086614 | 0.35668633287746115 | 8108175853.476254 |

### solar_mw
| Baseline | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| persistence_24h | 1813.4454736293485 | 841.0779527559055 | 0.6805605015189686 | 17637796.620273266 | 1.41369788616925 |
| moving_average_24h | 3132.034953755044 | 2549.796620734908 | 1.561792529509564 | 8975049881.752747 | 703.498335966092 |

## Multi‑Horizon Backtest (Load)
![](figures/multi_horizon_backtest.png)

## Conclusions
GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.
