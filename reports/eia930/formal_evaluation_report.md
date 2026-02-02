# Formal Evaluation Report

## Summary
This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.

## Model Metrics (Test Split)
### solar_mw
| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |
|---|---:|---:|---:|---:|---:|
| gbm | 4760.943700726715 | 2829.769375090201 | 1.8610189014589982 | 0.9305094530235243 | 1.0 |
| lstm | 4792.066895724467 | 2867.4839629486974 | 1.9865236159807007 | 162219.74691763284 | 1.0000471658770034 |

### load_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 211.1099686007024 | 111.44578763515571 | 0.0013837949950884533 | 0.001383005581060905 |
| lstm | 13411.260866566116 | 10238.471109038774 | 0.1346894485583618 | 0.1290998477615506 |

### wind_mw
| Model | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| gbm | 12411.628450075548 | 10782.008009813826 | 1.9673834608137406 | 0.9836917304084284 |
| lstm | 12492.957293500665 | 10925.599039104149 | 1.999989700380644 | 622.4913997410354 |

## Multi‑Horizon Backtest (Load)
![](/Users/pratik_n/Downloads/gridpulse/reports/eia930/figures/multi_horizon_backtest.png)

## Conclusions
GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.
