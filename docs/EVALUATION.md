# Evaluation

This document explains how GridPulse is evaluated and how impact metrics are computed.

## Forecasting metrics
- **RMSE**, **MAE**, **sMAPE** for all targets
- **Daylight MAPE** for solar (avoids night-time inflation)

Metrics are computed on the time-based test split to avoid leakage.

## Backtesting
- Walk-forward evaluation and multi-horizon backtests
- Outputs saved in `reports/` (OPSD) and `reports/eia930/`

## Impact evaluation (Level-4)
GridPulse compares optimized dispatch against two baselines:
1) **Grid-only baseline** (no battery shifting)
2) **Naive battery baseline** (charge night, discharge evening peak)

Impact metrics:
- **Cost savings (%)**
- **Carbon reduction (%)**
- **Peak shaving (%)**

These are computed by `scripts/build_reports.py` and written to:
- `reports/impact_summary.csv`
- `reports/eia930/impact_summary.csv`

## Price and carbon signals
- OPSD uses `DE_price_day_ahead` if available (mapped to `price_eur_mwh`).
- EIA-930 uses a time-of-day proxy (`price_usd_mwh`) unless a real tariff is provided.
- Carbon intensity (`carbon_kg_per_mwh`) is time-varying to enable carbon-aware dispatch.

If price and carbon are constant, cost/carbon savings will be near zero.

## Updating README
After generating reports:
```bash
python scripts/update_readme_impact.py
```
This keeps the README impact table consistent with the latest benchmark.
