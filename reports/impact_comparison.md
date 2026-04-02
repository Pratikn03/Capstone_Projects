# Impact Evaluation — Baseline vs ORIUS

This report compares dispatch outcomes for the same 7‑day forecast window (selected from the test split).

- Horizon: 168 hours (7 days)
- Window index: 360–528
- Forecast source: test split (proxy for day‑ahead forecast)
- Config: `configs/optimization.yaml`

## Policy Comparison
| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |
|---|---:|---:|---:|
| Grid‑only baseline | 155,149,015.93 | 1,798,244,455.25 | 89,912,222.76 |
| Naive battery | 155,133,643.93 | 1,798,360,428.37 | 89,918,021.42 |
| Peak‑shaving heuristic | 155,141,146.65 | 1,798,897,403.45 | 89,944,870.17 |
| Price‑greedy (MPC‑style) | 154,660,436.54 | 1,796,900,267.50 | 89,845,013.38 |
| ORIUS (forecast‑optimized) | 138,658,355.32 | 1,858,870,160.54 | 92,943,508.03 |
| Risk‑aware (interval) | 141,536,042.07 | 1,916,111,108.93 | 95,805,555.45 |
| Oracle upper bound (perfect forecast) | 154,760,487.68 | 1,795,509,121.18 | 89,775,456.06 |

## Savings vs Baseline (ORIUS vs Grid‑only)
- Cost savings: 16,490,660.61 (10.63%)
- Carbon reduction: -60,625,705.29 kg (-3.37%)

- Carbon source used for optimization: average

## Savings vs Naive Battery (ORIUS vs Naive)
- Cost savings: 16,475,288.61 (10.62%)
- Carbon reduction: -60,509,732.16 kg (-3.36%)

## Oracle Gap (ORIUS vs Perfect‑Forecast Upper Bound)
- Oracle cost: 154,760,487.68
- Gap vs oracle: -16,102,132.35

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
