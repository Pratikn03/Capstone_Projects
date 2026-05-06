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
| ORIUS (forecast‑optimized) | 138,674,954.97 | 1,859,022,585.09 | 92,951,129.25 |
| Risk‑aware (interval) | 141,535,391.56 | 1,913,347,052.68 | 95,667,352.63 |
| Oracle upper bound (perfect forecast) | 154,760,487.68 | 1,795,509,121.18 | 89,775,456.06 |

## Savings vs Baseline (ORIUS vs Grid‑only)
- Cost savings: 16,474,060.96 (10.62%)
- Carbon reduction: -60,778,129.84 kg (-3.38%)

- Carbon source used for optimization: average

## Savings vs Naive Battery (ORIUS vs Naive)
- Cost savings: 16,458,688.96 (10.61%)
- Carbon reduction: -60,662,156.71 kg (-3.37%)

## Oracle Gap (ORIUS vs Perfect‑Forecast Upper Bound)
- Oracle cost: 154,760,487.68
- Gap vs oracle: -16,085,532.71

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
