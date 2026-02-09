# Impact Evaluation — Baseline vs GridPulse

This report compares dispatch outcomes for the same 7‑day forecast window (selected from the test split).

- Horizon: 168 hours (7 days)
- Window index: 432–600
- Forecast source: test split (proxy for day‑ahead forecast)
- Config: `configs/optimization.yaml`

## Policy Comparison
| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |
|---|---:|---:|---:|
| Grid‑only baseline | 461,364,890.00 | 3,907,803,618.32 | 195,390,180.92 |
| Naive battery | 462,821,035.03 | 3,911,194,390.28 | 195,559,719.51 |
| Peak‑shaving heuristic | 461,540,022.90 | 3,908,660,249.57 | 195,433,012.48 |
| Price‑greedy (MPC‑style) | 460,068,067.28 | 3,925,065,173.59 | 196,253,258.68 |
| GridPulse (forecast‑optimized) | 459,394,985.65 | 3,908,934,723.03 | 195,446,736.15 |
| Oracle upper bound (perfect forecast) | 459,394,985.65 | 3,908,934,723.03 | 195,446,736.15 |

## Savings vs Baseline (GridPulse vs Grid‑only)
- Cost savings: 1,969,904.35 (0.43%)
- Carbon reduction: -1,131,104.71 kg (-0.03%)

- Carbon source used for optimization: average

## Savings vs Naive Battery (GridPulse vs Naive)
- Cost savings: 3,426,049.38 (0.74%)
- Carbon reduction: 2,259,667.24 kg (0.06%)

## Oracle Gap (GridPulse vs Perfect‑Forecast Upper Bound)
- Oracle cost: 459,394,985.65
- Gap vs oracle: 0.00

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
