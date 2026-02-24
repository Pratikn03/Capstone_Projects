# Impact Evaluation — Baseline vs GridPulse

This report compares dispatch outcomes for the same 7‑day forecast window (selected from the test split).

- Horizon: 168 hours (7 days)
- Window index: 432–600
- Forecast source: test split (proxy for day‑ahead forecast)
- Config: `configs/optimization.yaml`

## Policy Comparison
| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |
|---|---:|---:|---:|
| Grid‑only baseline | 155,555,570.42 | 1,799,672,005.70 | 89,983,600.28 |
| Naive battery | 154,664,978.99 | 1,795,929,078.13 | 89,796,453.91 |
| Peak‑shaving heuristic | 154,995,830.13 | 1,797,739,516.89 | 89,886,975.84 |
| Price‑greedy (MPC‑style) | 154,801,354.69 | 1,795,786,037.20 | 89,789,301.86 |
| GridPulse (forecast‑optimized) | 152,870,837.41 | 1,789,320,486.86 | 89,466,024.34 |
| Oracle upper bound (perfect forecast) | 152,870,837.41 | 1,789,320,486.86 | 89,466,024.34 |

## Savings vs Baseline (GridPulse vs Grid‑only)
- Cost savings: 2,684,733.01 (1.73%)
- Carbon reduction: 10,351,518.84 kg (0.58%)

- Carbon source used for optimization: average

## Savings vs Naive Battery (GridPulse vs Naive)
- Cost savings: 1,794,141.59 (1.16%)
- Carbon reduction: 6,608,591.27 kg (0.37%)

## Oracle Gap (GridPulse vs Perfect‑Forecast Upper Bound)
- Oracle cost: 152,870,837.41
- Gap vs oracle: 0.00

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
