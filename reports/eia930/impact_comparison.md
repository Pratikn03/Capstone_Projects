# Impact Evaluation — Baseline vs ORIUS

This report compares dispatch outcomes for the same 7‑day forecast window (selected from the test split).

- Horizon: 168 hours (7 days)
- Window index: 744–912
- Forecast source: test split (proxy for day‑ahead forecast)
- Config: `configs/optimization.yaml`

## Policy Comparison
| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |
|---|---:|---:|---:|
| Grid‑only baseline | 520,523,500.00 | 4,470,300,483.57 | 223,515,024.18 |
| Naive battery | 520,612,500.00 | 4,470,923,996.86 | 223,546,199.84 |
| Peak‑shaving heuristic | 520,874,976.74 | 4,472,648,015.69 | 223,632,400.78 |
| Price‑greedy (MPC‑style) | 520,629,737.05 | 4,472,815,283.52 | 223,640,764.18 |
| ORIUS (forecast‑optimized) | 520,262,250.00 | 4,468,230,307.98 | 223,411,515.40 |
| Oracle upper bound (perfect forecast) | 520,262,250.00 | 4,468,230,307.98 | 223,411,515.40 |

## Savings vs Baseline (ORIUS vs Grid‑only)
- Cost savings: 261,250.00 (0.05%)
- Carbon reduction: 2,070,175.59 kg (0.05%)

- Carbon source used for optimization: average

## Savings vs Naive Battery (ORIUS vs Naive)
- Cost savings: 350,250.00 (0.07%)
- Carbon reduction: 2,693,688.88 kg (0.06%)

## Oracle Gap (ORIUS vs Perfect‑Forecast Upper Bound)
- Oracle cost: 520,262,250.00
- Gap vs oracle: 0.00

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
