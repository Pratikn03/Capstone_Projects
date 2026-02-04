# Impact Evaluation — Baseline vs GridPulse

This report compares dispatch outcomes for the same 7‑day forecast window (selected from the test split).

- Horizon: 168 hours (7 days)
- Window index: 456–624
- Forecast source: test split (proxy for day‑ahead forecast)
- Config: `configs/optimization.yaml`

## Policy Comparison
| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |
|---|---:|---:|---:|
| Grid‑only baseline | 154,773,462.72 | 1,461,641,243.28 | 73,082,062.16 |
| Naive battery | 154,200,945.64 | 1,454,657,899.03 | 72,732,894.95 |
| Peak‑shaving heuristic | 155,096,309.40 | 1,465,616,652.67 | 73,280,832.63 |
| Price‑greedy (MPC‑style) | 154,098,659.76 | 1,457,458,733.64 | 72,872,936.68 |
| GridPulse (forecast‑optimized) | 150,305,710.89 | 1,453,149,137.03 | 72,657,456.85 |
| Oracle upper bound (perfect forecast) | 150,305,710.89 | 1,453,149,137.03 | 72,657,456.85 |

## Savings vs Baseline (GridPulse vs Grid‑only)
- Cost savings: 4,467,751.83 (2.89%)
- Carbon reduction: 8,492,106.26 kg (0.58%)

- Carbon source used for optimization: average

## Savings vs Naive Battery (GridPulse vs Naive)
- Cost savings: 3,895,234.75 (2.53%)
- Carbon reduction: 1,508,762.01 kg (0.10%)

## Oracle Gap (GridPulse vs Perfect‑Forecast Upper Bound)
- Oracle cost: 150,305,710.89
- Gap vs oracle: 0.00

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
