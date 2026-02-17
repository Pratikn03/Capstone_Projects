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
| GridPulse (forecast‑optimized) | 143,776,429.88 | 1,457,205,025.31 | 72,860,251.27 |
| Risk‑aware (interval) | 145,680,826.66 | 1,484,593,261.15 | 74,229,663.06 |
| Oracle upper bound (perfect forecast) | 152,331,496.60 | 1,451,467,789.95 | 72,573,389.50 |

## Savings vs Baseline (GridPulse vs Grid‑only)
- Cost savings: 10,997,032.84 (7.11%)
- Carbon reduction: 4,436,217.97 kg (0.30%)

- Carbon source used for optimization: average

## Savings vs Naive Battery (GridPulse vs Naive)
- Cost savings: 10,424,515.76 (6.76%)
- Carbon reduction: -2,547,126.28 kg (-0.18%)

## Oracle Gap (GridPulse vs Perfect‑Forecast Upper Bound)
- Oracle cost: 152,331,496.60
- Gap vs oracle: -8,555,066.72

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
