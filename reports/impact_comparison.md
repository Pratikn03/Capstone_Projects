# Impact Evaluation — Baseline vs GridPulse

This report compares dispatch outcomes for the same 7‑day forecast window (selected from the test split).

- Horizon: 168 hours (7 days)
- Window index: 360–528
- Forecast source: test split (proxy for day‑ahead forecast)
- Config: `configs/optimization.yaml`

## Policy Comparison
| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |
|---|---:|---:|---:|
| Grid‑only baseline | 155,149,015.93 | 1,798,244,455.25 | 89,912,222.76 |
| Naive battery | 154,864,061.39 | 1,795,398,663.85 | 89,769,933.19 |
| Peak‑shaving heuristic | 154,589,974.86 | 1,796,355,708.27 | 89,817,785.41 |
| Price‑greedy (MPC‑style) | 154,394,539.33 | 1,794,285,756.69 | 89,714,287.83 |
| GridPulse (forecast‑optimized) | 136,362,150.56 | 1,851,253,975.78 | 92,562,698.79 |
| Risk‑aware (interval) | 139,239,837.31 | 1,908,494,924.16 | 95,424,746.21 |
| Oracle upper bound (perfect forecast) | 152,464,282.92 | 1,787,892,936.41 | 89,394,646.82 |

## Savings vs Baseline (GridPulse vs Grid‑only)
- Cost savings: 18,786,865.37 (12.11%)
- Carbon reduction: -53,009,520.53 kg (-2.95%)

- Carbon source used for optimization: average

## Savings vs Naive Battery (GridPulse vs Naive)
- Cost savings: 18,501,910.83 (11.95%)
- Carbon reduction: -55,855,311.92 kg (-3.11%)

## Oracle Gap (GridPulse vs Perfect‑Forecast Upper Bound)
- Oracle cost: 152,464,282.92
- Gap vs oracle: -16,102,132.35

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
