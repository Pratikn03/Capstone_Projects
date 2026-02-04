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
| Naive battery | 154,336,933.24 | 1,458,193,528.79 | 72,909,676.44 |
| Peak‑shaving heuristic | 154,885,904.76 | 1,464,598,111.31 | 73,229,905.57 |
| Price‑greedy (MPC‑style) | 154,437,655.07 | 1,459,495,994.70 | 72,974,799.74 |
| GridPulse (forecast‑optimized) | 318,349,817.86 | 1,447,024,830.85 | 72,351,241.54 |
| Oracle upper bound (perfect forecast) | 318,349,817.86 | 1,447,024,830.85 | 72,351,241.54 |

## Savings vs Baseline (GridPulse vs Grid‑only)
- Cost savings: -163,576,355.14 (-105.69%)
- Carbon reduction: 14,616,412.43 kg (1.00%)

- Carbon source used for optimization: average

## Savings vs Naive Battery (GridPulse vs Naive)
- Cost savings: -164,012,884.62 (-106.27%)
- Carbon reduction: 11,168,697.94 kg (0.77%)

## Oracle Gap (GridPulse vs Perfect‑Forecast Upper Bound)
- Oracle cost: 318,349,817.86
- Gap vs oracle: 0.00

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
