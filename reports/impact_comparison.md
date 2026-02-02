# Impact Evaluation — Baseline vs GridPulse

This report compares dispatch outcomes for the same 24‑hour forecast window (last 24 hours of the test split).

- Horizon: 24 hours
- Forecast source: test split (proxy for day‑ahead forecast)
- Config: `configs/optimization.yaml`

## Policy Comparison
| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |
|---|---:|---:|---:|
| Grid‑only baseline | 9,040,750,000.00 | 101,287,750.76 | 5,064,387.54 |
| Naive battery | 9,037,380,430.08 | 101,287,750.76 | 5,064,387.54 |
| GridPulse (LP optimized) | 9,036,610,000.00 | 101,287,750.76 | 5,064,387.54 |

## Savings vs Baseline (GridPulse vs Grid‑only)
- Cost savings: 4,140,000.00 (0.05%)
- Carbon reduction: 0.00 kg (0.00%)

## Savings vs Naive Battery (GridPulse vs Naive)
- Cost savings: 770,430.08 (0.01%)
- Carbon reduction: 0.00 kg (0.00%)

## Dispatch Comparison
![](figures/dispatch_compare.png)

## Arbitrage Logic (Level-4)
![](figures/arbitrage_optimization.png)
