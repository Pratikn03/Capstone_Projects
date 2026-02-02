# Impact Evaluation — Baseline vs GridPulse

This report compares dispatch outcomes for the same 24‑hour forecast window (last 24 hours of the test split).

- Horizon: 24 hours
- Forecast source: test split (proxy for day‑ahead forecast)
- Config: `configs/optimization.yaml`

## Policy Comparison
| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |
|---|---:|---:|---:|
| Grid‑only baseline | 14,879,240,000.00 | 96,000,000.00 | 4,800,000.00 |
| Naive battery | 14,875,870,430.08 | 96,000,000.00 | 4,800,000.00 |
| GridPulse (LP optimized) | 14,875,100,000.00 | 96,000,000.00 | 4,800,000.00 |

## Savings vs Baseline (GridPulse vs Grid‑only)
- Cost savings: 4,140,000.00 (0.03%)
- Carbon reduction: 0.00 kg (0.00%)

## Savings vs Naive Battery (GridPulse vs Naive)
- Cost savings: 770,430.08 (0.01%)
- Carbon reduction: 0.00 kg (0.00%)

## Dispatch Comparison
![](figures/dispatch_compare.png)
