# Utility-Preserving Safety Scorecard

This scorecard separates strict safety from useful release behavior. A row passes only when ORIUS has no material excess TSVR over a domain fail-safe reference and preserves more useful work than that reference.

| Domain | Safety reference | Excess TSVR | Utility gain | Fallback reduction | Intervention reduction | Gate |
|---|---:|---:|---:|---:|---:|---:|
| Battery Energy Storage | immediate_shutdown | 0.000000 | inf |  |  | True |
| Autonomous Vehicles | always_brake | 0.000000 | 3.985808 | 0.719886 | 0.425571 | True |
| Medical and Healthcare Monitoring | always_alert | 0.000000 | inf | 0.520093 | 0.520093 | True |

Claim boundary: this is bounded predeployment evidence. It does not claim road deployment, live clinical deployment, or physical battery field certification.
