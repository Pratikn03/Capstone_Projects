# Utility-Preserving Safety Scorecard

This scorecard separates strict safety from useful release behavior. A row passes only when ORIUS has no material excess TSVR over a domain fail-safe reference and preserves more useful work than that reference.

| Domain | Safety reference | Excess TSVR | Utility gain | Fallback reduction | Intervention reduction | Gate |
|---|---:|---:|---:|---:|---:|---:|
| Battery Energy Storage | immediate_shutdown | 0.000000 | inf | not_defined_for_battery_t8 | not_defined_for_battery_t8 | True |
| Autonomous Vehicles | always_brake | 0.000000 | 3.985808 | 0.719886 | 0.425571 | True |
| Medical and Healthcare Monitoring | always_alert | 0.000000 | inf | 0.520093 | 0.520093 | True |

## Claim-facing comparisons

| Domain | Comparison | Reference | Reference TSVR | ORIUS TSVR | TSVR reduction | Utility delta | Relation | Comparability |
|---|---|---:|---:|---:|---:|---:|---|---|
| Battery Energy Storage | shutdown_or_fallback_only_conservatism | immediate_shutdown | 0.000000 | 0.000000 | 0.000000 | 10.100000 | less_conservative_than_shutdown_or_fallback_only | comparable_fail_safe_reference |
| Autonomous Vehicles | shutdown_or_fallback_only_conservatism | always_brake | 0.241286 | 0.241200 | 0.000086 | 5869.636141 | less_conservative_than_shutdown_or_fallback_only | comparable_fail_safe_reference |
| Medical and Healthcare Monitoring | shutdown_or_fallback_only_conservatism | always_alert | 0.000000 | 0.000000 | 0.000000 | 142767.000000 | less_conservative_than_shutdown_or_fallback_only | comparable_fail_safe_reference |
| Battery Energy Storage | predictor_only_safety | stronger_predictor_without_runtime_adaptation (deep:dc3s_wrapped) | 0.000000 | 0.000000 | 0.000000 | metric_not_reported_in_compact_comparator | no_observed_safety_separation | non_comparable_no_observed_safety_separation |
| Autonomous Vehicles | predictor_only_safety | stronger_predictor_without_runtime_adaptation (predictor_only_no_runtime) | 0.289250 | 0.000163 | 0.289087 | metric_not_reported_in_compact_comparator | safer_than_predictor_only | comparable_runtime_native |
| Medical and Healthcare Monitoring | predictor_only_safety | stronger_predictor_without_runtime_adaptation (predictor_only_no_runtime) | 0.194489 | 0.000000 | 0.194489 | metric_not_reported_in_compact_comparator | safer_than_predictor_only | comparable_runtime_native |

## Required ablation surfaces

| Surface | Domain | Evidence | Baseline TSVR | ORIUS TSVR | TSVR reduction | Comparability |
|---|---|---|---:|---:|---:|---|
| no_certificate_gate | Autonomous Vehicles | runtime_denominator | 0.263610 | 0.000163 | 0.263447 | non_comparable_combined_certificate_temporal_guard |
| no_fallback | Autonomous Vehicles | runtime_denominator | 0.263610 | 0.000163 | 0.263447 | non_comparable_combined_with_temporal_guard |
| no_reliability | Autonomous Vehicles | runtime_denominator | 0.252319 | 0.000163 | 0.252156 | comparable_runtime_native |
| no_repair | Autonomous Vehicles | runtime_denominator | 0.289250 | 0.000163 | 0.289087 | comparable_runtime_native |
| no_uncertainty | Autonomous Vehicles | not_isolated_in_compact_evidence | 0.289250 | 0.000163 | 0.289087 | non_comparable_combined_nominal_no_uncertainty_surface |
| no_certificate_gate | Battery Energy Storage | runtime_denominator | 0.000000 | 0.000000 | 0.000000 | non_comparable_combined_certificate_temporal_guard |
| no_fallback | Battery Energy Storage | runtime_denominator | 0.000000 | 0.000000 | 0.000000 | non_comparable_combined_with_temporal_guard |
| no_reliability | Battery Energy Storage | runtime_denominator | 0.000000 | 0.000000 | 0.000000 | comparable_runtime_native |
| no_repair | Battery Energy Storage | runtime_denominator | 0.000000 | 0.000000 | 0.000000 | comparable_runtime_native |
| no_uncertainty | Battery Energy Storage | not_isolated_in_compact_evidence | 0.000000 | 0.000000 | 0.000000 | non_comparable_combined_nominal_no_uncertainty_surface |
| no_certificate_gate | Medical and Healthcare Monitoring | runtime_denominator | 0.000029 | 0.000000 | 0.000029 | non_comparable_combined_certificate_temporal_guard |
| no_fallback | Medical and Healthcare Monitoring | runtime_denominator | 0.000029 | 0.000000 | 0.000029 | non_comparable_combined_with_temporal_guard |
| no_reliability | Medical and Healthcare Monitoring | runtime_denominator | 0.200420 | 0.000000 | 0.200420 | comparable_runtime_native |
| no_repair | Medical and Healthcare Monitoring | runtime_denominator | 0.194489 | 0.000000 | 0.194489 | comparable_runtime_native |
| no_uncertainty | Medical and Healthcare Monitoring | not_isolated_in_compact_evidence | 0.194489 | 0.000000 | 0.194489 | non_comparable_combined_nominal_no_uncertainty_surface |
| no_signature_hash_gate | Cross-domain governance | governance_fail_closed | not_a_tsvr_metric | not_a_tsvr_metric | not_a_tsvr_metric | non_comparable_governance_gate |

Claim boundary: this is bounded predeployment evidence. It does not claim road deployment, live clinical deployment, or physical battery field certification.
