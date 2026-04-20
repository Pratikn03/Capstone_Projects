# ORIUS Claim Ledger — 3-Domain Flagship Surface

> Every headline ORIUS claim must resolve to a current artifact in the promoted Battery + AV + Healthcare lane.
> If a statement cannot be traced to a live artifact, it must be narrowed or deleted.

## Canonical Reference Set

- `reports/battery_av_healthcare/overall/release_summary.json`
- `reports/publication/orius_submission_scorecard.csv`
- `reports/publication/three_domain_ml_benchmark.csv`
- `reports/publication/three_domain_reliability_calibration.csv`
- `reports/publication/three_domain_grouped_coverage.csv`
- `reports/publication/three_domain_nonvacuity_checks.json`
- `reports/publication/novelty_separation_matrix.csv`
- `reports/publication/what_orius_is_not_matrix.csv`
- `reports/publication/active_theorem_audit.json`

The flagship novelty sentence is:

> ORIUS identifies OASG as the degraded-observation release hazard and provides a reliability-aware runtime safety layer across Battery, AV, and Healthcare.

## Bucket A — Locked Headline Claims

| ID | Claim | Artifact source |
|----|-------|-----------------|
| A1 | Battery witness row reduces TSVR from `0.0393` to `0.0000` on the locked publication-nominal surface. | `reports/publication/three_domain_ml_benchmark.csv` |
| A2 | AV bounded defended row reduces TSVR from `0.1250` to `0.0417`. | `reports/publication/three_domain_ml_benchmark.csv` |
| A3 | Healthcare bounded defended row reduces TSVR from `0.2917` to `0.0417` on the promoted MIMIC-backed monitoring row. | `reports/publication/three_domain_ml_benchmark.csv` |
| A4 | Grouped reliability-bucket calibration is emitted for Battery, AV, and Healthcare with non-empty buckets. | `reports/publication/three_domain_grouped_coverage.csv`, `reports/publication/three_domain_nonvacuity_checks.json` |
| A5 | The active readiness target is `three_domain_93_candidate`, with `critical_gap_count = 0`, `high_gap_count = 0`, and `meets_93_gate = True`. | `reports/publication/orius_submission_scorecard.csv` |
| A6 | The flagship novelty surface is separated explicitly from conformal prediction, runtime assurance, safe-control methods, drift detection, and generic uncertainty estimation. | `reports/publication/novelty_separation_matrix.csv` |

## Bucket B — Bounded / Qualified Claims

| ID | Claim | Required qualification | Artifact source |
|----|-------|------------------------|-----------------|
| B1 | Battery is the deepest proof and artifact surface. | Battery is the witness row, not proof that every promoted domain has equal depth. | `reports/battery_av_healthcare/overall/release_summary.json` |
| B2 | AV is defended. | AV remains bounded to the TTC plus predictive-entry-barrier contract; it is not full autonomous-driving closure. | `reports/battery_av_healthcare/overall/release_summary.json` |
| B3 | Healthcare is defended. | Healthcare remains bounded to monitoring-and-alert semantics on the promoted MIMIC row; it is not clinical deployment readiness. | `reports/battery_av_healthcare/overall/release_summary.json` |
| B4 | ORIUS has a 3-domain baseline and ablation package. | `three_domain_baseline_suite.csv` and `three_domain_ablation_matrix.csv` are diagnostic cross-domain proxy surfaces, not replacements for the deeper battery witness surface. | `reports/publication/three_domain_baseline_suite.csv`, `reports/publication/three_domain_ablation_matrix.csv` |
| B5 | ORIUS has a strong calibration story. | The promoted calibration claim is grouped reliability-bucket calibration under degraded observation, not a new conditional-coverage theorem. | `reports/publication/three_domain_reliability_calibration.csv`, `reports/publication/what_orius_is_not_matrix.csv` |
| B6 | Learned reliability contributes novelty. | Deep or learned reliability remains a bounded secondary lane unless it clears the same 3-domain gates as the engineered reliability path. | `paper/monograph/ch08_witness_results_and_failure_analysis.tex` |

## Bucket C — Explicit Non-Claims

| ID | ORIUS does **not** claim | Why |
|----|---------------------------|-----|
| C1 | A new universal controller | ORIUS wraps inherited domain controllers rather than replacing them. |
| C2 | A new conditional-coverage theorem | The active ML surface is grouped calibration under degraded observation. |
| C3 | Better forecasting by default | Forecast quality matters only insofar as it improves runtime release safety. |
| C4 | Full autonomous-driving closure | The AV row is bounded to the promoted longitudinal contract. |
| C5 | Clinical deployment readiness | The Healthcare row is bounded to monitoring-and-alert semantics. |

## Audit Rules

1. Every promoted number must trace to the canonical reference set above.
2. Every Bucket B claim must carry its qualification in the same paragraph or caption.
3. No promoted prose may imply Bucket C.
4. Battery is always the witness row; AV and Healthcare are bounded defended rows.
5. The active theorem audit is the theory authority; headline ML and novelty credit does not come from draft theorem rows.
