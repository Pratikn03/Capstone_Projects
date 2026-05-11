# ORIUS Claim Ledger — Three-Domain Runtime Package

> Generated from the canonical runtime, theorem, and utility-preserving safety artifacts.

ORIUS provides a reliability-aware runtime safety layer for physical AI under degraded observation, enforcing certificate-backed action release through uncertainty coverage, repair, and fallback.

## Theory Presentation Rule

For submission writing, the theorem package should be presented as five
headline results: observation-aware safety gap, observation necessity, ORIUS
safety preservation/risk envelope, temporal certificate/fallback discipline,
and typed transfer with covered observation-ambiguity optimality.  The permanent
T1--T11 labels remain registry and audit identifiers.  T5 is definitional,
T8 is supporting, and T9/T10 remain scoped extensions unless the generated
promotion gates explicitly upgrade them.

## Governing Inputs

- `reports/publication/three_domain_ml_benchmark.csv`
- `reports/publication/three_domain_forecast_calibration_runtime_evidence.csv`
- `reports/publication/runtime_release_contract_witnesses.csv`
- `reports/publication/utility_preserving_safety_scorecard.csv`
- `reports/publication/active_theorem_audit.csv`
- `submission_scope=battery_av_healthcare`

## Bucket A — Directly Artifact-Backed

| ID | Claim | Governing Artifact |
|----|-------|-------------------|
| A1 | Battery remains the `reference` witness row. | `reports/publication/three_domain_ml_benchmark.csv` |
| A2 | Battery baseline TSVR is 0.83% and ORIUS TSVR is 0.0%. | `reports/battery_av/battery/runtime_summary.csv`, `reports/publication/three_domain_ml_benchmark.csv` |
| A3 | AV remains a bounded `runtime_contract_closed` row. | `reports/publication/three_domain_ml_benchmark.csv` |
| A4 | AV baseline TSVR is 28.925% and ORIUS TSVR is 0.0163% on the promoted nuPlan runtime denominator. | `reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_summary.csv` |
| A5 | Healthcare remains a bounded `runtime_contract_closed` row. | `reports/publication/three_domain_ml_benchmark.csv` |
| A6 | Healthcare baseline TSVR is 19.4489% and ORIUS TSVR is 0.0% on the promoted retrospective runtime denominator. | `reports/healthcare/runtime_summary.csv` |
| A7 | All promoted domains emit the canonical runtime release contract witness fields. | `reports/publication/runtime_release_contract_witnesses.csv` |
| A8 | ORIUS is utility-preserving relative to fail-safe references in Battery, AV, and Healthcare. | `reports/publication/utility_preserving_safety_scorecard.csv` |

## Bucket B — Bounded / Qualified Claims

| ID | Claim | Qualification |
|----|-------|---------------|
| B1 | AV closed-loop/planner evidence is real. | It is bounded nuPlan kinematic/planner evidence, not road deployment or full autonomous-driving field closure. |
| B2 | Healthcare runtime evidence is real. | It is retrospective monitoring evidence, not prospective clinical trial or live clinical deployment. |
| B3 | Battery remains the deepest witness. | That does not imply equal real-world maturity across the promoted domains. |
| B4 | Utility-preserving safety is claimed relative to fail-safe references. | It does not mean interventions/fallbacks are already optimized for deployment utility. |

## Bucket C — Explicitly Not Claimed

| ID | Non-Claim | Governing Reason |
|----|-----------|------------------|
| C1 | Industrial is promoted in this submission. | `outside_current_submission_scope_battery_av_healthcare_lane` |
| C2 | Healthcare is live/prospective clinical validation. | Current evidence is retrospective monitoring only. |
| C3 | Navigation is a promoted defended row. | It remains a non-promoted row in the current parity matrix. |
| C4 | Aerospace is a promoted defended row. | It remains a non-promoted row in the current parity matrix. |
| C5 | AV is full autonomous-driving field closure. | The current AV row is explicitly bounded. |
| C6 | T9 is a promoted flagship theorem. | The theorem promotion package promotes T9 as the scoped impossibility of quality-ignorant mandatory release, with ambiguity witnesses and explicit mandatory-release boundaries. |
| C7 | T10 is a promoted flagship theorem. | The theorem promotion package promotes T10 as the two-state boundary-indistinguishability lower bound, not as an unrestricted global minimax theorem. |
