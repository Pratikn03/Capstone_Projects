# ORIUS Claim Ledger — Battery + AV Submission Lane

> Generated from the canonical closure artifacts.

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

- `reports/battery_av/overall/release_summary.json`
- `reports/battery_av/overall/publication_closure_override.json`
- `reports/publication/orius_equal_domain_parity_matrix.csv`
- `reports/publication/theorem_promotion_scorecard.json`
- `reports/publication/theorem_promotion_gates.csv`
- `submission_scope=battery_av_only`

## Bucket A — Directly Artifact-Backed

| ID | Claim | Governing Artifact |
|----|-------|-------------------|
| A1 | Battery remains the `reference` witness row in the current submission lane. | `publication_closure_override.json` |
| A2 | Battery canonical TSVR is 0.0% on 288 canonical runtime rows. | `reports/battery_av/battery/runtime_summary.csv`, `release_summary.json` |
| A3 | Battery CertOS chain is valid with 0 certificates. | `reports/battery_av/battery/certos_verification_summary.json` |
| A4 | AV remains the bounded `runtime_contract_closed` row. | `publication_closure_override.json` |
| A5 | AV baseline TSVR is 51.4% and ORIUS TSVR is 0.0%. | `reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_summary.csv` |
| A6 | AV runtime rows total = 35,752; canonical ORIUS rows = 4,469. | `release_summary.json` |
| A7 | AV ORIUS OASG cases identified = 93. | `reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/oasg_domain_summary.csv`, `release_summary.json` |
| A8 | Only `battery` and `vehicle` are promoted rows under `submission_scope=battery_av_only`. | `reports/publication/orius_equal_domain_parity_matrix.csv` |

## Bucket B — Bounded / Qualified Claims

| ID | Claim | Qualification |
|----|-------|---------------|
| B1 | AV runtime-contract closure is real. | It is bounded to the current brake-hold release contract. |
| B2 | Battery remains the deepest witness. | That does not imply equal maturity across domains outside the current battery+AV lane. |
| B3 | Shift-aware uncertainty is active in both rows. | Coverage guarantees under arbitrary shift are not claimed from these artifacts. |

## Bucket C — Explicitly Not Claimed

| ID | Non-Claim | Governing Reason |
|----|-----------|------------------|
| C1 | Industrial is promoted in this submission. | `outside_current_submission_scope_battery_av_lane` |
| C2 | Healthcare is promoted in this submission. | `outside_current_submission_scope_battery_av_lane` |
| C3 | Navigation is a promoted defended row. | It remains a non-promoted row in the current parity matrix. |
| C4 | Aerospace is a promoted defended row. | It remains a non-promoted row in the current parity matrix. |
| C5 | AV is full autonomous-driving field closure. | The current AV row is explicitly bounded. |
| C6 | T9 is a promoted flagship theorem. | The generated promotion package keeps T9 `draft_non_defended` until Battery, AV, and Healthcare all discharge the witness constant, mixing bridge, mechanized proof, and remaining gates. |
| C7 | T10 is a promoted flagship theorem. | The generated promotion package keeps T10 `draft_non_defended` until Battery, AV, and Healthcare all discharge the boundary-mass/TV bridge, mechanized proof, and remaining gates. |
