# ORIUS Claim Ledger — Battery + AV Submission Lane

> Generated from the canonical closure artifacts.

ORIUS identifies OASG as the degraded-observation release hazard and provides a reliability-aware runtime safety layer across Battery, AV, and Healthcare.

## Governing Inputs

- `reports/battery_av/overall/release_summary.json`
- `reports/battery_av/overall/publication_closure_override.json`
- `reports/publication/orius_equal_domain_parity_matrix.csv`
- `submission_scope=battery_av_nuplan_three_city_closure`

## Bucket A — Directly Artifact-Backed

| ID | Claim | Governing Artifact |
|----|-------|-------------------|
| A1 | Battery remains the `reference` witness row in the current submission lane. | `publication_closure_override.json` |
| A2 | Battery canonical TSVR is 0.0% on 288 canonical runtime rows. | `reports/battery_av/battery/runtime_summary.csv`, `release_summary.json` |
| A3 | Battery CertOS chain is valid with 0 certificates. | `reports/battery_av/battery/certos_verification_summary.json` |
| A4 | AV remains the bounded `proof_validated` row. | `publication_closure_override.json` |
| A5 | AV baseline TSVR is 25.3% and ORIUS TSVR is 0.0%. | `reports/orius_av/full_corpus/runtime_summary.csv` |
| A6 | AV runtime rows total = 15,744; canonical ORIUS rows = 1,968. | `release_summary.json` |
| A7 | AV ORIUS OASG cases identified = 88. | `reports/orius_av/full_corpus/oasg_domain_summary.csv`, `release_summary.json` |
| A8 | Only `battery` and `vehicle` are promoted rows under `submission_scope=battery_av_nuplan_three_city_closure`. | `reports/publication/orius_equal_domain_parity_matrix.csv` |

## Bucket B — Bounded / Qualified Claims

| ID | Claim | Qualification |
|----|-------|---------------|
| B1 | AV proof-validation is real. | It is bounded to the current longitudinal TTC plus predictive-entry-barrier contract. |
| B2 | Battery remains the deepest witness. | That does not imply equal maturity across domains outside the current battery+AV lane. |
| B3 | Shift-aware uncertainty is active in both rows. | Coverage guarantees under arbitrary shift are not claimed from these artifacts. |

## Bucket C — Explicitly Not Claimed

| ID | Non-Claim | Governing Reason |
|----|-----------|------------------|
| C1 | Industrial is promoted in this submission. | `outside_current_submission_scope_battery_av_lane` |
| C2 | Healthcare is promoted in this submission. | `outside_current_submission_scope_battery_av_lane` |
| C3 | Navigation is a promoted defended row. | It remains a non-promoted row in the current parity matrix. |
| C4 | Aerospace is a promoted defended row. | It remains a non-promoted row in the current parity matrix. |
| C5 | AV is full autonomous-driving closure. | The current AV row is explicitly bounded. |
