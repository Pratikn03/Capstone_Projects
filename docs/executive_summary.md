# ORIUS — Executive Summary

> Generated from the canonical battery + AV closure artifacts.

## What ORIUS Is

ORIUS identifies OASG as the degraded-observation release hazard and provides a reliability-aware runtime safety layer across Battery, AV, and Healthcare.

ORIUS (Observation–Reality Integrity for Universal Safety) is a runtime safety layer for physical AI systems under degraded observation. It treats the observation–action safety gap as the governing hazard and responds through the Detect → Calibrate → Constrain → Shield → Certify lane.

## Current Submission Scope

- `submission_scope=battery_av_nuplan_three_city_closure`
- `battery` is the reference witness row.
- `av` is the bounded proof-validated row under the longitudinal TTC plus predictive-entry-barrier contract.
- `industrial`, `healthcare`, `navigation`, and `aerospace` are not promoted in this battery+AV submission lane.

## Locked Battery + AV Results

| Domain | Tier | Key Result | Evidence |
|--------|------|------------|----------|
| **Battery (BESS)** | `reference` | Canonical TSVR = 0.0% on 288 canonical runtime rows; 0 OASG cases identified. | 30 locked artifacts; chain valid = False; certificates = 0 |
| **Autonomous Vehicles** | `proof_validated` | Baseline TSVR = 25.3%, ORIUS TSVR = 0.0% on 1,968 canonical runtime rows (15,744 total trace rows); 88 OASG cases identified on the ORIUS AV defended row. | 44 locked artifacts; chain valid = True; certificates = 1,968 |

## What This Submission Does Not Claim

- Industrial and healthcare are intentionally outside the promoted `battery_av_only` submission lane.
- AV remains a bounded longitudinal result; it is not a claim of full autonomous-driving closure.
- Navigation and aerospace remain non-promoted rows.
- Adversarial completeness and production deployment readiness are not claimed from this surface.

## Canonical Artifacts

- `reports/battery_av/overall/release_summary.json`
- `reports/battery_av/overall/publication_closure_override.json`
- `reports/publication/orius_equal_domain_parity_matrix.csv`
- `reports/battery_av/battery/`
- `reports/orius_av/full_corpus/`
