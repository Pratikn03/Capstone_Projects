# ORIUS — Executive Summary

> Generated from the canonical battery + AV closure artifacts.

## What ORIUS Is

ORIUS provides a reliability-aware runtime safety layer for physical AI under degraded observation, enforcing certificate-backed action release through uncertainty coverage, repair, and fallback.

ORIUS (Observation–Reality Integrity for Universal Safety) treats the observation–action safety gap as the governing hazard and responds through the Detect → Calibrate → Constrain → Shield → Certify lane.

## Current Submission Scope

- `submission_scope=battery_av_only`
- `battery` is the reference witness row.
- `av` is the bounded runtime-contract row under the narrowed brake-hold release contract.
- `industrial`, `healthcare`, `navigation`, and `aerospace` are not promoted in this battery+AV submission lane.

## Locked Battery + AV Results

| Domain | Tier | Key Result | Evidence |
|--------|------|------------|----------|
| **Battery (BESS)** | `reference` | Canonical TSVR = 0.0% on 288 canonical runtime rows; 0 OASG cases identified. | 30 locked artifacts; chain valid = False; certificates = 0 |
| **Autonomous Vehicles** | `runtime_contract_closed` | Baseline TSVR = 51.4%, ORIUS TSVR = 0.0% on 4,469 canonical runtime rows (35,752 total trace rows); 93 OASG cases identified on the ORIUS AV defended row. | 44 locked artifacts; chain valid = True; certificates = 4,469 |

## What This Submission Does Not Claim

- Industrial and healthcare are intentionally outside the promoted `battery_av_only` submission lane.
- AV remains a bounded longitudinal result; it is not a claim of full autonomous-driving field closure.
- Navigation and aerospace remain non-promoted rows.
- Adversarial completeness and production deployment readiness are not claimed from this surface.

## Canonical Artifacts

- `reports/battery_av/overall/release_summary.json`
- `reports/battery_av/overall/publication_closure_override.json`
- `reports/publication/orius_equal_domain_parity_matrix.csv`
- `reports/battery_av/battery/`
- `reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/`
