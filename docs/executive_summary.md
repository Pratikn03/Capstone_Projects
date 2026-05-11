# ORIUS — Executive Summary

> Generated from the canonical three-domain runtime and utility-preserving safety artifacts.

## What ORIUS Is

ORIUS provides a reliability-aware runtime safety layer for physical AI under degraded observation, enforcing certificate-backed action release through uncertainty coverage, repair, and fallback.

ORIUS (Observation–Reality Integrity for Universal Safety) treats the observation–action safety gap as the governing hazard and responds through the Detect → Calibrate → Constrain → Shield → Certify lane.

## Current Submission Scope

- `submission_scope=battery_av_healthcare`
- `battery` is the reference witness row.
- `av` is the bounded nuPlan runtime/closed-loop planner row.
- `healthcare` is the bounded retrospective monitoring row.
- `industrial`, `navigation`, and `aerospace` are not promoted defended rows in this package.

## Claim-Governing Runtime Results

| Domain | Tier | Key Result | Evidence |
|--------|------|------------|----------|
| **Battery (BESS)** | `reference` | Baseline TSVR = 0.83%, ORIUS TSVR = 0.0% on the locked battery witness runtime. | `reports/publication/three_domain_ml_benchmark.csv` |
| **Autonomous Vehicles** | `runtime_contract_closed` | Baseline TSVR = 28.925%, ORIUS TSVR = 0.0163% on the promoted nuPlan runtime denominator. | `reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_summary.csv` |
| **Healthcare Monitoring** | `runtime_contract_closed` | Baseline TSVR = 19.4489%, ORIUS TSVR = 0.0% on the promoted retrospective MIMIC runtime denominator. | `reports/healthcare/runtime_summary.csv` |

## Utility-Preserving Safety

Safety alone would be a weak result if ORIUS merely collapsed to degenerate
fallback. The current claim-facing utility scorecard therefore compares ORIUS
against fail-safe references:

| Domain | Fail-safe reference | Utility-preserving result |
|--------|---------------------|---------------------------|
| **Battery** | immediate shutdown | optimized graceful degradation keeps zero TSVR while preserving 10.1 MWh useful work. |
| **AV** | always-brake | no excess TSVR over always-brake with 3.99x useful work and lower fallback/intervention. |
| **Healthcare** | always-alert | zero TSVR with 142,767 useful monitoring units and lower fallback/intervention. |

Governing artifact: `reports/publication/utility_preserving_safety_scorecard.csv`.

## What This Submission Does Not Claim

- AV remains a bounded longitudinal result; it is not a claim of full autonomous-driving field closure.
- Healthcare remains retrospective monitoring evidence; it is not a prospective clinical trial or live clinical deployment.
- Battery remains predeployment/simulator-HIL evidence; it is not physical field certification.
- Navigation and aerospace remain non-promoted rows.
- Adversarial completeness and production deployment readiness are not claimed from this surface.

## Canonical Artifacts

- `reports/publication/three_domain_ml_benchmark.csv`
- `reports/publication/utility_preserving_safety_scorecard.csv`
- `reports/publication/runtime_release_contract_witnesses.csv`
- `reports/battery_av/battery/`
- `reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/`
- `reports/healthcare/runtime_summary.csv`
