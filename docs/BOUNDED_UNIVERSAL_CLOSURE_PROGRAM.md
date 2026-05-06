# ORIUS Three-Domain Closure Program

This document defines the active closure target for ORIUS:

> ORIUS provides a reliability-aware runtime safety layer for physical AI under degraded observation, enforcing certificate-backed action release through uncertainty coverage, repair, and fallback.

This is not an equal-theorem-depth claim. It is a code, data, replay, artifact,
and governance closure target for the active repo-tracked program.

## Current truth and closure target

Current tracked evidence:

- `battery`: witness row
- `av`: defended bounded row
- `healthcare`: defended bounded row

Closure target:

- keep Battery as witness
- keep AV bounded to the TTC plus predictive-entry-barrier contract
- keep Healthcare bounded to the promoted MIMIC monitoring contract

## Promotion gate

A row is only treated as promoted when all of the following are true:

- raw-source provenance manifest exists
- processed dataset exists under the canonical contract
- train, calibration, validation, and test splits exist and are non-empty
- universal replay completes under the canonical benchmark schema
- CertOS lifecycle, fallback, and certificate traces exist
- material safety improvement is demonstrated on the governing safety object
- release artifacts are written and referenced by the closure matrix, claim matrix, and release summary

## Canonical runtime and evidence surfaces

- `src/orius/universal_framework/`: active three-domain runtime adapter registry
- `src/orius/dc3s/`: canonical repair, fallback, and certificate logic
- `paper/assets/data/data_manifest.json`: canonical tracked data identity manifest
- `reports/publication/orius_domain_closure_matrix.csv`: canonical closure matrix
- `reports/publication/orius_submission_scorecard.csv`: canonical readiness scorecard
- `reports/battery_av_healthcare/overall/`: canonical promoted-lane bundle

## Per-domain notes

### Battery

- deepest theorem-to-artifact closure
- remains the witness row

### AV

- canonical raw closure target: Waymo Open Motion
- bounded to the TTC plus predictive-entry-barrier contract

### Healthcare

- canonical primary row: MIMIC-III bounded monitoring surface
- BIDMC remains supplemental only

## Active scripts

- `scripts/refresh_real_data_manifests.py`
- `scripts/verify_real_data_preflight.py`
- `scripts/run_universal_training_audit.py`
- `scripts/run_universal_orius_validation.py`
- `scripts/build_domain_closure_matrix.py`
- `scripts/build_orius_monograph_assets.py`

## Recommended execution order

1. Refresh provenance manifests and inspect the status report.
2. Verify real-data preflight in the target environment.
3. Run universal training audit.
4. Run universal ORIUS validation.
5. Rebuild the domain closure matrix and publication surfaces.
6. Rebuild the monograph and review dossier.

## Non-claims

- This program does not imply theorem-depth symmetry across all physical-AI domains.
- This program does not claim raw provider data should be committed into git.
- This program does not allow manuscript or publication claims to widen ahead of tracked artifacts.
