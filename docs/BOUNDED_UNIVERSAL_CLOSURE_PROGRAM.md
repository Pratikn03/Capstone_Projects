# ORIUS Bounded-Universal Closure Program

This document defines the active closure target for ORIUS:

> ORIUS is a universal runtime safety layer for Physical AI under degraded observation, with six defended domain instantiations under bounded domain contracts.

This is not an equal-theorem-depth claim. It is a code, data, replay, artifact,
and governance closure target.

## Current truth vs closure target

Current tracked evidence:

- `battery`: witness row
- `av`: defended bounded row
- `industrial`: defended bounded row
- `healthcare`: defended bounded row
- `navigation`: blocked by missing defended real-data replay closure
- `aerospace`: blocked by missing defended multi-flight runtime replay closure

Closure target:

- `battery`: witness row
- `av`, `industrial`, `healthcare`, `navigation`, `aerospace`: defended bounded rows

## Defended-row promotion gate

A domain is only treated as a defended bounded row when all of the following are true:

- raw-source provenance manifest exists
- processed dataset exists under the canonical contract
- train, calibration, validation, and test splits exist and are non-empty
- model bundle exists
- uncertainty and backtest artifacts exist
- universal replay completes under the canonical benchmark schema
- CertOS lifecycle, fallback, and certificate traces exist
- material safety improvement is demonstrated on the governing safety object
- release artifacts are written and referenced by the parity matrix, claim matrix, and release manifest

## Canonical runtime and evidence surfaces

- `src/orius/dc3s/domain_adapter.py`: only canonical runtime adapter contract
- `src/orius/orius_bench/adapter.py`: benchmark evaluation hooks
- `src/orius/certos/runtime.py`: domain-neutral governance layer
- `paper/assets/data/data_manifest.json`: canonical tracked data identity manifest
- backend `/research/*` endpoints: only canonical dashboard/report truth path

Canonical benchmark fields:

- `true_constraint_violated`
- `observed_constraint_satisfied`
- `true_margin`
- `observed_margin`
- `intervened`
- `fallback_used`
- `certificate_valid`
- `latency_us`
- `domain_metrics`

## Per-domain closure notes

### Battery

- deepest theorem-to-artifact closure
- keep as witness row
- refresh provenance and governed artifacts only

### AV

- canonical raw closure target: Waymo Open Motion
- companion robustness surfaces: Argoverse 2 motion and sensor
- current repo may still contain HEE legacy compatibility data; that does not satisfy the canonical closure target by itself

### Industrial

- canonical primary row: CCPP
- companion raw evidence: ZeMA hydraulic
- defended row remains bounded to the current plant family and replay protocol

### Healthcare

- canonical primary row: BIDMC / PhysioNet CSV corpus
- defended row remains bounded to the current monitoring and intervention contract

### Navigation

- canonical primary row: KITTI Odometry
- blocked until the real-data processed row, training surfaces, replay surfaces, and governed artifacts are all present
- synthetic navigation traces do not satisfy defended-row closure

### Aerospace

- trainable companion surface: NASA C-MAPSS
- canonical defended runtime surface: provider-approved multi-flight telemetry
- C-MAPSS alone does not close the defended aerospace row

## Active scripts

- `scripts/refresh_real_data_manifests.py`
- `scripts/verify_real_data_preflight.py`
- `scripts/run_universal_training_audit.py`
- `scripts/run_universal_orius_validation.py`
- `scripts/build_domain_closure_matrix.py`

Live blocker/status artifact:

- `reports/real_data_contract_status.json`

## Recommended execution order

1. Refresh provenance manifests and inspect blocker report.
2. Verify raw-data preflight in the target closure environment.
3. Rebuild missing processed rows and split artifacts per blocked domain.
4. Run universal training audit.
5. Run universal ORIUS validation.
6. Rebuild the domain closure matrix and publication surfaces.
7. Rebuild the monograph and final release package.

## Non-claims

- This program does not imply equal-domain theorem-depth parity.
- This program does not claim raw provider data should be committed into git.
- This program does not allow manuscript or publication claims to widen ahead of tracked artifacts.
