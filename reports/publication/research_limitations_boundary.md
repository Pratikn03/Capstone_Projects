# Research Limitations Boundary

## Required Boundary Language

- AV is not full autonomous-driving field closure.
- Healthcare is not live clinical deployment.
- Healthcare is not clinical decision support approval.
- Healthcare is not prospective trial evidence.
- Battery is not yet physical HIL or field deployment.
- Universality means shared runtime contract and adapter discipline, not equal real-world maturity in every domain.

## Evidence Tiers

- Battery: deepest witness row; software HIL/simulator evidence is predeployment evidence, not unrestricted field validation.
- Autonomous Vehicles: all-zip grouped nuPlan replay/surrogate runtime-contract evidence; not full autonomous-driving field closure, CARLA completion, or road deployment.
- Healthcare: retrospective source-holdout/time-forward monitoring evidence; not live prospective clinical validation, clinical decision support approval, or a regulated deployment.

## Freeze Boundary

Freeze status is `freeze_complete_manifest_present`. Do not claim a completed frozen release until `predeployment_release_manifest.json`, `frozen_artifact_hashes.csv`, and `frozen_artifact_hashes.json` exist for the release.

## Universal Claim Boundary

Universality means a shared runtime contract, typed adapter discipline, and governed evidence ladder. It does not mean a single universal controller, a new conditional-coverage theorem, equal real-world domain maturity, or deployment-grade proof across all physical systems.
