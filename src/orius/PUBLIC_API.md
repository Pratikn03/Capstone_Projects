# ORIUS Public API Stability Matrix

This document defines what downstream code may rely on across releases.  It is
the single source of truth for the public surface; the package docstring in
`__init__.py` mirrors a condensed view.

## Tiers

| Tier | Meaning | Versioning |
|------|---------|------------|
| **Stable** | API guaranteed across patch releases.  Breaking change requires a minor-version bump and a deprecation cycle of at least one minor release. | semver |
| **Defended** | Concrete adapter implementations that back paper rows.  API is stable; runtime semantics are governed by the paper's claim boundary. | semver |
| **Experimental** | Subject to change without deprecation.  Do not depend on internal signatures.  May be removed in any minor release. | none |
| **Legacy / Compat** | Preserved for backward compatibility only.  Will be removed in a future major release.  Not part of the defended evidence package. | deprecation |

## Module-by-module

### Stable

| Module | Purpose |
|--------|---------|
| `orius.adapters` | Canonical typed adapter entry points implementing the `DomainInstantiation` protocol. |
| `orius.dc3s.kernel` | Five-stage `execute_universal_step` orchestrator. |
| `orius.dc3s.contracts` | Typed dataclasses for `ObservationPacket`, `ReliabilityAssessment`, `ObservationConsistentStateSet`, `SafeActionSet`, `RepairDecision`, `SafetyCertificate`. |
| `orius.dc3s.calibration` | RAC-Cert inflation and base conformal interval construction. |
| `orius.dc3s.drift` | Page-Hinkley + history-based detectors used by Stage 1. |
| `orius.dc3s.risk_bounds` | T10/T11 risk computations cited by Appendix C. |
| `orius.certos.verification` | Certificate-chain hash verification, CertOS re-verification. |
| `orius.universal_theory.contract_objects` | Audit-bearing contract object schemas referenced by the theorem chain. |

### Defended

| Module | Defended row | Paper section |
|--------|--------------|---------------|
| `orius.dc3s.battery_adapter` | Battery witness | §VIII |
| `orius.av_waymo.runtime` | AV bounded (replay) | §IX |
| `orius.universal_framework.healthcare_adapter` | Healthcare bounded (retrospective) | §X |

### Experimental

| Module | Status |
|--------|--------|
| `orius.forecasting.dl_tft`, `.dl_lstm`, `.dl_nbeats`, `.dl_patchtst`, `.dl_tcn` | Research / diagnostic only.  Only `ml_gbm` feeds a defended row. |
| `orius.multi_agent.*` | Composition layer; not exercised by paper rows. |
| `orius.orius_bench.*` | Benchmark export layer; subject to schema changes. |
| `orius.monitoring.*` | Telemetry/monitoring scaffolding; not part of paper claims. |

### Legacy / Compat

| Module | Note |
|--------|------|
| `orius.legacy.aerospace_adapter` | Extensibility demonstration; not defended.  See `legacy/__init__.py`. |
| `orius.legacy.industrial_adapter` | Extensibility demonstration; not defended. |
| `orius.legacy.navigation_adapter` | Extensibility demonstration; not defended. |
| `orius.universal_framework.{aerospace,industrial,navigation}_adapter` | One-line compat re-exports of the above. |

## Versioning policy (0.x)

ORIUS is currently 0.x (pre-1.0).  Within the 0.x line:

- **Patch releases** (0.1.x): bug fixes, no API change.
- **Minor releases** (0.x.0): may include breaking changes to Stable APIs with at least one minor release of deprecation notice.
- **1.0 cut**: planned after publication of the IEEE manuscript and one independent reproduction of the battery witness row.

## Stability commitments

Within a stable submodule:

- Public function/class names will not be removed or renamed without a deprecation cycle.
- Required arguments will not change.
- Return types will not change in incompatible ways (additions are allowed).
- Certificate schema fields are append-only within a minor release.

Internal helpers (any name starting with `_`) are not part of the public API.
