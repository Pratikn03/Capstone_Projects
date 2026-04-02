# ORIUS Universal Kernel Architecture

This document is the thesis-facing architecture spec for ORIUS as a universal
degraded-observation safety framework. It keeps the current repo paths stable
while defining the canonical logical layers that the manuscript and code should
refer to.

## Logical layers

| Logical layer | Purpose | Canonical modules |
|---|---|---|
| `core` | typed contracts, step semantics, reliability-aware uncertainty, risk bounds | `src/orius/universal_theory/`, `src/orius/dc3s/quality.py`, `src/orius/dc3s/domain_adapter.py` |
| `domains` | plant-specific telemetry parsing, uncertainty construction, repair, fallback semantics | `src/orius/adapters/`, `src/orius/universal_framework/*_adapter.py`, `src/orius/dc3s/battery_adapter.py` |
| `bench` | replay schema, metrics, fault model, latency and audit summaries | `src/orius/orius_bench/`, `reports/universal_orius_validation/`, `reports/publication/` |
| `governance` | certificate lifecycle, audit continuity, recovery, claim authority, tracked release surfaces | `src/orius/certos/`, `paper/metrics_manifest.json`, `paper/claim_matrix.csv`, `services/api/routers/research.py` |

## Canonical runtime flow

1. `ingest_telemetry` parses raw telemetry into an observation packet or state view.
2. `compute_oqe` scores reliability and emits fault/degradation flags.
3. `build_uncertainty_set` constructs an observation-consistent state set widened by reliability loss.
4. `tighten_action_set` derives the safe action set under the current constraint surface.
5. `repair_action` projects or otherwise repairs the candidate action into the safe set.
6. `emit_certificate` records the repaired action, uncertainty, reliability, and causal metadata for audit.

## Canonical public surfaces

- `orius.dc3s.domain_adapter.DomainAdapter` is the runtime adapter contract.
- `orius.domain.adapter.DomainAdapter` is retained for legacy compatibility only.
- `orius.universal_theory` exports only domain-neutral contracts, kernel builders, and risk-bound helpers.
- battery-specific theorem helpers must be imported from the battery domain package, not from `orius.universal_theory`.

## Evidence-tier interpretation

- `reference witness`: deepest theorem-to-code-to-artifact closure
- `proof-validated`: bounded defended instantiation under locked replay and runtime checks
- `proof-candidate`: substantial closure progress without full promotion
- `shadow-synthetic`: portability evidence without defended real-data closure
- `experimental`: adapter/runtime surface exists but promotion contract remains open

## Architecture boundary rules

1. Core code must not assume SOC, MWh, charge, discharge, or battery-only fallback semantics.
2. Domain modules may encode battery, vehicle, industrial, healthcare, navigation, or aerospace specifics.
3. Benchmark and governance layers must consume domain-neutral interfaces even when a domain-specific payload carries richer fields.
4. Frontend report data is backend-served from tracked publication artifacts only.
