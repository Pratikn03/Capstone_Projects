# Vehicles Extension — Traceability Stub

**Status**: Prototype. Not in paper.tex. Ready for future chapter mapping.

## Chapter-to-Code Mapping (Future)

When/if a vehicle chapter is added to the manuscript:

| Chapter | Content | Code | Artifacts |
|---------|---------|------|-----------|
| ch_vehicles_proto | 1D longitudinal vehicle adapter | `src/orius/vehicles/` | `reports/vehicles_prototype/` |
| — | VehicleDomainAdapter interface | `vehicle_adapter.py` | — |
| — | Vehicle plant dynamics | `plant.py` | — |
| — | Benchmark harness | `vehicle_runner.py` | `vehicle_episode_log.csv` |
| — | Metrics | `compute_vehicle_metrics` | `vehicle_metrics.json` |

## Theorem Mapping (Future)

All formal theorems T1–T8 remain **battery-only**. A future vehicle theorem set would require:
- Vehicle-specific safety predicate φ(x)
- Vehicle-specific dynamics and reachability
- Separate evidence family under `reports/vehicles_prototype/`

No such theorems exist in the current prototype.

## Claim Boundary

- **Locked**: Battery-only claims in paper.tex
- **Prototype**: Vehicles adapter and benchmark — exploratory only
- **Isolation**: `reports/vehicles_prototype/` never overwrites `reports/publication/`
