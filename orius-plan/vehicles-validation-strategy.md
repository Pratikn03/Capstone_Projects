# Vehicles Prototype — Validation Strategy

**Status**: Implemented. No theorem upgrades. Battery-only theorems remain locked.

## Unit-Level Tests

| Component | Test file | Coverage |
|-----------|-----------|----------|
| VehiclePlant | `test_vehicle_adapter.py::TestVehiclePlant` | reset, step, speed limit violation |
| ingest_telemetry | `test_vehicle_adapter.py::TestVehicleDomainAdapterIngestTelemetry` | passthrough, NaN zero-order hold |
| compute_oqe | `test_vehicle_adapter.py::TestVehicleDomainAdapterComputeOqe` | w_t in [0,1] |
| build_uncertainty_set | `test_vehicle_adapter.py::TestVehicleDomainAdapterBuildUncertaintySet` | interval bounds |
| repair_action | `test_vehicle_adapter.py::TestVehicleDomainAdapterRepairAction` | clipping, pass-through when safe |
| emit_certificate | `test_vehicle_adapter.py::TestVehicleDomainAdapterEmitCertificate` | certificate structure |

## End-to-End Sanity Checks

| Test | File | Purpose |
|------|------|---------|
| run_vehicle_episode_completes | `test_vehicle_runner_e2e.py` | Full episode runs without error |
| compute_vehicle_metrics | `test_vehicle_runner_e2e.py` | Metrics aggregation |
| dc3s_interventions_in_toy_scenario | `test_vehicle_runner_e2e.py` | Violations stay low under DC3S |

## No Theorem Upgrades

All formal theorems T1–T8 remain battery-only. The vehicles prototype has no formal theorem layer. If a future vehicles theorem set is drafted, it would require separate evidence and a distinct chapter.
