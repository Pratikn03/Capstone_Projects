# VehicleDomainAdapter Specification

**Status**: Prototype extension. Not part of locked battery thesis claims.

## 1. State Definition

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `position_m` | float | m | Position along lane centerline |
| `speed_mps` | float | m/s | Longitudinal speed |
| `speed_limit_mps` | float | m/s | Posted speed limit |
| `lead_position_m` | float \| None | m | Lead vehicle position (if present) |
| `ts_utc` | str | ISO8601 | Timestamp |

## 2. Action Space

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `acceleration_mps2` | float | m/s² | Longitudinal acceleration setpoint |

## 3. Constraints

| Constraint | Formula | Config key |
|------------|---------|------------|
| Speed limit | v ≤ v_max(t) | `speed_limit_mps` |
| Headway | x_lead - x ≥ d_min(v) | `min_headway_m`, `headway_time_s` |
| Acceleration bounds | a ∈ [a_min, a_max] | `accel_min_mps2`, `accel_max_mps2` |
| Physical speed cap | v ≤ v_phys_max | `speed_phys_max_mps` |

## 4. Fault Taxonomy

| Fault | OQE indicator | Effect on w_t |
|-------|---------------|---------------|
| Dropout | NaN in position/speed | → 0.0 |
| Delay/jitter | timestamp gap > cadence | exponential decay |
| Spike | \|value - last\| exceeds range | multiply by (1 - beta) |
| Stale/frozen | unchanged for k steps | FTIT stale counter |
| Bias/drift | residual exceeds threshold | Page-Hinkley |

## 5. Method Semantics

- **ingest_telemetry**: Parse raw packet → {position_m, speed_mps, speed_limit_mps, lead_position_m?, ts_utc}; zero-order hold for NaN.
- **compute_oqe**: Reuse `compute_reliability` with vehicle event keys (load_mw→speed_mps, etc.) or vehicle-specific feature extraction.
- **build_uncertainty_set**: Conformal box over (position_m, speed_mps) inflated by w_t; interval representation.
- **tighten_action_set**: Feasible acceleration set A_t from speed-limit and headway constraints given uncertainty.
- **repair_action**: Clip acceleration to A_t; L2 projection onto safe set.
- **emit_certificate**: Same structure as battery; action fields = {acceleration_mps2}.
