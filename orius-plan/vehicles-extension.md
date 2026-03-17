## ORIUS Vehicles Extension — Scope and Claims

### 1. Minimal Vehicle Scenario

- **Domain**: Single-lane longitudinal control for a road vehicle (no lane changes, no intersections).
- **State**: 1D kinematics along a lane:
  - position \(x_t\) (m) along the lane centerline
  - speed \(v_t\) (m/s)
  - optional environment scalars (posted speed limit \(v_{\max}(t)\), lead-vehicle position \(x^{\text{lead}}_t\) if present)
- **Action**: Scalar longitudinal command:
  - acceleration setpoint \(a_t\) (m/s\(^2\)) or equivalent throttle/brake command mapped to \(a_t\)
- **Dynamics**: Simple discrete-time integrator with bounded acceleration:
  - \(v_{t+1} = \mathrm{clip}(v_t + a_t \Delta t,\; 0,\; v_{\max}^{\text{phys}})\)
  - \(x_{t+1} = x_t + v_{t+1} \Delta t\)
- **Safety predicates**:
  - speed limit: \(v_t \le v_{\max}(t)\)
  - headway: \(x^{\text{lead}}_t - x_t \ge d_{\min}(v_t)\) when a lead vehicle is present

This is intentionally a **toy longitudinal scenario**, not full autonomous driving. It is chosen to keep the first ORIUS–vehicles adapter bounded while still exercising the full DC3S pipeline (telemetry, reliability, inflation, repair, certificate).

### 2. Telemetry and Fault Model

- **Telemetry vector \(z_t\)**:
  - observed position and speed \((\tilde{x}_t, \tilde{v}_t)\)
  - observed speed limit \(\tilde{v}_{\max}(t)\)
  - optional lead-vehicle estimate \(\tilde{x}^{\text{lead}}_t\)
- **Fault families** (vehicle domain analogue of battery telemetry faults):
  - dropout and delay-jitter in GNSS and speed sensors
  - spikes in measured speed or position
  - stale or frozen lead-vehicle estimates
  - low-rate bias or drift in speed or position
- **OQE features**:
  - inter-arrival time and timestamp consistency
  - finite-difference residuals between predicted and observed position/speed
  - windowed statistics of spikes and bias indicators

The **OQE architecture** and FTIT logic are reused from the battery domain; only the concrete features and thresholds change.

### 3. ORIUS Objects in the Vehicle Domain

- \(z_t\): vehicle telemetry packet (position, speed, limits, optional lead vehicle)
- \(w_t\): scalar reliability score from OQE over vehicle faults
- \(C_t^{\text{RAC}}\): conformal region over \((x_t, v_t)\) (e.g., a box or ellipsoid)
- \(\mathcal{A}_t\): feasible acceleration set given current state and safety predicates
- \(a_t^{\text{safe}}\): repaired acceleration that preserves speed-limit and headway safety under worst-case state in \(C_t^{\text{RAC}}\)

These are direct vehicle-domain instantiations of the chapter-21 abstraction; the **ORIUS principle** remains unchanged.

### 4. Claim Boundaries Relative to the Battery Thesis

- **Locked battery-only claims**:
  - All theorems T1–T8, the DC3S coverage results, and the empirical safety and impact results remain **battery-only** and unchanged.
  - `paper/paper.tex` continues to claim only that the battery implementation is validated on DE and US load profiles under the locked evidence family.
- **Vehicles adapter status**:
  - The vehicles work is a **prototype extension** and **not** part of the locked thesis or paper claims.
  - No universal cross-domain safety guarantees, deployment guarantees, or production readiness are claimed for vehicles.
  - Any vehicles metrics or artifacts will be documented as exploratory evidence only (e.g., in `orius-plan/vehicles-extension.md` and future extension docs), not as new publication-grade results.
- **Scope discipline**:
  - When code is refactored to introduce a `DomainAdapter` abstraction, the battery adapter remains the **reference implementation** and the only adapter tied into locked publication artifacts.
  - Vehicles-specific configuration defaults, metrics, and scripts must not alter existing locked battery artifacts or change results in `reports/publication/` unless explicitly isolated under new filenames.

This document defines the **minimal vehicles scenario** and **claim boundary** that the ORIUS vehicles extension will respect while the battery thesis and paper remain the sole authoritative publication surfaces.

### 5. ORIUS/DC3S Core: Domain-Agnostic vs. Battery-Specific

- **Domain-agnostic infrastructure (reused for vehicles)**:
  - Reliability/OQE: `src/orius/dc3s/quality.py` → `compute_reliability(...)`
  - RAC-Cert inflation and uncertainty sets: `src/orius/dc3s/calibration.py`, `ambiguity.py`, `rac_cert.py`
  - Certificates and audit store: `src/orius/dc3s/certificate.py`
  - FTIT state tracking and fault counters: `src/orius/dc3s/ftit.py`
  - Drift detector: `src/orius/dc3s/drift.py`
  - Coverage/core-bound helpers: `src/orius/dc3s/coverage_theorem.py`
  - Monitoring and health summaries: `src/orius/monitoring/dc3s_health.py`

- **Battery-specific components (to be wrapped behind adapters)**:
  - Battery shield projection and safe-landing logic:
    - `src/orius/dc3s/shield.py` (charge/discharge MWh/MW, SOC envelope, ramp constraints)
    - `src/orius/dc3s/safety_filter_theory.py` (SOC tube interpretation)
  - Temporal theorems and tubes in SOC space:
    - `src/orius/dc3s/temporal_theorems.py`
  - Battery plant and CPSBench track:
    - `src/orius/cpsbench_iot/plant.py`, `runner.py`, `baselines.py`, `scenarios.py`, `metrics.py`
  - Battery-specific metrics and artifacts:
    - SOC violation metrics, MWh-based severities, and battery-only tables/figures in `reports/` and `reports/publication/`

The vehicles extension will treat the first group as **shared ORIUS core** and will introduce a `DomainAdapter` abstraction so that the second group becomes one `BatteryDomainAdapter` implementation alongside a new `VehicleDomainAdapter`.

---

## 6. Implementation Status (Prototype)

- **VehicleDomainAdapter**: `src/orius/vehicles/vehicle_adapter.py` — implements all six DomainAdapter methods
- **VehiclePlant**: `src/orius/vehicles/plant.py` — 1D longitudinal dynamics
- **Vehicle runner**: `src/orius/vehicles/vehicle_runner.py` — CPSBench-like harness
- **Benchmark script**: `scripts/run_vehicle_benchmark.py` — outputs to `reports/vehicles_prototype/`
- **Specification**: `orius-plan/vehicles-adapter-spec.md` — state, actions, constraints, faults

**Traceability**: Vehicles artifacts live under `reports/vehicles_prototype/` only. No changes to `reports/publication/` or locked battery evidence. Paper claims remain battery-only.

### 7. AV Datasets

Best datasets for 1D longitudinal control: NGSIM (US highway), highD (German highway), HEE (Bosch GitHub). Run `make av-datasets` to generate synthetic trajectories. For real data: `python scripts/download_av_datasets.py --source ngsim` (requires Kaggle CLI) or `--source hee`. See `orius-plan/AV_DATASETS.md`.


