# ORIUS Battery Framework — Phase 4: Core APIs and Abstractions

**Status**: All runtime objects defined. All are implemented in the repo. Specs below match actual code.

---

## 1. Runtime Object Reference

All 13 canonical runtime objects from the extracted implementation plan §4.2.

### 1.1 `z_t` — Clean Telemetry

| Field | Type | Description |
|-------|------|-------------|
| `device_id` | `str` | Battery device identifier |
| `zone_id` | `str` | Grid zone / balancing authority |
| `timestamp` | `str` (ISO 8601) | Measurement timestamp |
| `value` | `float` | Measured value (load MW, SOC MWh, etc.) |
| `target` | `str` | Signal name (`load_mw`, `solar_mw`, `wind_mw`) |

**Code**: `src/orius/streaming/schemas.py` → `TelemetryEvent`

### 1.2 `tilde_z_t` — Degraded Telemetry

Same schema as `z_t` but with fault injection applied:
- dropped values → `NaN`
- stale values → repeated prior reading
- delayed values → delayed timestamp
- out-of-order events → timestamp reordering
- spike values → outlier injection

**Code**: `src/orius/cpsbench_iot/scenarios.py` → fault injection functions

### 1.3 `o_t` — Observed State

| Field | Description |
|-------|-------------|
| `observed_soc` | SOC as seen by the controller (from degraded telemetry) |
| `timestamp` | When observation was made |
| `zone_id`, `device_id` | Identity |

**Code**: `src/orius/dc3s/state.py` → `DC3SStateStore` rows (field `last_yhat`)

### 1.4 `x_t` — True Physical State

| Field | Description |
|-------|-------------|
| `true_soc` | Real battery state of charge (ground truth, not observed) |
| `timestamp` | Simulation clock time |

**Code**: `src/orius/cpsbench_iot/plant.py` → `true_soc` attribute

### 1.5 `w_t` — Observation Quality Score

| Property | Value |
|----------|-------|
| Range | `[min_w, 1.0]` where `min_w = 0.05` |
| Formula | `w_t = max(w_min, p_miss · p_delay · p_ooo · p_spike)` |
| Config | `dc3s.reliability.min_w = 0.05` |

**Code**: `src/orius/dc3s/quality.py` → `compute_reliability(event, last_event, cfg)`

```python
# Returns a float in [0.05, 1.0]
w_t = compute_reliability(event=telemetry_event, last_event=prev_event, cfg=dc3s_cfg)
```

### 1.6 `d_t` — Drift Evidence / Flag

| Property | Value |
|----------|-------|
| Type | `bool` (drift detected) + float (Page-Hinkley statistic) |
| Detector | Page-Hinkley (online, non-stationary) |
| Config | `dc3s.drift.ph_delta = 0.01`, `ph_lambda = 5.0` |

**Code**: `src/orius/dc3s/drift.py` → `PageHinkleyDetector`

```python
detector = PageHinkleyDetector(delta=0.01, lambda_=5.0, warmup_steps=48)
d_t = detector.update(residual)  # returns True if drift detected
```

### 1.7 `s_t` — Sensitivity / Staleness Signal

| Property | Value |
|----------|-------|
| Type | `float` ∈ [0, 1] |
| Meaning | How sensitive the dispatch decision is to forecast uncertainty |
| Config | `dc3s.rac_cert.k_sensitivity = 0.4` |

**Code**: `src/orius/dc3s/rac_cert.py` → `compute_dispatch_sensitivity()`

### 1.8 `C_t(α)` — Base Conformal Interval

| Property | Value |
|----------|-------|
| Type | `(lower: float, upper: float)` tuple in MW |
| Alpha | Miscoverage rate, `α = 0.10` by default |
| Method | Split conformal / CQR |

**Code**: `src/orius/forecasting/uncertainty/cqr.py` → `CQRCalibrator`

```python
lower, upper = cqr_calibrator.predict(X_test, alpha=0.10)
```

### 1.9 `C_t^RAC(α)` — Reliability-Adaptive Conformal Interval

| Property | Value |
|----------|-------|
| Formula | `C_t^RAC = C_t(α) · [1 + κ_r(1−w_t) + κ_d·d_t + κ_s·s_t]` |
| Config | `κ_r = k_quality = 0.2`, `κ_d = k_drift = 0.0`, `κ_s = k_sensitivity = 0.4` |
| Max multiplier | `max_q_multiplier = 3.0` |

**Code**: `src/orius/dc3s/rac_cert.py` → `RACCertModel.predict()` + `compute_q_multiplier()`

```python
rac_model = RACCertModel(cfg=dc3s_cfg)
rac_model.fit(calibration_residuals)
q_mult = rac_model.compute_q_multiplier(w_t=w_t, d_t=d_t, s_t=s_t)
lower_rac = lower * q_mult
upper_rac = upper * q_mult
```

### 1.10 `U_t(α)` — Uncertainty Set

| Property | Value |
|----------|-------|
| Type | `(lower_bound, upper_bound)` box in state space |
| Built from | RAC-inflated intervals projected onto SOC space |

**Code**: `src/orius/dc3s/calibration.py` → `build_uncertainty_set()`

```python
uncertainty_set = build_uncertainty_set(
    lower=lower_rac, upper=upper_rac,
    current_soc=observed_soc, cfg=dc3s_cfg
)
```

### 1.11 `a_t^*` — Optimizer Candidate Action

| Property | Value |
|----------|-------|
| Type | `float` in MW (positive = charge, negative = discharge) |
| Constraints | Power bounds, ramp bounds, predicted SOC bounds |
| Ignores | Observation reliability (that is the point — it needs shielding) |

**Code**: `src/orius/optimizer/lp_dispatch.py` → `LPDispatcher.solve()`

```python
dispatcher = LPDispatcher(cfg=optimization_cfg)
a_star = dispatcher.solve(
    forecast=load_forecast,
    soc=observed_soc,
    constraints=battery_constraints,
)
```

### 1.12 `a_t^safe` — Repaired Safe Action

| Property | Value |
|----------|-------|
| Formula | `a_safe = argmin_{a ∈ A_t} ‖a − a_star‖₂` |
| Mode | L2 projection (configured by `dc3s.shield.mode = projection`) |
| Fallback | CVaR robust dispatch if projection fails |

**Code**: `src/orius/dc3s/shield.py` → `repair_action(a_star, state, uncertainty_set, constraints, cfg)`

```python
a_safe, intervened, reason = repair_action(
    a_star=a_star,
    state=dc3s_state,
    uncertainty_set=uncertainty_set,
    constraints=battery_constraints,
    cfg=dc3s_cfg,
)
```

### 1.13 `A_t` — Tightened Safe Action Set

| Property | Value |
|----------|-------|
| Type | Feasible region in MW space |
| Computed from | `U_t(α)` + BMS hard limits + SOC buffer |
| SOC buffer | `reserve_soc_pct_drift = 0.08` (8% in drift regime) |

**Code**: `src/orius/dc3s/safety_filter_theory.py` → `tightened_soc_bounds()` + `src/orius/safety/bms.py`

---

## 2. Certificate Object Schema

From `src/orius/dc3s/certificate.py` → `make_certificate()`:

```python
certificate = {
    # Identity
    "command_id": str,           # UUID for this dispatch step
    "certificate_id": str,       # same as command_id
    "created_at": str,           # ISO 8601 UTC timestamp
    "device_id": str,
    "zone_id": str,
    "controller": str,           # "dc3s_wrapped" | "dc3s_ftit" | etc.

    # Actions
    "proposed_action": dict,     # a_t^*
    "safe_action": dict,         # a_t^safe

    # Safety state
    "uncertainty": dict,         # U_t(α) bounds
    "reliability": dict,         # w_t components
    "drift": dict,               # d_t state

    # Intervention
    "intervened": bool,          # True if a_safe ≠ a_star
    "intervention_reason": str,  # "soc_bound" | "power_bound" | "ramp_bound"
    "reliability_w": float,      # w_t scalar
    "drift_flag": bool,          # d_t
    "inflation": float,          # RAC multiplier applied

    # Guarantee checks
    "guarantee_checks_passed": bool,
    "guarantee_fail_reasons": list[str],

    # Physical verification
    "true_soc_violation_after_apply": bool | None,  # set in CPSBench only
    "gamma_mw": float,           # effective dispatch command
    "e_t_mwh": float,            # current energy level
    "soc_tube_lower_mwh": float, # tightened SOC lower bound
    "soc_tube_upper_mwh": float, # tightened SOC upper bound

    # Provenance
    "model_hash": str,           # SHA256 of model artifacts
    "config_hash": str,          # SHA256 of dc3s.yaml
    "prev_hash": str | None,     # previous certificate hash (chain)
    "assumptions_version": str,  # "dc3s-assumptions-v1"

    # Content-addressed hash
    "certificate_hash": str,     # SHA256 of canonical JSON payload
}
```

Storage: `data/audit/dc3s_audit.duckdb` → `dispatch_certificates` table.

---

## 3. DC3S State Object Schema

From `src/orius/dc3s/state.py` → `DC3SStateStore`:

```python
state = {
    "last_timestamp": str | None,
    "last_yhat": float | None,       # observed SOC (o_t)
    "last_y_true": float | None,     # true SOC if available (x_t)
    "drift_state": dict,             # PageHinkley internal state
    "adaptive_state": dict,          # RAC-Cert adaptive state
    "last_prev_hash": str | None,    # certificate chain
    "last_inflation": float | None,  # last applied RAC multiplier
    "last_event": dict | None,       # last telemetry event
    "last_action": dict | None,      # last dispatch action
}
```

Key: `f"{zone_id}:{device_id}:{target}"` — one state row per (zone, device, target) tuple.

---

## 4. Theorem-to-Code Mapping

### Theorem 1 — Battery OASG Existence
**Statement**: There exists a degraded-telemetry battery episode where an action appears safe on observed state but violates true physical SOC limits.

**Code verification**:
```python
# src/orius/cpsbench_iot/scenarios.py
# Fault injection creates: tilde_z_t where observed_soc != true_soc
# src/orius/cpsbench_iot/runner.py
# CPSBench measures true_soc_violation_after_apply for det-LP baseline
# Result: 3.9% TSVR for deterministic_lp under nominal scenario
```

**Locked evidence**: `reports/publication/dc3s_main_table_ci.csv` row `nominal,deterministic_lp`, `violation_rate_mean = 0.039286`

### Theorem 2 — One-Step Safety Preservation
**Statement**: If `x_t ∈ U_t(α)` and `a_safe ∈ A_t`, then `x_{t+1}` satisfies the SOC safety constraint.

**Code verification**:
```python
# src/orius/dc3s/guarantee_checks.py
result = evaluate_guarantee_checks(
    soc=observed_soc,
    action=a_safe,
    uncertainty_set=U_t,
    constraints=battery_constraints,
)
assert result.guarantee_checks_passed
next_soc = next_soc(current_soc=true_soc, action=a_safe, battery_params=params)
assert soc_min <= next_soc <= soc_max
```

**Locked evidence**: `dc3s_main_table_ci.csv` — `dc3s_wrapped` controller has `violation_rate_mean = 0.0` across all scenarios.

### Theorem 3 — Core Safety Bound
**Statement**: `E[V] ≤ α(1 − w̄)T`

Where:
- `V` = number of true-state violations
- `α` = miscoverage rate (`dc3s.alpha0 = 0.10`)
- `w̄` = mean observation quality over episode
- `T` = episode length

**Code verification**:
```python
# src/orius/dc3s/coverage_theorem.py
# Verifies marginal coverage at each step
# Expected: if w_t = 1 everywhere → 0 violations guaranteed
# If w_t = 0 → bound relaxes to alpha * T
```

**Locked evidence**: Under DC3S with real observation quality scores, 0% TSVR achieved.

### Theorem 4 — No Free Safety
**Statement**: Any quality-ignorant controller fails under some admissible fault sequence.

**Code verification**:
```python
# src/orius/cpsbench_iot/baselines.py
# deterministic_lp: quality-ignorant (no w_t, no RAC)
# robust_fixed_interval: quality-ignorant (fixed uncertainty, no w_t)
# src/orius/cpsbench_iot/scenarios.py
# drift_combo scenario: admissible fault sequence
# Result: deterministic_lp has violation_rate_mean = 0.044 under drift_combo
```

**Locked evidence**: `dc3s_main_table_ci.csv` rows `drift_combo,deterministic_lp` and `drift_combo,robust_fixed_interval` both show `violation_rate_mean > 0`.

---

## 5. Assumption Register (A1–A8)

### A1 — Bounded Model Error
**Statement**: Forecast model error is bounded: `|ŷ_t − y_t| ≤ ε_model` with high probability.

**Enforcement**:
- `dc3s.alpha0 = 0.10` sets the coverage level (1 − α = 90% coverage)
- CQR calibration in `forecasting/uncertainty/cqr.py` guarantees `E[coverage] ≥ 1 − α`

**Config keys**: `dc3s.alpha0`, `dc3s.alpha_min`

**Violation**: If model error exceeds the conformal bound, T2 cannot guarantee safety → triggers recalibration via `monitoring/retraining.py`

---

### A2 — Bounded Telemetry Error
**Statement**: Telemetry error is bounded: degraded observation satisfies `|tilde_z_t − z_t| ≤ ε_obs`.

**Enforcement**:
- `dc3s.reliability.min_w = 0.05` — floor prevents `w_t → 0` (infinite inflation)
- `dc3s.infl_max = 2.0` — caps inflation multiplier so `U_t` stays bounded

**Config keys**: `dc3s.reliability.min_w`, `dc3s.infl_max`, `dc3s.rac_cert.max_q_multiplier = 3.0`

---

### A3 — Feasible Safe Repair Exists
**Statement**: The tightened safe action set `A_t` is always non-empty; a safe repair always exists.

**Enforcement**:
- `dc3s.shield.mode = projection` — L2 projection onto feasible set
- `src/orius/dc3s/shield.py` → `repair_action()` always returns a feasible point by projecting onto the hard constraint boundary
- `src/orius/safety/bms.py` → BMS hard walls are the fallback if tightened set fails

**Violation trigger**: If even the BMS hard limits are infeasible (SOC already out of range), an emergency safe-landing action is returned.

---

### A4 — Known / Identified Dynamics
**Statement**: Battery dynamics are known: `SOC_{t+1} = f(SOC_t, a_t)` as in §5.1.

**Enforcement**:
- `src/orius/cpsbench_iot/plant.py` — physics truth model with known `eta_c`, `eta_d`, `E_max`, `dt`
- `src/orius/optimizer/lp_dispatch.py` — dispatch uses same dynamics model
- `src/orius/dc3s/guarantee_checks.py` → `next_soc()` — one-step SOC prediction

**Violation**: If real battery drifts from the identified model (aging) → covered by A1 (uncertainty grows) and late extension ch26.

---

### A5 — Monotone Bounded Uncertainty Inflation
**Statement**: Inflation `g(w_t, d_t, s_t)` is monotone non-decreasing as quality degrades and bounded above.

**Enforcement**:
- `dc3s.infl_max = 2.0` — global cap
- `dc3s.rac_cert.max_q_multiplier = 3.0` — RAC-Cert cap
- `dc3s/ambiguity.py` `widen_bounds()` — linear inflation formula (monotone by construction)
- `dc3s/rac_cert.py` `compute_q_multiplier()` — clamps to `[1.0, max_q_multiplier]`

---

### A6 — Bounded Detector Lag
**Statement**: The drift detector has bounded detection lag: `τ_detect ≤ τ_max`.

**Enforcement**:
- `dc3s.drift.warmup_steps = 48` — detector warms up on first 48 steps before flagging
- `dc3s.drift.ph_lambda = 5.0` — Page-Hinkley threshold (lower → faster detection, higher → fewer false positives)
- `dc3s.drift.cooldown_steps = 24` — prevents repeated re-triggers

**Config keys**: `dc3s.drift.ph_delta`, `dc3s.drift.ph_lambda`, `dc3s.drift.warmup_steps`

---

### A7 — Causal Certificate Update Rule
**Statement**: Each certificate is computed from information available at time `t` only (no future lookahead).

**Enforcement**:
- `dc3s/certificate.py` `make_certificate()` — called per-step, forward-only
- `prev_hash` chain links each certificate to the previous (tamper-evident)
- No batching or retrospective modification

---

### A8 — Admissible Fallback Policy
**Statement**: A safe fallback policy exists that keeps the system within the safety envelope during observation loss.

**Enforcement**:
- `dc3s.shield.mode = projection` — projection fallback
- `dc3s.shield.cvar.beta = 0.90` — CVaR robust fallback
- `dc3s.shield.reserve_soc_pct_drift = 0.08` — 8% SOC reserve in drift regime
- `src/orius/safety/bms.py` → BMS hard stop as final fallback
- ch29 extends this to graceful degradation / safe landing policy

---

## 6. DC3S Pipeline Entry Point

Full one-step pipeline in `src/orius/dc3s/__init__.py`:

```python
from orius.dc3s import run_dc3s_step

# One step of the full DC3S pipeline
result = run_dc3s_step(
    event=telemetry_event,           # TelemetryEvent (z_t or tilde_z_t)
    state=dc3s_state_store,          # DC3SStateStore
    candidate_action=a_star,         # float MW from optimizer
    forecast_intervals=(lower, upper), # base CQR intervals
    cfg=dc3s_cfg,                    # loaded from configs/dc3s.yaml
    battery_constraints=constraints, # E_max, P_max, R_max, SOC bounds
    model_hash=model_hash,           # for certificate provenance
    config_hash=config_hash,         # for certificate provenance
)

# result fields:
# result.w_t               → float: observation quality score
# result.d_t               → bool: drift detected
# result.s_t               → float: dispatch sensitivity
# result.rac_interval       → (lower_rac, upper_rac)
# result.uncertainty_set    → U_t bounds
# result.a_safe             → float MW: repaired action
# result.intervened         → bool
# result.certificate        → dict: full certificate object
# result.guarantee_passed   → bool
```

---

## 7. Versioning and Extensibility

### Current version
- `assumptions_version = "dc3s-assumptions-v1"` (set in `dc3s.yaml` and written to every certificate)
- This version string is the immutable anchor for all A1–A8 in the current battery system

### Upgrade protocol
1. Bump `assumptions_version` in `dc3s.yaml`
2. Document what changed (which assumption was relaxed or tightened)
3. Re-run locked benchmark to verify new assumption set does not break T1–T4
4. Update `reproducibility_lock.json` with new run_id
5. Update downstream paper tables

### Extension hooks for non-battery domains
Each DC3S stage is separately callable:
- `quality.compute_reliability()` — domain-agnostic; override factor computation for non-battery signals
- `ambiguity.widen_bounds()` — domain-agnostic linear inflation
- `shield.repair_action()` — domain-specific only in constraint definition (SOC bounds → swap for domain constraints)
- `plant.py` — domain-specific physics; replace with domain simulator

---

*Next: see `05-battery-adapter-design.md` for the full battery domain adapter interface.*
