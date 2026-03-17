# ORIUS Universal Domain Physical Safety Framework

Domain-agnostic pipeline for physical safety across six domains (thesis Ch 18): Energy, AV, Industrial, Healthcare, Surgical Robotics, Aerospace.

---

## 1. Pipeline Stages

| Stage | Name | Adapter Method | Output |
|-------|------|----------------|--------|
| 1 | Detect | `compute_oqe` | w_t, flags |
| 2 | Calibrate | `build_uncertainty_set` | U_t, meta |
| 3 | Constrain | `tighten_action_set` | A_t |
| 4 | Shield | `repair_action` | safe_action, repair_meta |
| 5 | Certify | `emit_certificate` | certificate |

---

## 2. Domain State Schema

| Domain | State Fields | Unit | Description |
|--------|--------------|------|-------------|
| Energy | soc_mwh, load_mw, renewables_mw | MWh, MW | Battery SOC, grid load, renewables |
| AV | position_m, speed_mps, speed_limit_mps, lead_position_m | m, m/s | Longitudinal kinematics |
| Industrial | temp_c, pressure_mbar, humidity_pct, power_mw | °C, mbar, %, MW | Process variables |
| Healthcare | hr_bpm, spo2_pct, respiratory_rate | bpm, %, /min | Vital signs |
| Surgical Robotics | hr_bpm, spo2_pct, respiratory_rate | bpm, %, /min | Vital signs in OR (alias of Healthcare) |
| Aerospace | altitude_m, airspeed_kt, bank_angle_deg, fuel_remaining_pct | m, kt, deg, % | Flight envelope |

---

## 3. Safety Predicates

| Domain | Safety Predicate | Config Key |
|--------|------------------|------------|
| Energy | SOC ∈ [min_soc, max_soc]; charge/discharge mutual exclusion | min_soc_mwh, max_soc_mwh |
| AV | v ≤ v_max; x_lead - x ≥ d_min(v) | speed_limit_mps, min_headway_m |
| Industrial | temp ∈ [T_min, T_max]; power ∈ [0, P_max] | temp_min_c, power_max_mw |
| Healthcare | hr ∈ [hr_min, hr_max]; SpO2 ≥ spo2_min; rr ∈ [rr_min, rr_max] | hr_min_bpm, spo2_min_pct |
| Surgical Robotics | Same as Healthcare (vital signs in OR) | hr_min_bpm, spo2_min_pct |
| Aerospace | v ∈ [v_min, v_max]; bank ∈ [-max, max]; fuel ≥ fuel_min | v_min_kt, v_max_kt, max_bank_deg, fuel_min_pct |

---

## 4. Fault Taxonomy

| Fault | OQE Indicator | Effect on w_t | Domains |
|-------|----------------|---------------|---------|
| Dropout | NaN in signal | → 0.0 | All |
| Delay/jitter | timestamp gap > cadence | exponential decay | All |
| Spike | \|value - last\| exceeds range | multiply by (1 - β) | All |
| Stale/frozen | unchanged for k steps | FTIT stale counter | All |
| Bias/drift | residual exceeds threshold | Page-Hinkley | All |

---

## 5. Code Usage

```python
from orius.universal_framework import run_universal_step, get_adapter, list_domains

# List domains: energy, av, industrial, healthcare, surgical_robotics, aerospace
print(list_domains())

# Get adapter and run one step
adapter = get_adapter("industrial", {"expected_cadence_s": 3600})
result = run_universal_step(
    domain_adapter=adapter,
    raw_telemetry={"temp_c": 25, "pressure_mbar": 1010, "power_mw": 450, "ts_utc": "2026-01-01T00:00:00Z"},
    history=None,
    candidate_action={"power_setpoint_mw": 480},
    constraints={"power_max_mw": 500},
    quantile=30,
)
# result["certificate"], result["safe_action"], result["reliability_w"], ...
```

---

## 6. File Layout

| File | Purpose |
|------|---------|
| `src/orius/universal_framework/__init__.py` | Package exports |
| `src/orius/universal_framework/domain_registry.py` | Domain registration |
| `src/orius/universal_framework/pipeline.py` | `run_universal_step` |
| `src/orius/universal_framework/industrial_adapter.py` | Industrial domain |
| `src/orius/universal_framework/healthcare_adapter.py` | Healthcare domain |
| `src/orius/universal_framework/aerospace_adapter.py` | Aerospace domain (placeholder) |
| `src/orius/universal_framework/tables.py` | Framework tables (programmatic) |

**Note:** `surgical_robotics` is an alias for `healthcare` (vital signs in OR).

---

## 7. Datasets and Training

| Domain | Dataset Path | Download | Train |
|--------|--------------|----------|-------|
| Energy | `data/raw/opsd/`, `data/raw/us_eia930/`, `data/processed/` | OPSD, EIA-930 | `scripts/train_dataset.py` |
| AV | `data/av/processed/av_trajectories_orius.csv` | `make av-datasets` | Synthetic / NGSIM |
| Industrial | `data/industrial/processed/industrial_orius.csv` | `make industrial-datasets` | Synthetic / CCPP |
| Healthcare | `data/healthcare/processed/healthcare_orius.csv` | `make healthcare-datasets` | Synthetic / BIDMC |
| Surgical Robotics | Same as Healthcare | `make healthcare-datasets` | — |
| Aerospace | — | No dataset yet (placeholder) | — |

See `orius-plan/AV_DATASETS.md`, `orius-plan/INDUSTRIAL_DATASETS.md`, `orius-plan/HEALTHCARE_DATASETS.md` for details.
