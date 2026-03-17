"""Framework tables for ORIUS Universal Domain Physical Safety."""

DOMAIN_STATE_TABLE = """
| Domain | State Fields | Unit | Description |
|--------|--------------|------|-------------|
| Energy | soc_mwh, load_mw, renewables_mw | MWh, MW | Battery SOC, grid load, renewables |
| AV | position_m, speed_mps, speed_limit_mps, lead_position_m | m, m/s | Longitudinal kinematics |
| Industrial | temp_c, pressure_mbar, humidity_pct, power_mw | °C, mbar, %, MW | Process variables |
| Healthcare | hr_bpm, spo2_pct, respiratory_rate | bpm, %, /min | Vital signs |
"""

DOMAIN_SAFETY_TABLE = """
| Domain | Safety Predicate | Config Key |
|--------|------------------|------------|
| Energy | SOC ∈ [min_soc, max_soc]; charge/discharge mutual exclusion | min_soc_mwh, max_soc_mwh |
| AV | v ≤ v_max; x_lead - x ≥ d_min(v) | speed_limit_mps, min_headway_m |
| Industrial | temp ∈ [T_min, T_max]; power ∈ [0, P_max] | temp_min_c, power_max_mw |
| Healthcare | hr ∈ [hr_min, hr_max]; SpO2 ≥ spo2_min; rr ∈ [rr_min, rr_max] | hr_min_bpm, spo2_min_pct |
"""

FAULT_TAXONOMY_TABLE = """
| Fault | OQE Indicator | Effect on w_t | Domains |
|-------|----------------|----------------|---------|
| Dropout | NaN in signal | → 0.0 | All |
| Delay/jitter | timestamp gap > cadence | exponential decay | All |
| Spike | |value - last| exceeds range | multiply by (1 - β) | All |
| Stale/frozen | unchanged for k steps | FTIT stale counter | All |
| Bias/drift | residual exceeds threshold | Page-Hinkley | All |
"""

PIPELINE_STAGES_TABLE = """
| Stage | Name | Adapter Method | Output |
|-------|------|----------------|--------|
| 1 | Detect | compute_oqe | w_t, flags |
| 2 | Calibrate | build_uncertainty_set | U_t, meta |
| 3 | Constrain | tighten_action_set | A_t |
| 4 | Shield | repair_action | safe_action, repair_meta |
| 5 | Certify | emit_certificate | certificate |
"""
