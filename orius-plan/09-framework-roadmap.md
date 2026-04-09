# ORIUS Battery Framework — Phase 9: Hardening, Generalization & Roadmap

**Status**: Gap audit complete. Extension map defined. Final output checklist with current status.

---

## 1. Gap Audit Table

Complete status of every item from the extracted implementation plan
replace-next queue (§13), cross-referenced with the current repo state.

### Priority 1 — Must replace first

| Item | Plan ref | Current status | Target file | Command |
|------|---------------|----------------|-------------|---------|
| Fault-performance stress table | §7.4, §13.1 | **MISSING** — CPSBench sweep exists but fault-perf pivot not generated | `reports/publication/fault_performance_table.csv` | `python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml` then pivot |
| 48-hour operational trace | §7.4, §13.1 | **MISSING** — script does not exist yet | `reports/publication/48h_trace.csv` + `fig_48h_trace.pdf` | Build `scripts/generate_48h_trace.py`, then run it |
| Latency micro-benchmark (p99, max) | §10.3, §13.1 | **PARTIAL** — mean/p95 locked; p99/max missing | `reports/publication/dc3s_latency_summary.csv` | `python scripts/benchmark_dc3s_steps.py --n-trials 10000` |
| HIL results table + plot | §10, §13.1 | **MISSING** — hardware not run; software HIL available | `reports/hil/` | `python iot/simulator/run_closed_loop.py` (software) |

### Priority 2 — Replace second

| Item | Plan ref | Current status | Target file | Command |
|------|---------------|----------------|-------------|---------|
| Hyperparameter sweep surfaces (α, κ_r, κ_s) | §10.4, §13.2 | **MISSING** — `run_sensitivity_sweeps.py` exists but not for DC3S params | `reports/publication/hyperparam_sweep.csv` | `python scripts/run_sensitivity_sweeps.py` (extend for α/κ) |
| Reliability-bin / conditional coverage | §6.5, §13.2 | **DONE** — `reliability_group_coverage.csv` locked | — | `python scripts/compute_reliability_group_coverage.py` to re-run |
| Blackout / half-life outputs | §9.1, §13.2 | **MISSING** — ch28 written, no simulation runs | `reports/publication/blackout_halflife.csv` | Build `scripts/run_blackout_study.py` |
| Graceful degradation traces | §9.2, §13.2 | **MISSING** — ch29 written, no traces generated | `reports/publication/graceful_degradation.csv` | Extend `iot/simulator/run_closed_loop.py` with blackout fault |

### Priority 3 — Replace third

| Item | Plan ref | Current status | Target file | Command |
|------|---------------|----------------|-------------|---------|
| Battery benchmark leaderboard | §7.4, §13.3 | **PARTIAL** — `cpsbench_merged_sweep.csv` exists; leaderboard format missing | `reports/publication/cpsbench_leaderboard.csv` | Derive from fault-perf table after Priority 1 |
| Battery fleet composition results | §9.4, §13.3 | **MISSING** — ch31 written, plant not extended to two-battery | `reports/publication/fleet_composition.csv` | Extend `cpsbench_iot/plant.py` + `runner.py` |
| Active probing / spoofing results | §9.5, §13.3 | **MISSING** — ch32 written, sensitivity probe is heuristic only | `reports/publication/probing_detection.csv` | Extend `dc3s/rac_cert.py` sensitivity probe |
| Battery aging / asset preservation table | §9.1 (ch26), §13.3 | **MISSING** — ch26 written, no aging runs | `reports/publication/aging_calibration.csv` | Extend `monitoring/retraining.py` with SOH trigger |

---

## 2. Extension Layer Map

All extensions must be clearly marked in code and manuscript as not part of the locked battery base.

### Extension 1: Dynamic Drift D(x,u) — ch20 late

**Status**: MISSING

**Plan ref**: §5.3.1

**Goal**: Replace constant drift bound `D` with state-dependent `D(x_t, a_t)`.

**Implementation path**:
```python
# src/orius/dc3s/safety_filter_theory.py

# Current (static):
drift_bound = cfg.dc3s.drift.constant_bound

# Extension (dynamic):
def compute_dynamic_drift_bound(soc_mwh: float, action_mw: float, cfg) -> float:
    """State-dependent drift: larger near SOC boundaries."""
    soc_normalized = soc_mwh / cfg.battery.e_max_mwh
    boundary_proximity = min(soc_normalized, 1.0 - soc_normalized)
    # Drift grows as battery approaches edge
    return cfg.dc3s.drift.base_bound * (1.0 + cfg.dc3s.drift.edge_scale / (boundary_proximity + 0.01))
```

**Theorem linkage**: Prove that `D(x,u)` upper-bounds physical drift near the battery edge (ch20 extension theorem).

**Manuscript location**: ch20 §5 — "Dynamic drift extension"

---

### Extension 2: Soft Electrochemical Boundary — ch26

**Status**: MISSING

**Plan ref**: §5.3.2

**Goal**: Move from hard SOC walls to a soft electrochemical stress boundary that penalizes depth-of-discharge and power stress.

**Implementation path**:
```python
# src/orius/optimizer/lp_dispatch.py

# Add electrochemical stress penalty to objective
def _electrochemical_stress_penalty(soc_mwh: float, action_mw: float, cfg) -> float:
    """Approximate electrochemical stress as depth-of-discharge penalty."""
    dod = 1.0 - (soc_mwh / cfg.battery.e_max_mwh)
    power_stress = abs(action_mw) / cfg.battery.p_max_mw
    return cfg.optimizer.stress_weight * (dod ** 2 + power_stress ** 2)
```

**Manuscript location**: ch26 — "Asset preservation and aging proxy"

---

### Extension 3: ECM / Non-Linear Battery Model — future

**Status**: MISSING — advanced extension, mark clearly as future work

**Plan ref**: §5.3.3

**Goal**: Replace simple linear SOC dynamics with an equivalent circuit model (ECM) that captures voltage sag, internal resistance, temperature effects.

**Implementation path**: New file `src/orius/cpsbench_iot/plant_ecm.py` — inherits from `BatteryPlant` but overrides `step()` with ECM dynamics.

**Manuscript location**: Appendix or future work section only.

---

### Extension 4: Online Calibration Under Aging — ch26

**Status**: MISSING

**Plan ref**: §6.6

**Goal**: Rolling conformal recalibration triggered by state-of-health decline.

**Implementation path**: `src/orius/monitoring/retraining.py` → add `OnlineCalibrator` class (spec in `06-certificates-and-forecasting.md` §9).

**Key parameters**:
- `ema_alpha = 0.01` (EWMA decay for residual store)
- `soh_trigger_threshold = 0.95` (recalibrate when SOH drops below 95%)
- Rolling window: 8760 hours (1 year)

---

### Extension 5: Certificate Half-Life Under Blackout — ch28

**Status**: PARTIAL (chapter written, no simulation runs)

**Plan ref**: §9.1

**Goal**: Quantify how long a certificate remains valid under SCADA blackout (12h / 24h / 48h).

**Implementation path**:

```bash
# Build: scripts/run_blackout_study.py
python - <<'PY'
# Pseudocode
from orius.cpsbench_iot.scenarios import generate_episode
from orius.cpsbench_iot.runner import CPSBenchRunner

for blackout_hours in [12, 24, 48]:
    artifacts = generate_episode(scenario='dropout', horizon=blackout_hours, seed=42)
    runner = CPSBenchRunner(config=cpsbench_cfg)
    results = runner.run_single_episode(
        x_obs=artifacts.x_obs, x_true=artifacts.x_true,
        scenario='dropout', controller='dc3s_wrapped'
    )
    # Measure: at what step does the certificate's guarantee expire?
    # Expected: validity shrinks as blackout extends
PY
```

**Target outputs**:
- `reports/publication/blackout_halflife.csv` — columns: `blackout_hours`, `certificate_validity_steps`, `final_tsvr`
- Figure: validity horizon vs blackout duration

---

### Extension 6: Graceful Degradation / Safe Landing — ch29

**Status**: PARTIAL (chapter written, no traces)

**Plan ref**: §9.2

**Goal**: Demonstrate that DC3S switches from profit-seeking to safe landing mode under extended observation loss.

**Implementation path**: Extend `dc3s/shield.py` with a `safe_landing_mode` flag:
```python
# When w_t < w_landing_threshold for k_landing steps in a row:
# - Disable LP optimizer
# - Switch to: a_safe = project onto soc_target (safe zone center)
# - Emit certificate with intervention_reason = "safe_landing"
```

---

### Extension 7: Battery Fleet Composition — ch31

**Status**: MISSING

**Plan ref**: §9.4

**Goal**: Compositional safety for a two-battery microgrid sharing a transformer constraint.

**Implementation path**:

```python
# src/orius/cpsbench_iot/plant_fleet.py

class BatteryFleetPlant:
    """Two-battery fleet with shared transformer constraint."""
    def __init__(self, battery_1: BatteryPlant, battery_2: BatteryPlant, transformer_limit_mw: float):
        self.b1 = battery_1
        self.b2 = battery_2
        self.transformer_limit = transformer_limit_mw

    def step(self, action_1_mw: float, action_2_mw: float) -> tuple[float, float]:
        # Enforce: |action_1 + action_2| <= transformer_limit_mw
        total = action_1_mw + action_2_mw
        if abs(total) > self.transformer_limit:
            scale = self.transformer_limit / abs(total)
            action_1_mw *= scale
            action_2_mw *= scale
        return self.b1.step(action_1_mw, 0.0), self.b2.step(action_2_mw, 0.0)
```

---

### Extension 8: Active Probing / Adversarial Robustness — ch32

**Status**: MISSING

**Plan ref**: §9.5

**Goal**: Active sensitivity probing to detect sensor spoofing; adversarial fault injection to test robustness.

**Implementation path**: Upgrade `dc3s/rac_cert.py` sensitivity probe from `heuristic` to `active`:
```yaml
# configs/dc3s.yaml
rac_cert:
  sensitivity_probe: active   # was: heuristic
  sens_eps_mw: 25.0
```

Active probing: perturb action by `±sens_eps_mw`, measure SOC response difference, detect if response is inconsistent with known dynamics (spoofing indicator).

---

## 3. New Domain Plugin Pattern

When extending ORIUS beyond batteries, implement this minimal interface:

```python
# Template: src/orius/domains/<domain_name>/adapter.py

class DomainAdapter:
    """
    Interface that any new ORIUS domain must implement.
    Battery adapter: src/orius/dc3s/__init__.py (run_dc3s_step)
    """

    def compute_oqe(self, event, last_event, cfg) -> float:
        """Compute observation quality score w_t ∈ [w_min, 1.0]."""
        raise NotImplementedError

    def build_uncertainty_set(self, lower, upper, w_t, d_t, s_t, cfg) -> tuple:
        """Build uncertainty set U_t from RAC-inflated interval."""
        raise NotImplementedError

    def tighten_action_set(self, uncertainty_set, domain_constraints, cfg) -> object:
        """Compute tightened safe action set A_t."""
        raise NotImplementedError

    def repair_action(self, a_star, action_set, cfg) -> tuple[float, bool, str]:
        """Project a_star onto A_t. Returns (a_safe, intervened, reason)."""
        raise NotImplementedError

    def run_dynamics(self, state, action) -> object:
        """Execute one-step domain dynamics. Returns new true state."""
        raise NotImplementedError

    def check_violation(self, true_state, constraints) -> dict:
        """Check if true state violates domain safety constraints."""
        raise NotImplementedError
```

**Battery implementation**: All 6 methods are implemented across `dc3s/quality.py`, `calibration.py`, `safety_filter_theory.py`, `shield.py`, `cpsbench_iot/plant.py`, `cpsbench_iot/metrics.py`.

**What a non-battery domain must define**:
1. Its own `compute_oqe()` — domain-specific telemetry quality signals
2. Its own `tighten_action_set()` — domain-specific constraint geometry
3. Its own `run_dynamics()` — domain physics
4. Its own `check_violation()` — domain-specific safety definition

`build_uncertainty_set()` and `repair_action()` can reuse the battery implementations unchanged (they are domain-agnostic).

---

## 4. Final Output Package Checklist

### 4.1 Code / Runtime Outputs

| Output | Target | Status | Command |
|--------|--------|--------|---------|
| Locked battery benchmark table | `reports/publication/fault_performance_table.csv` | **MISSING** | `run_cpsbench.py` + pivot |
| Locked 48-hour trace | `reports/publication/48h_trace.csv` + `fig_48h_trace.pdf` | **MISSING** | `scripts/generate_48h_trace.py` |
| Locked latency table (full) | `reports/publication/dc3s_latency_summary.csv` | **PARTIAL** | `benchmark_dc3s_steps.py` |
| Locked subgroup coverage | `reports/publication/reliability_group_coverage.csv` | **DONE** | — |
| Locked transfer stress | `reports/publication/transfer_stress.csv` | **PARTIAL** (pending artifacts) | `run_transfer_stress.py` |
| Locked ablation table | `reports/publication/dc3s_ablation_table.csv` | **DONE** | — |
| Locked cost-safety Pareto | `reports/publication/cost_safety_pareto.csv` | **DONE** | — |
| Locked CPSBench merged sweep | `reports/publication/cpsbench_merged_sweep.csv` | **DONE** | — |
| HIL evidence (software) | `reports/hil/` | **MISSING** | `iot/simulator/run_closed_loop.py` |
| Hyperparameter sweep surfaces | `reports/publication/hyperparam_sweep.csv` | **MISSING** | `run_sensitivity_sweeps.py` (extend) |
| Blackout / half-life tables | `reports/publication/blackout_halflife.csv` | **MISSING** | `scripts/run_blackout_study.py` |
| Graceful degradation traces | `reports/publication/graceful_degradation.csv` | **MISSING** | extend `run_closed_loop.py` |
| Fleet composition tables | `reports/publication/fleet_composition.csv` | **MISSING** | extend `plant.py` + `runner.py` |
| Active probing detection | `reports/publication/probing_detection.csv` | **MISSING** | extend `rac_cert.py` |

### 4.2 Manuscript Outputs

| Output | Status | Notes |
|--------|--------|-------|
| Unified battery-only PDF | **PARTIAL** — `paper/paper.tex` compiles | Run `make paper-refresh` |
| Unified main `.tex` | **DONE** — `paper/paper.tex` | Single master source |
| Source zip | **PARTIAL** — `dist/orius-0.1.0.tar.gz` | Rebuild after final lock |
| Figure folder (vector) | **PARTIAL** — some figures done, 48h trace and latency missing | See figure list below |
| Bibliography / citation-clean | **PARTIAL** | Run `bibtex paper` during compile |

**Missing figures**:
| Figure | File | Script |
|--------|------|--------|
| 48h operational trace | `paper/assets/figures/fig_48h_trace.pdf` | `scripts/generate_48h_trace.py` |
| Latency stack | `paper/assets/figures/fig_latency_stack.pdf` | `scripts/benchmark_dc3s_steps.py` |
| Hyperparameter surfaces | `paper/assets/figures/fig_hyperparam_heatmap.pdf` | `scripts/run_sensitivity_sweeps.py` |
| Blackout horizon | `paper/assets/figures/fig_blackout_halflife.pdf` | `scripts/run_blackout_study.py` |
| Graceful degradation flow | `paper/assets/figures/fig_graceful_degradation.pdf` | `iot/simulator/run_closed_loop.py` |
| HIL setup | `paper/assets/figures/fig_hil_setup.pdf` | Manual / hardware diagram |

### 4.3 Proof Outputs

| Output | Status | Notes |
|--------|--------|-------|
| Updated theorem appendix | **PARTIAL** — `app_c_full_proofs` in LaTeX | Review T3 / T4 against locked code |
| Theorem-to-evidence mapping | **PARTIAL** — `paper/claim_matrix.csv` | Update after Priority 1 runs |
| Assumption register | **PARTIAL** — `app_b_assumptions` in LaTeX | A1–A8 defined; verify code enforcement |

---

## 5. Implementation Priority Order for the Next Agent Session

When picking up from here, work in this exact order:

### Step 1 — Generate fault-performance table (1-2 hours)
```bash
python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml
# Then: pivot to fault_performance_table.csv (see 07-evaluation-and-audits.md §5)
```

### Step 2 — Build and run 48h trace generator (2-4 hours)
1. Write `scripts/generate_48h_trace.py` using the spec in `07-evaluation-and-audits.md` §6
2. Run it with: `--region DE --fault stale_sensor --window 48 --seed 42`
3. Verify output columns match spec
4. Generate figure

### Step 3 — Lock full latency table (30 minutes)
```bash
python scripts/benchmark_dc3s_steps.py --n-trials 10000
# Copy output to reports/publication/dc3s_latency_summary.csv
```

### Step 4 — Software HIL run (1-2 hours)
```bash
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 &
python iot/simulator/run_closed_loop.py --scenario stale_sensor --horizon 48
kill %1
```

### Step 5 — Paper refresh and pre-publication audit (1 hour)
```bash
make paper-assets && make paper-compile
make publish-audit
```

### Step 6 — Hyperparameter sweeps (2-4 hours)
Extend `scripts/run_sensitivity_sweeps.py` to sweep over α and κ_r.

---

## 6. Scope Discipline Reminders

As new chapters and experiments are added, enforce these rules:

1. **Battery-only in main body**: ch1–ch14 and all locked tables must use battery-domain language only.
2. **Extension chapters honest scope**: ch21–ch32 extensions must explicitly state "battery-only" or "battery-first interpretation."
3. **No synthetic values in locked layers**: ch11–ch14 tables must all trace to a locked run_id or CSV row.
4. **Placeholder marking**: Any synthetic value in ch21–ch32 must be annotated `% PLACEHOLDER — replace before lock` in LaTeX.
5. **Evidence mapping**: Every new claim must be added to `paper/claim_matrix.csv` with a locked evidence pointer.
6. **Assumptions version**: Any change to A1–A8 must bump `assumptions_version` in `dc3s.yaml` and all generated certificates.

---

## 7. Six-Month Roadmap

| Month | Focus | Key deliverables |
|-------|-------|-----------------|
| Month 1 | Generate Priority 1 missing outputs | fault-perf table, 48h trace, latency table, software HIL |
| Month 2 | Hyperparameter sweeps + conditional coverage | hyperparam surfaces, reliability audit updates |
| Month 3 | Blackout / half-life study | ch28 simulation runs, blackout_halflife.csv |
| Month 4 | Graceful degradation + fleet composition | ch29 traces, ch31 two-battery results |
| Month 5 | Active probing + adversarial robustness | ch32 spoofing results, probing_detection.csv |
| Month 6 | Final manuscript lock | all tables replaced, paper PDF frozen, publication artifact |

---

*All 10 ORIUS SOP files are now complete. Start execution with `08-operations-sop.md` §4 (Priority 1 missing outputs).*
