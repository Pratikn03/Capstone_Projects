# ORIUS Battery Framework — Phase 1: Implementation Plan (Extracted & Annotated)

**Source**: Recovered from the artifact ZIP. Original title: "ORIUS Battery-Only Implementation Plan."
**Status legend**: `[DONE]` = implemented in repo | `[PARTIAL]` = code exists, output not locked | `[MISSING]` = must build/run

---

## §1 — Central Battery-Only Claim

The system must prove:

> A battery dispatch controller can appear safe on observed telemetry while violating true physical battery limits under degraded observation, and a certificate-aware runtime layer can close that gap by making observation reliability part of the control state.

**Empirical anchors (locked — do not overwrite):**

| Metric | Value | Source |
|--------|-------|--------|
| Deterministic baseline TSVR | 3.9% | `reports/publication/dc3s_main_table_ci.csv` row `nominal,deterministic_lp` |
| Deterministic P95 severity | 333.375 MWh | paper.pdf locked table |
| DC3S TSVR | 0.0% | `dc3s_main_table_ci.csv` row `nominal,dc3s_wrapped` |
| DC3S intervention rate | 2.8% | same file |
| DE cost savings | 7.11% | `reports/impact_summary.csv` |
| DE carbon reduction | 0.30% | `reports/impact_summary.csv` |
| DE peak shaving | 6.13% | `reports/impact_summary.csv` |
| US cost savings | 0.11% | locked — paper.pdf |
| US carbon reduction | 0.13% | locked — paper.pdf |
| US peak shaving | 0.00% | locked — paper.pdf |

> **Rule**: never overwrite these with synthetic values. If a new run updates them, re-lock all downstream tables.

---

## §2 — Source-of-Truth File Hierarchy

### Tier A — Direct battery code/evidence (consult first)
| File | Repo equivalent | Status |
|------|----------------|--------|
| `paper.pdf` | `paper/paper.tex` + `paper/assets/` | [DONE] — primary LaTeX source |
| `orius_updated.pdf` | `paper/FLAGSHIP_IMPLEMENTATION_PLAN.md` + revision notes | [DONE] — revision guidance |
| locked CSV artifacts | `reports/publication/*.csv`, `reports/impact_summary.csv` | [DONE] — locked results |

### Tier B — Theorem and proof discipline
| File | Repo equivalent | Status |
|------|----------------|--------|
| `ORIUS_Theorem_Math_Master_Handbook.docx` | `paper/assets/` + ch15–ch19 LaTeX | [DONE] — captured in thesis chapters |
| `ORIUS_COMPLETE_MATHEMATICAL_PROOFS.*` | appendix `app_c_full_proofs` | [DONE] — in LaTeX appendix |
| `ORIUS_PROOF_VERIFICATION_CHECKLIST.*` | `app_m_verified_theorems_and_gap_audit` | [PARTIAL] — needs gap fill |

### Tier C — Battery-only extensions
| Paper | Topic | Primary chapter |
|-------|-------|----------------|
| Paper 2: Certificate Half-Life | Blackout / horizon | ch28 |
| Paper 3: Graceful Degradation | Safe landing | ch29 |
| Paper 4: CPSBench | Battery track only | ch10, ch30 |
| Paper 5: Compositional Safety | Battery fleet | ch31 |
| Paper 6: CertOS | Deployment/runtime | ch27 (HIL) |

---

## §3 — Non-Negotiable Rules (agent must enforce)

### 3.1 Battery-only scope
- Use: "battery-domain result", "battery-first validation surface", "reference-domain foundation"
- Never: "fully universal", "validated across all cyber-physical systems",
  "deployment-complete"

### 3.2 Evidence discipline
Every claim maps to one of: locked run ID | theorem ID | CSV row | known script | stated limitation.

### 3.3 Placeholder discipline
Synthetic values allowed only in ch21–ch32 (hardening) until replaced by real runs. Mark with `% PLACEHOLDER — replace before lock` in LaTeX.

### 3.4 Proof discipline (7 steps per theorem)
1. Define symbols
2. Define assumptions by ID (A1–A8)
3. State theorem narrowly
4. Give proof strategy
5. Give readable class-style proof
6. Explain operational meaning
7. State limits of the theorem

---

## §4 — System Architecture

### Control loop
```
Forecast → Optimize → Dispatch → Measure → Monitor
                ↑ highest risk ↑
         Optimize → Dispatch
```

### Runtime objects (all 12 must persist in code and docs)
| Symbol | Name | Code location |
|--------|------|--------------|
| `z_t` | Clean telemetry | `streaming/schemas.py` TelemetryEvent |
| `tilde_z_t` | Degraded telemetry | `cpsbench_iot/scenarios.py` fault injection |
| `o_t` | Observed state | `dc3s/state.py` DC3SState.observed_soc |
| `x_t` | True physical state | `cpsbench_iot/plant.py` true_soc |
| `w_t` | Observation quality score | `dc3s/quality.py` compute_reliability() |
| `d_t` | Drift evidence flag | `dc3s/drift.py` PageHinkleyDetector |
| `s_t` | Sensitivity / staleness signal | `dc3s/rac_cert.py` compute_dispatch_sensitivity() |
| `C_t(α)` | Base conformal interval | `forecasting/uncertainty/cqr.py` |
| `C_t^RAC(α)` | RAC-Cert interval | `dc3s/rac_cert.py` RACCertModel |
| `U_t(α)` | Uncertainty set | `dc3s/calibration.py` build_uncertainty_set() |
| `a_t^*` | Optimizer candidate action | `optimizer/lp_dispatch.py` |
| `a_t^safe` | Repaired safe action | `dc3s/shield.py` repair_action() |
| `A_t` | Tightened safe action set | `dc3s/shield.py` + `safety_filter_theory.py` |

### Five runtime stages
| Stage | Name | Code file | Status |
|-------|------|-----------|--------|
| 1 | Detect / OQE | `dc3s/quality.py` | [DONE] |
| 2 | Calibrate / RUI / RAC-Cert | `dc3s/rac_cert.py`, `calibration.py`, `ambiguity.py` | [DONE] |
| 3 | Constrain / SAF set tightening | `dc3s/shield.py`, `safety_filter_theory.py` | [DONE] |
| 4 | Shield / repair | `dc3s/shield.py` repair_action() | [DONE] |
| 5 | Certify / audit | `dc3s/certificate.py` make_certificate() | [DONE] |

---

## §5 — Battery Model

### SOC dynamics
```
SOC_{t+1}^true = SOC_t^true + (eta_c * [a_t]+ - [a_t]- / eta_d) / E_max * dt
```

Implemented in:
- `cpsbench_iot/plant.py` — physics truth model
- `optimizer/lp_dispatch.py` — dispatch constraints
- `dc3s/guarantee_checks.py` → `next_soc()` — one-step safety check

### Battery parameters (from `configs/dc3s.yaml` + optimization.yaml)
| Parameter | Meaning | Config key |
|-----------|---------|-----------|
| E_max | Energy capacity (MWh) | `ftit.e_max_mwh` |
| P_max | Power limit (MW) | dispatch config |
| R_max | Ramp limit (MW/h) | dispatch config |
| eta_c | Charge efficiency | plant config |
| eta_d | Discharge efficiency | plant config |
| SOC_min / SOC_max | Hard SOC walls | `shield.reserve_soc_pct_drift` |

### Late extensions (mark clearly in code)
| Extension | Description | Status |
|-----------|-------------|--------|
| Dynamic drift D(x,u) | State-dependent drift bound | [MISSING — ch20 extension] |
| Soft electrochemical boundary | DoD / voltage sag penalty | [MISSING — ch26 extension] |
| ECM non-linear model | Equivalent circuit model | [MISSING — advanced, mark as future] |

---

## §6 — Data and Forecasting

### Locked datasets
| Dataset | Region | Files |
|---------|--------|-------|
| DE OPSD battery surface | Germany | `data/raw/opsd/`, `configs/forecast.yaml` |
| US EIA-930 / BA surfaces | MISO, PJM, ERCOT | `data/raw/eia930/`, `configs/forecast_eia930.yaml` |

### Forecasting model families (all 6 implemented)
| Model | Type | File | Status |
|-------|------|------|--------|
| GBM (LightGBM) | ML | `forecasting/ml_gbm.py` | [DONE] — strongest baseline |
| LSTM | DL | `forecasting/dl_lstm.py` | [DONE] |
| TCN | DL | `forecasting/dl_tcn.py` | [DONE] |
| N-BEATS | DL | `forecasting/dl_nbeats.py` | [DONE] |
| TFT | DL | `forecasting/dl_tft.py` | [DONE] |
| PatchTST | DL | `forecasting/dl_patchtst.py` | [DONE] |

### Uncertainty calibration pipeline
```
Raw predictions → CQR (cqr.py) → base intervals → RACCertModel (rac_cert.py) → C_t^RAC
```
| Step | File | Status |
|------|------|--------|
| CQR base intervals | `forecasting/uncertainty/cqr.py` | [DONE] |
| Regime-stratified CQR | `forecasting/uncertainty/reliability_mondrian.py` | [DONE] |
| RAC-Cert inflation | `dc3s/rac_cert.py` | [DONE] |
| Walk-forward evaluation | `forecasting/backtest.py` | [DONE] |

### Known calibration weaknesses (must stay documented)
1. Marginal coverage > conditional coverage
2. Subgroup under-coverage in some regimes
3. Reliability-conditioned audits needed → `scripts/compute_reliability_group_coverage.py`

### Late extension: online calibration under aging
Path: `monitoring/retraining.py` — extend with EWMA residual store + SOH-aware trigger
Status: [MISSING — ch26]

---

## §7 — Benchmark Implementation

### CPSBench battery track
| Component | File | Status |
|-----------|------|--------|
| Benchmark runner | `cpsbench_iot/runner.py` | [DONE] |
| Battery plant physics | `cpsbench_iot/plant.py` | [DONE] |
| Fault scenarios | `cpsbench_iot/scenarios.py` | [DONE] |
| Metrics (TSVR, IR, severity) | `cpsbench_iot/metrics.py` | [DONE] |
| Baselines (det-LP, robust-LP, CVaR) | `cpsbench_iot/baselines.py` | [DONE] |
| Full sweep script | `scripts/run_cpsbench.py` | [DONE] |

### Benchmark invariants (must never break)
1. Truth/observed trajectory separation — `plant.py` maintains `true_soc` vs `observed_soc`
2. Replayable fault schedule — `scenarios.py` uses deterministic seed
3. Stable metric schema — `metrics.py` column names are frozen
4. Stable logging schema — `dc3s/state.py` DuckDB schema

### Core metrics (all must be present in every run)
| Metric | Column name | File |
|--------|------------|------|
| True-State Violation Rate | `violation_rate_mean` | `dc3s_main_table_ci.csv` |
| P95 severity | `severity_p95` | needs run |
| Intervention Rate | `intervention_rate_mean` | `dc3s_main_table_ci.csv` |
| Cost savings | `expected_cost_usd_mean` | `dc3s_main_table_ci.csv` |
| Useful work preserved | to be added | [MISSING] |

### Mandatory benchmark outputs still missing
| Output | Target file | Run command |
|--------|------------|-------------|
| Fault-performance table (7 faults × 4 controllers) | `reports/publication/fault_performance_table.csv` | `python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml` |
| 48-hour operational trace | `reports/publication/48h_trace.csv` | `python scripts/generate_48h_trace.py` — **must build** |
| Benchmark leaderboard | `reports/publication/cpsbench_leaderboard.csv` | derived from fault-perf table |

---

## §8 — Theorems and Proofs

### Four-theorem ladder
| Theorem | Statement | Code verification | Status |
|---------|-----------|------------------|--------|
| T1: OASG existence | ∃ episode where action is obs-safe but true-unsafe | `cpsbench_iot/scenarios.py` fault episodes | [DONE] |
| T2: One-step safety preservation | true state inside U_t + repaired action ∈ A_t → next state safe | `dc3s/guarantee_checks.py` | [DONE] |
| T3: Core Safety Bound | E[V] ≤ α(1−w̄)T | `dc3s/coverage_theorem.py` | [DONE] |
| T4: No Free Safety | quality-ignorant controller fails under admissible fault sequence | `cpsbench_iot/baselines.py` + `scenarios.py` | [DONE] |

### Temporal/behavioral theorems (ch20, ch28–ch29)
| Theorem | Status |
|---------|--------|
| Certificate validity horizon | [PARTIAL — ch20 written, code needs link] |
| Expiration bound | [PARTIAL] |
| Half-life corollary | [PARTIAL — ch28] |
| Feasible fallback existence | [PARTIAL — ch29] |
| Graceful degradation dominance | [PARTIAL — ch29] |
| Safe-budget monotonicity | [PARTIAL] |

### Assumption register
| ID | Assumption | Config / code enforcement |
|----|-----------|--------------------------|
| A1 | Bounded model error | `dc3s.yaml` → `alpha0`, `infl_max` |
| A2 | Bounded telemetry error | `dc3s.yaml` → `reliability.min_w` = 0.05 |
| A3 | Feasible safe repair exists | `dc3s/shield.py` projection always feasible by construction |
| A4 | Known/identified dynamics | `cpsbench_iot/plant.py` — known SOC model |
| A5 | Monotone bounded uncertainty inflation | `dc3s/ambiguity.py` → `max_extra` = 1.0 |
| A6 | Bounded detector lag | `dc3s/drift.py` → `warmup_steps` = 48 |
| A7 | Causal certificate update rule | `dc3s/certificate.py` — per-step forward-only |
| A8 | Admissible fallback policy | `dc3s/shield.py` → `mode: projection` |

---

## §9 — Late Extension Chapters

| Chapter | Topic | Status | Implementation path |
|---------|-------|--------|---------------------|
| ch28 | Certificate Half-Life (blackout) | [PARTIAL — written, no sim runs] | `iot/simulator/run_closed_loop.py` with SCADA blackout fault |
| ch29 | Graceful Degradation / Safe Landing | [PARTIAL — written, no runs] | `dc3s/shield.py` fallback mode |
| ch30 | Battery Benchmark Track | [PARTIAL — needs leaderboard table] | `scripts/run_cpsbench.py` full sweep |
| ch31 | Battery Fleet Composition | [MISSING] | Extend `cpsbench_iot/plant.py` to two-battery |
| ch32 | Adversarial Robustness / Active Probing | [MISSING] | Extend `dc3s/rac_cert.py` sensitivity probe |

---

## §10 — Hardware Hardening Plan

### Missing evidence (highest priority)
| Item | Status | Path |
|------|--------|------|
| Fault-performance table | [MISSING] | `scripts/run_cpsbench.py` |
| 48-hour dropout trace | [MISSING] | `scripts/generate_48h_trace.py` |
| Latency table (p99) | [PARTIAL] | `scripts/benchmark_dc3s_steps.py` — relock |
| HIL evidence package | [MISSING — hardware] | `iot/edge_agent/` + `iot/simulator/run_closed_loop.py` |

### HIL minimum package
1. Setup diagram (benchtop battery emulator + instrumented telemetry path)
2. Hardware table (BOM)
3. Timing table
4. Real fault-response plot
5. Safety outcome table

Simulator fallback: `iot/edge_agent/drivers/sim.py` (SimBatteryDriver) provides physics emulation for software-only HIL runs.

### Latency benchmark (current locked values)
From `reports/publication/dc3s_latency_summary.csv`:

| Stage | Mean (ms) | P95 (ms) | P99 | Status |
|-------|-----------|----------|-----|--------|
| Reliability scoring (OQE) | 0.0196 | 0.0239 | missing | [PARTIAL] |
| Drift update (Page-Hinkley) | 0.0004 | 0.0004 | missing | [PARTIAL] |
| Uncertainty set build | 0.0100 | 0.0132 | missing | [PARTIAL] |
| Action repair (SAF) | 0.0019 | 0.0020 | missing | [PARTIAL] |
| Full DC3S step | 0.0329 | 0.0354 | missing | [PARTIAL] |

> Re-run `scripts/benchmark_dc3s_steps.py` with `--percentiles 50,95,99,max` to get p99/max columns.

### Hyperparameter sweeps needed
| Parameter | Range | Output |
|-----------|-------|--------|
| alpha (α) | 0.02–0.15 | TSVR vs IR heatmap |
| kappa_r (k_quality) | 0.1–0.5 | interval width vs coverage |
| kappa_s (k_sensitivity) | 0.2–0.8 | sensitivity vs repair rate |

---

## §11 — Exact Command Execution Plan

### Environment inspection
```bash
cd <repo-root>
pwd && ls -la
python -m orius --version  # if CLI exists
```

### Locate core battery modules
```bash
find src/orius/dc3s -type f -name "*.py" | sort
find src/orius/optimizer -type f -name "*.py" | sort
find src/orius/cpsbench_iot -type f -name "*.py" | sort
```

### Verify locked evidence
```bash
python - <<'PY'
import pandas as pd
files = {
    'reports/impact_summary.csv': 'DE dispatch impact',
    'reports/publication/dc3s_main_table_ci.csv': 'DC3S main results',
    'reports/publication/dc3s_latency_summary.csv': 'DC3S latency',
    'reports/publication/reliability_group_coverage.csv': 'Group coverage',
    'reports/publication/transfer_stress.csv': 'Transfer stress',
}
for f, label in files.items():
    try:
        df = pd.read_csv(f)
        print(f'\n=== {label} ({f}) ===')
        print(df.to_string(index=False))
    except Exception as e:
        print(f'MISSING: {f} — {e}')
PY
```

### Run CPSBench battery track
```bash
python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml
```

### Run latency benchmark
```bash
python scripts/benchmark_dc3s_steps.py
```

### Generate fault-performance table (after CPSBench run)
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('reports/publication/cpsbench_merged_sweep.csv')
pivot = df.pivot_table(
    index='scenario', columns='controller',
    values=['violation_rate_mean', 'intervention_rate_mean', 'expected_cost_usd_mean'],
    aggfunc='first'
)
pivot.to_csv('reports/publication/fault_performance_table.csv')
print(pivot)
PY
```

### Generate 48-hour trace (script must be built — see §07)
```bash
python scripts/generate_48h_trace.py --region DE --fault stale --window 48
```

---

## §12 — Manuscript Build Plan

### Build policy
One master source tree — do not create separate stitched versions.
```bash
cd <repo-root>/paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

### Figure rebuild priority
| Figure | Target | Script |
|--------|--------|--------|
| Battery system overview | `paper/assets/figures/fig01_architecture.*` | rebuild in consistent vector style |
| DC3S pipeline | `paper/assets/figures/fig02_dc3s_step.png` | `scripts/run_dc3s_demo.py` |
| SOC violation vs dropout | `paper/assets/figures/fig03_04_soc_violation_vs_dropout.*` | `scripts/run_sensitivity_sweeps.py` |
| 48-hour trace | missing | `scripts/generate_48h_trace.py` |
| Latency stack | missing | `scripts/benchmark_dc3s_steps.py` |

### Table lock policy
Tables in ch11–ch14 must not contain unresolved placeholders. Tables in ch21–ch32 may use synthetic values marked `% PLACEHOLDER`.

---

## §13 — Replace-Next Queue (Prioritized Task List)

### Priority 1 — Must replace first
| Item | Current status | Target file | Command |
|------|---------------|-------------|---------|
| Fault-performance stress table | MISSING | `reports/publication/fault_performance_table.csv` | `python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml` |
| 48-hour operational trace | MISSING | `reports/publication/48h_trace.csv` | `python scripts/generate_48h_trace.py` |
| Latency micro-benchmark (p99) | PARTIAL | `reports/publication/dc3s_latency_summary.csv` | `python scripts/benchmark_dc3s_steps.py` |
| HIL results table + plot | MISSING | `reports/hil/` | `python iot/simulator/run_closed_loop.py` |

### Priority 2 — Replace second
| Item | Command |
|------|---------|
| Hyperparameter sweep surfaces | `python scripts/run_sensitivity_sweeps.py` |
| Reliability-bin / conditional coverage | `python scripts/compute_reliability_group_coverage.py` |
| Blackout / half-life outputs | build ch28 sim run |
| Graceful degradation traces | `python iot/simulator/run_closed_loop.py --fault blackout` |

### Priority 3 — Replace third
| Item | Command |
|------|---------|
| Battery benchmark leaderboard | `python scripts/run_cpsbench.py` full sweep → derive |
| Battery fleet composition results | extend `cpsbench_iot/plant.py` → two-battery |
| Active probing / spoofing results | extend `dc3s/rac_cert.py` |
| Battery aging / asset preservation table | extend `monitoring/retraining.py` |

---

## §14 — Final Output Package Checklist

### 14.1 Code / runtime outputs
| Output | Target | Status |
|--------|--------|--------|
| Locked battery benchmark table | `reports/publication/fault_performance_table.csv` | [MISSING] |
| Locked 48-hour trace | `reports/publication/48h_trace.csv` + figure | [MISSING] |
| Locked latency table (full) | `reports/publication/dc3s_latency_summary.csv` | [PARTIAL — p99 missing] |
| Locked subgroup coverage | `reports/publication/reliability_group_coverage.csv` | [DONE] |
| Locked transfer stress | `reports/publication/transfer_stress.csv` | [DONE] |
| Locked fleet composition tables | `reports/publication/fleet_composition.csv` | [MISSING] |
| Locked probing detection tables | `reports/publication/probing_detection.csv` | [MISSING] |
| Locked blackout / half-life tables | `reports/publication/blackout_halflife.csv` | [MISSING] |
| Locked graceful-degradation traces | `reports/publication/graceful_degradation.csv` | [MISSING] |

### 14.2 Manuscript outputs
| Output | Status |
|--------|--------|
| Unified battery-only PDF | [PARTIAL — paper.tex compiles] |
| Unified main .tex | [DONE — paper.tex] |
| Source zip | [PARTIAL — dist/orius-0.1.0.tar.gz exists] |
| Figure folder with vector diagrams | [PARTIAL — some figures done] |
| Bibliography / citation-clean build | [PARTIAL] |

### 14.3 Proof outputs
| Output | Status |
|--------|--------|
| Updated theorem appendix | [PARTIAL — app_c in LaTeX] |
| Theorem-to-evidence mapping table | [PARTIAL — paper/claim_matrix.csv] |
| Assumption register | [PARTIAL — app_b in LaTeX] |

---

## §15 — What the Agent Must Not Do

1. Do not broaden main claims beyond battery domain — use "battery-first" language only
2. Do not overwrite locked metrics (§1 table) with synthetic values
3. Do not introduce AV / surgery / aerospace sections into the main empirical path
4. Do not hide missing evidence — mark it clearly and generate where possible
5. Do not let tables remain placeholder-heavy in ch11–ch14

---

## §16 — Short Operational Summary

**If you only read one section**, read this:

1. The battery system has a **real hidden failure** (3.9% TSVR under det-LP) and a **real shielded fix** (0% TSVR under DC3S).
2. `paper/paper.tex` is the main manuscript source of truth.
3. Proof discipline is defined in ch15 and app_b, app_c.
4. Highest-priority missing evidence (per `orius_updated.pdf`):
   - fault-performance table → `python scripts/run_cpsbench.py`
   - 48-hour trace → `python scripts/generate_48h_trace.py` (must build)
   - latency micro-benchmark → `python scripts/benchmark_dc3s_steps.py`
5. HIL is still not done. `iot/edge_agent/drivers/sim.py` provides software HIL.
6. The right final output is battery-only, deeply integrated, and honest about locked vs missing.

---

*Next: see `02-framework-architecture.md` for the module tree and data-flow diagrams.*
