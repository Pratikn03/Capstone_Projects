# ORIUS Battery Framework — Phase 0: Inputs & Orientation

**Status**: Orientation complete. All three canonical inputs confirmed. Repo fully mapped.

**Source of truth:** The repo thesis (`orius_battery_409page_figures_upgraded_main.tex` + `chapters/` + `appendices/`) is the canonical source of edits. Agent artifact zips are reference-only. See `SOURCE_OF_TRUTH_POLICY.md` for the full hierarchy. Run `make analyze-artifact` to compare an artifact zip against the repo thesis.

---

## 1. Three Canonical Inputs

| # | File | Location | Role |
|---|------|----------|------|
| 1 | `orius_battery_409page_figures_upgraded_main.tex` | `<repo-root>/` | LaTeX driver for the 409-page ORIUS battery thesis — master document structure reference |
| 2 | `orius_battery_proofs_tables_figures_expanded_updated.pdf` | `<repo-root>/` | Compiled PDF — primary reading surface for reviewers and proofs |
| 3 | Implementation plan (pasted from artifact ZIP) | Captured in `01-codex-plan-extracted.md` | 16-section battery-only implementation manual — the canonical agent task list |

> The implementation plan that lived **inside** the artifact ZIP is the
> 16-section document titled "ORIUS Battery-Only Implementation Plan." It was
> recovered from the ZIP and is re-structured in
> `01-codex-plan-extracted.md`.

---

## 2. LaTeX Chapter-to-Concept Map

The main LaTeX driver (`orius_battery_409page_figures_upgraded_main.tex`) organizes the thesis into 6 parts.

### Part I — Why ORIUS Exists (Background)
| Chapter | Title (inferred) | Core ORIUS concept |
|---------|------------------|--------------------|
| ch01 | Introduction | Motivation: observational safety illusion |
| ch02 | Related Work | Prior art (model-free, MPC, conformal) |
| ch03 | Problem Formulation | Formal setup: `z_t`, `x_t`, observed vs true state |
| ch04 | Observed State Safety Illusion | **Central claim**: OASG existence, hidden violation |

### Part II — Battery Domain: First Full Validation
| Chapter | Title (inferred) | Core ORIUS concept |
|---------|------------------|--------------------|
| ch05 | ORIUS System Context | System architecture, control loop, data pipeline |
| ch06 | Data, Telemetry, Scope | DE OPSD + US EIA-930 datasets, targets, signals |
| ch07 | Battery Dynamics & Dispatch | SOC model, LP/robust dispatch, optimizer |
| ch08 | Forecasting & Calibration | 6 model families, CQR, walk-forward evaluation |
| ch09 | DC3S Battery Adapter | OQE→RUI→SAF→Shield→Certify pipeline |
| ch10 | CPSBench Battery Track | Benchmark structure, fault schedules, metrics |

### Part III — Battery Experiments & Empirical Proof
| Chapter | Title (inferred) | Core ORIUS concept |
|---------|------------------|--------------------|
| ch11 | Main Battery Results | **Locked**: 3.9% TSVR baseline, 0% DC3S, 2.8% IR |
| ch12 | Ablations & Failure Analysis | Component ablations (OQE-only, RUI-only, no-shield) |
| ch13 | Case Studies & Operational Traces | 48-hour trace, fault episode deep-dives |
| ch14 | Battery Lessons & Domain Interpretation | What the numbers mean physically |

### Part IV — Battery Theorem Validation
| Chapter | Title (inferred) | Core ORIUS concept |
|---------|------------------|--------------------|
| ch15 | Assumptions, Notation, Proof Discipline | A1–A8 assumption register |
| ch16 | Battery Theorem: OASG Existence | Theorem 1 proof |
| ch17 | Battery Theorem: Safety Preservation | Theorem 2 proof |
| ch18 | ORIUS Core Bound (Battery) | Theorem 3: E[V] ≤ α(1−w̄)T |
| ch19 | No Free Safety (Battery) | Theorem 4: quality-ignorant failure |
| ch20 | Temporal & Behavioral Extensions | Certificate horizon, expiration, half-life |

### Part V — Extended Battery Validation & Thesis Hardening
| Chapter | Title (inferred) | Core ORIUS concept |
|---------|------------------|--------------------|
| ch21 | Fault Performance & Stress Tests | **Missing**: full fault-performance table |
| ch22 | Latency & Systems Footprint | **Missing**: locked latency table |
| ch23 | Hyperparameter Surface & Stability | α, κ_r, κ_s sweeps |
| ch24 | Conditional Coverage & Subgroups | Reliability-bin coverage audit |
| ch25 | Regional Decomposition & Real Prices | DE vs US transfer |
| ch26 | Asset Preservation & Aging Proxy | SOH-aware recalibration |
| ch27 | Hardware-in-the-Loop Validation | **Missing**: HIL evidence package |

### Part VI — Battery-Exclusive Advanced Extensions
| Chapter | Title (inferred) | Core ORIUS concept |
|---------|------------------|--------------------|
| ch28 | Certificate Half-Life & Blackout | SCADA blackout, 12/24/48h studies |
| ch29 | Graceful Degradation & Safe Landing | Fallback policy, safe-zone descent |
| ch30 | ORIUS Bench Battery Track | Benchmark release chapter |
| ch31 | Compositional Safety (Battery Fleets) | Two-battery fleet, shared transformer |
| ch32 | Adversarial Robustness & Active Probing | Sensor spoofing, active sensitivity probing |

### Lifting Battery to ORIUS (near-conclusion chapters — also ch21–ch24 alias)
| Chapter | Content |
|---------|---------|
| ch21 (alias) | Battery-to-Universal ORIUS generalization |
| ch22 (alias) | What this thesis proves |
| ch23 (alias) | Research roadmap |
| ch24 (alias) | Conclusion |

### Appendices (A–Q)
| Appendix | Content |
|----------|---------|
| app_a | Notation register |
| app_b | Assumptions (A1–A8) |
| app_c | Full proofs |
| app_d | Extended results |
| app_e | Reliability audits |
| app_f | Fault specifications |
| app_g | Adapter interface |
| app_h | Four-reviewer audit |
| app_i | HIL BOM & safety |
| app_j | Sweep and latency protocols |
| app_k | Blueprint coverage matrix |
| app_l | Artifact figure/table index |
| app_m | Verified theorems & gap audit |
| app_n | Defense lock templates |
| app_o | Claim scope & citation policy |
| app_p | Battery gap hardening synthesis |
| app_q | Editorial integration & locking |

---

## 3. Artifact ZIP → Repo Directory Map

The artifact ZIP contents map directly to the confirmed ORIUS repo at `<repo-root>/`.

| Artifact ZIP concept | Repo directory / file | Status |
|----------------------|-----------------------|--------|
| DC3S pipeline code | `src/orius/dc3s/` | **Implemented** |
| OQE (observation quality) | `src/orius/dc3s/quality.py` | **Implemented** |
| RUI (uncertainty inflation) | `src/orius/dc3s/ambiguity.py`, `calibration.py` | **Implemented** |
| SAF (safe action feasibility) | `src/orius/dc3s/shield.py`, `safety_filter_theory.py` | **Implemented** |
| Certificate generation | `src/orius/dc3s/certificate.py`, `rac_cert.py` | **Implemented** |
| FTIT (fault-tolerant interval) | `src/orius/dc3s/ftit.py` | **Implemented** |
| Drift detection | `src/orius/dc3s/drift.py` (Page-Hinkley) | **Implemented** |
| Forecasting models (6 families) | `src/orius/forecasting/ml_gbm.py`, `dl_lstm.py`, `dl_nbeats.py`, `dl_patchtst.py`, `dl_tcn.py`, `dl_tft.py` | **Implemented** |
| CQR / conformal calibration | `src/orius/forecasting/uncertainty/cqr.py`, `conformal.py` | **Implemented** |
| Reliability-Mondrian CQR | `src/orius/forecasting/uncertainty/reliability_mondrian.py` | **Implemented** |
| Battery LP dispatch | `src/orius/optimizer/lp_dispatch.py` | **Implemented** |
| Robust / CVaR dispatch | `src/orius/optimizer/robust_dispatch.py`, `risk.py` | **Implemented** |
| CPSBench benchmark | `src/orius/cpsbench_iot/runner.py`, `plant.py`, `scenarios.py`, `metrics.py` | **Implemented** |
| Battery plant physics | `src/orius/cpsbench_iot/plant.py` | **Implemented** |
| Monitoring / drift | `src/orius/monitoring/` | **Implemented** |
| Streaming pipeline | `src/orius/streaming/` | **Implemented** |
| FastAPI REST service | `services/api/` | **Implemented** |
| HIL edge agent | `iot/edge_agent/agent.py`, `drivers/sim.py`, `drivers/modbus_tcp.py` | **Implemented (sim); hardware pending** |
| Closed-loop simulator | `iot/simulator/run_closed_loop.py` | **Implemented** |
| DC3S demo script | `scripts/run_dc3s_demo.py` | **Implemented** |
| CPSBench run script | `scripts/run_cpsbench.py` | **Implemented** |
| Latency benchmark | `scripts/benchmark_dc3s_steps.py` | **Implemented; output needs lock** |
| 48h trace generator | `scripts/generate_48h_trace.py` | **MISSING — must build** |
| Fault-performance table | `reports/publication/fault_performance_table.csv` | **MISSING — needs final run** |
| Locked impact results | `reports/impact_summary.csv` | **Locked (DE surface)** |
| Locked CPSBench results | `reports/publication/dc3s_main_table_ci.csv` | **Locked** |
| Locked latency summary | `reports/publication/dc3s_latency_summary.csv` | **Locked (p95 only; p99 missing)** |
| Walk-forward backtest | `reports/walk_forward_report.json` | **Locked** |
| Subgroup coverage | `reports/publication/reliability_group_coverage.csv` | **Locked** |
| Transfer stress | `reports/publication/transfer_stress.csv` | **Locked** |
| Paper LaTeX | `paper/paper.tex` | **Active** |
| Paper configs/assets | `paper/assets/`, `paper/metrics_manifest.json` | **Locked snapshots** |
| Reproducibility lock | `reports/publish/reproducibility_lock.json` | **Locked** |

---

## 4. Concept-to-Repo Quick Lookup

| ORIUS concept | Primary repo location |
|---------------|-----------------------|
| `z_t` (clean telemetry) | `src/orius/streaming/schemas.py` |
| `tilde_z_t` (degraded telemetry) | `src/orius/cpsbench_iot/scenarios.py` (fault injection) |
| `w_t` (observation quality) | `src/orius/dc3s/quality.py` → `compute_reliability()` |
| `d_t` (drift flag) | `src/orius/dc3s/drift.py` → `PageHinkleyDetector` |
| `C_t(α)` (base conformal) | `src/orius/forecasting/uncertainty/cqr.py` |
| `C_t^RAC(α)` (RAC interval) | `src/orius/dc3s/rac_cert.py` → `compute_q_multiplier()` |
| `U_t(α)` (uncertainty set) | `src/orius/dc3s/calibration.py` → `build_uncertainty_set()` |
| `a_t^*` (candidate action) | `src/orius/optimizer/lp_dispatch.py` |
| `a_t^safe` (repaired action) | `src/orius/dc3s/shield.py` → `repair_action()` |
| `A_t` (tightened safe set) | `src/orius/dc3s/shield.py` + `safety_filter_theory.py` |
| Certificate object | `src/orius/dc3s/certificate.py` → `make_certificate()` |
| Theorem 1 (OASG existence) | `src/orius/cpsbench_iot/scenarios.py` (fault episodes) |
| Theorem 2 (safety preservation) | `src/orius/dc3s/guarantee_checks.py` |
| Theorem 3 (core bound) | `src/orius/dc3s/coverage_theorem.py` |
| Theorem 4 (no free safety) | `src/orius/cpsbench_iot/baselines.py` + `scenarios.py` |

---

## 5. Highest-Priority Gaps (from the extracted implementation plan §10, §13)

These items are confirmed missing and must be generated before the manuscript is locked:

| Priority | Item | Target file | Script to run/build |
|----------|------|-------------|---------------------|
| 1 | Fault-performance table | `reports/publication/fault_performance_table.csv` | `scripts/run_cpsbench.py` full sweep |
| 2 | 48-hour operational trace | `reports/publication/48h_trace.csv` + figure | `scripts/generate_48h_trace.py` — **must build** |
| 3 | Latency table (p99 missing) | `reports/publication/dc3s_latency_summary.csv` | `scripts/benchmark_dc3s_steps.py` → relock |
| 4 | HIL evidence package | `reports/hil/` | `iot/simulator/run_closed_loop.py` (sim) or real modbus |

---

*Next: see `01-codex-plan-extracted.md` for the full annotated implementation
plan.*
