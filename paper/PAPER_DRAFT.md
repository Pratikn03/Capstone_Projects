# GridPulse: An Integrated Machine Learning System for Renewable Energy Forecasting and Carbon-Aware Battery Dispatch

**Author:** Pratik Niroula  
**Affiliation:** MNSU CIT (Major), Math (Minor)  
**Date:** February 19, 2026

## Abstract
GridPulse is an end-to-end decision system that links forecasting, uncertainty estimation, battery dispatch optimization, operational safety controls, and post-decision evaluation for power-system operations. The implementation is organized as a closed loop (`Forecast -> Optimize -> Dispatch -> Measure -> Monitor`) so that model outputs are directly converted into constrained actions and then measured against baselines in the same artifact pipeline. This manuscript uses a strict dataset-scoped latest evidence policy locked to repository artifacts dated February 17, 2026. Canonical decision-impact sources are `reports/impact_summary.csv` for Germany (DE) and `reports/eia930/impact_summary.csv` for the United States (US). Canonical stochastic metrics are taken from `reports/research_metrics_de.csv` run `20260217_165756` and `reports/research_metrics_us.csv` run `20260217_182305`.

Under this lock, DE impact is **7.11%** cost savings, **0.30%** carbon reduction, and **6.13%** peak shaving. US impact is **0.11%** cost savings, **0.13%** carbon reduction, and **0.00%** peak shaving. Stochastic outcomes are DE EVPI_robust **2.32**, DE EVPI_deterministic **-30.40**, DE VSS **2,708.61**, and US EVPI_robust **10,279,851.74**, US EVPI_deterministic **24,915,503.93**, US VSS **297,092.71**. Beyond reporting these outcomes, the core thesis contribution is governance: each publication-facing claim is tied to explicit run IDs, source files, and rounding rules through `paper/metrics_manifest.json`, `paper/claim_matrix.csv`, and `scripts/validate_paper_claims.py`.

The evidence shows two simultaneous realities: (1) robust/stochastic value is positive in both regions, and (2) direct percent impact magnitude can differ substantially by region even with a shared pipeline. Therefore, this thesis frames GridPulse as a decision-quality and reproducibility system, not only a forecasting benchmark.

## Keywords
Machine Learning, Energy Forecasting, Battery Dispatch, Robust Optimization, Conformal Prediction, Stochastic Programming, MLOps, Grid Operations

## 1. Introduction

### 1.1 Operational Motivation
Prediction-only systems stop before operational action. In grid operations, this is not sufficient: an operator needs dispatch decisions that satisfy physical constraints under uncertain demand and renewable generation. GridPulse is built as a closed decision loop:

`Forecast -> Optimize -> Dispatch -> Measure -> Monitor`

This loop is implemented across model training, uncertainty calibration, optimization, monitoring, and API/dashboard delivery, with explicit governance to prevent metric drift across manuscript versions.

In practice, this architecture addresses a common implementation failure in applied ML projects: high model quality in isolation with weak operational accountability once predictions are handed off. GridPulse explicitly binds model outputs to dispatch feasibility, safety guards, and measurable economic/carbon outcomes so that "good forecast metrics" and "good operational decisions" are assessed together.

### 1.2 Problem Statement
The problem addressed in this thesis is integrated decision quality, not isolated prediction error. The system must:
1. Produce accurate short-horizon forecasts for load, wind, and solar.
2. Quantify forecast uncertainty in a form usable by dispatch optimization.
3. Compute physically feasible battery/grid actions under cost and carbon objectives.
4. Demonstrate measurable decision impact against baseline policies.
5. Maintain reproducible claim traceability across markdown, LaTeX, and release documents.

This framing moves the optimization target from "minimize one model metric" to "optimize a constrained cyber-physical workflow under uncertainty." In that workflow, a low forecast error can still yield poor decisions if interval calibration, solver behavior, or control-plane safety are weak.

### 1.3 Research Questions
1. Can one architecture maintain high forecasting performance across DE and US datasets with different regimes?
2. Does uncertainty-aware dispatch provide measurable value relative to deterministic dispatch?
3. How large is decision impact (cost/carbon/peak) under dataset-scoped latest artifacts?
4. Can run-scoped governance prevent cross-document metric contradictions?

Each question is answered with locked in-repo evidence. This manuscript intentionally avoids claims requiring untracked notebooks, ad hoc local reruns, or post hoc manual calculations that are not serialized in the evidence paths declared in Section 5 and Section 7.

### 1.4 Contributions
1. A production-oriented multi-layer GridPulse implementation with forecast, optimization, monitoring, and serving layers.
2. A conformal + adaptive interval workflow integrated into robust dispatch evaluation.
3. A dataset-scoped latest metric lock with explicit run IDs and reproducible evidence paths.
4. A publication governance mechanism (`metrics_manifest`, claim matrix, validator script) that prevents legacy-claim regression.

Contribution boundaries are stated explicitly. This thesis does not claim complete market realism for all settlement regimes, universal transferability across all operators, or causal policy-level effects. It claims an operationally integrated and evidence-governed system with reproducible DE/US outcomes under the locked artifacts.

### 1.5 Thesis Scope and Reading Guide
To support future editing, this draft is organized so each section can be expanded or trimmed independently:
1. Sections 2-4 define implementation and methods with code-level anchors.
2. Sections 5-8 define evidence policy and claim governance.
3. Section 6 contains the canonical quantitative results.
4. Sections 9-13 interpret findings, limits, and roadmap.
5. Appendices provide replication commands, artifact inventory, and copy-ready writing guidance.

If manuscript length must be reduced later, keep Sections 6, 7, and 14 intact first; those contain the metric-locked claims that must remain synchronized across formats.

## 2. System Architecture and Implementation

### 2.1 Decision Loop and Layering
GridPulse is implemented as an eight-layer pipeline consistent with repository modules and docs (`docs/ARCHITECTURE.md`):
1. Data ingestion and feature generation.
2. Forecast model training/inference.
3. Uncertainty interval calibration.
4. Anomaly detection.
5. Dispatch optimization (deterministic + robust).
6. Impact and stochastic evaluation.
7. Monitoring and retraining logic.
8. API + dashboard serving.

The layering is intentionally conservative: each layer writes explicit artifacts consumed by the next layer instead of relying on implicit in-memory contracts. This supports auditability and simplifies failure isolation. For example, optimizer-level anomalies can be traced back to concrete forecast and interval artifacts instead of inferred from logs alone.

### 2.2 Component-to-Code Map
| Layer | Purpose | Primary paths |
|---|---|---|
| Data pipeline | Build/validate region features and splits | `src/gridpulse/data_pipeline/`, `src/gridpulse/pipeline/run.py` |
| Forecasting | GBM/LSTM/TCN train + predict | `src/gridpulse/forecasting/` |
| Uncertainty | Conformal + adaptive interval logic | `src/gridpulse/forecasting/uncertainty/conformal.py` |
| Optimization | Deterministic MILP and robust dispatch | `src/gridpulse/optimizer/lp_dispatch.py`, `src/gridpulse/optimizer/robust_dispatch.py` |
| Monitoring | Drift detection and retraining decisions | `src/gridpulse/monitoring/` |
| Safety | BMS-style limits and watchdog health | `src/gridpulse/safety/` |
| API serving | Forecast/optimize/monitor/anomaly endpoints | `services/api/main.py`, `services/api/routers/` |
| Frontend | Operator dashboard and regional views | `frontend/` |

The code map is publication-relevant because this thesis claims implementation-level contributions, not only conceptual workflows. All functional claims in this manuscript should be traceable to these modules or their direct dependencies.

### 2.3 Runtime Interfaces
FastAPI includes routes for forecast, intervals, anomaly, optimization, and monitoring:
- `/forecast`
- `/forecast/with-intervals`
- `/anomaly`
- `/optimize`
- `/monitor`
- `/monitor/model-info`
- `/dc3s/step`
- `/dc3s/audit/{command_id}`
- `/health`, `/ready`, `/metrics`

Operational control routes also exist (`/system/health`, `/system/heartbeat`, `/control/dispatch`) with API-key scope checks and safety validation in `services/api/main.py`.

Endpoint behavior is scoped as follows:
1. Forecasting routes provide point forecasts and interval outputs.
2. Optimization routes expose deterministic and robust dispatch pathways.
3. Monitoring routes return drift and retraining state plus research metric snapshots.
4. Control routes are guarded by scope checks and runtime safety checks (watchdog and BMS validation).

Existing runtime endpoints remain unchanged; these interfaces are additive extensions for Section 1-3 implementation coverage.

### 2.4 Artifact Handoff Contracts
Primary artifact contracts used by this manuscript:
1. Forecast/dashboard metrics: `data/dashboard/de_metrics.json`, `data/dashboard/us_metrics.json`.
2. Dataset profiles: `data/dashboard/de_stats.json`, `data/dashboard/us_stats.json`, `data/dashboard/manifest.json`.
3. Decision impact: `reports/impact_summary.csv`, `reports/eia930/impact_summary.csv`.
4. Stochastic metrics: `reports/research_metrics_de.csv`, `reports/research_metrics_us.csv`.
5. Robustness diagnostics: `reports/publication/tables/table6_robustness.csv`.

These contracts are treated as public interfaces between offline pipelines and publication outputs. If any contract path or schema changes, `paper/metrics_manifest.json` and `paper/claim_matrix.csv` must be updated in the same revision.

### 2.5 Deployment and Runtime Modes
GridPulse supports both research and operational execution patterns:
1. Offline batch mode for training, evaluation, and report generation.
2. API service mode (`services/api/main.py`) for live forecast/optimization requests.
3. Dashboard mode (`frontend/`) for operator-facing visualization and inspection.

The manuscript does not assume uninterrupted real-time operation for all components. Instead, it documents explicit fallbacks (for example, baseline dispatch when optimization is infeasible) and health checks that allow safe degradation.

### 2.6 End-to-End Request Lifecycle
An operational request path can be summarized in six steps:
1. Load latest feature window and resolve requested target(s).
2. Retrieve model bundle and produce forecast vectors.
3. Optionally attach conformal intervals where calibration artifacts exist.
4. Solve deterministic or robust dispatch with configured constraints.
5. Return action plan plus cost/carbon summaries when available.
6. Record monitoring signals and maintain control-plane heartbeats.

This lifecycle is central to thesis reproducibility because it ties observed outcomes to deterministic processing stages with explicit files, routes, and solver statuses.

### 2.7 Section 1-3 Implementation Update (Publish Blockers, Streaming, DC3S)
Section 1 (immediate publish blockers) was implemented by replacing frontend chat tool mock outputs with backend-derived FastAPI calls in `frontend/src/app/api/chat/tool-executors.ts`. Production tool paths now resolve through `/forecast`, `/forecast/with-intervals`, `/optimize`, `/optimize/baseline`, `/monitor`, `/anomaly`, `/health`, `/ready`, and `/monitor/model-info`, and tool execution explicitly fails on backend errors instead of returning fabricated fallback payloads. Deployment/runtime hygiene changes were also applied: dashboard container/service ports were normalized to `3000`, dashboard deploy workflow Dockerfile targeting was corrected, local absolute report paths were removed, and tracked coverage artifacts were removed and ignored for reproducibility hygiene.

Section 2 (streaming correctness) was implemented with a new CLI module `src/gridpulse/streaming/run_consumer.py` supporting:
`python -m gridpulse.streaming.run_consumer --config configs/streaming.yaml --max-messages <N>`.
This loader maps YAML config into streaming `AppConfig`, instantiates `StreamingIngestConsumer`, and closes resources cleanly. Persistence behavior was made explicit via public `write_event(...)` in `src/gridpulse/streaming/consumer.py` and integrated in `src/gridpulse/streaming/worker.py` after schema validation. The expected persistence artifact is `data/interim/streaming.duckdb` with table `telemetry_events`.

Section 3 (DC3S) was implemented with new config `configs/dc3s.yaml`, DC3S modules in `src/gridpulse/dc3s/` (`quality.py`, `drift.py`, `calibration.py`, `shield.py`, `certificate.py`, `state.py`), and FastAPI router endpoints in `services/api/routers/dc3s.py`: `POST /dc3s/step` and `GET /dc3s/audit/{command_id}`. The step endpoint returns proposed and shield-repaired safe actions, uncertainty payloads, and certificate identifiers; audit retrieval returns persisted certificate JSON. The `heuristic` controller mode is intentionally not implemented in this phase and is explicitly rejected at runtime.

Frontend live-ops behavior for this flow is now explicitly implemented in `frontend/src/app/(dashboard)/page.tsx`, `frontend/src/lib/api/dc3s-client.ts`, `frontend/src/components/dashboard/DC3SLiveCard.tsx`, and `frontend/src/app/api/dc3s/audit/[commandId]/route.ts`. The dashboard initializes DC3S polling at 15 seconds, allows operator selection of auto-refresh (`Off`, `5s`, `10s`, `15s`, `30s`, `60s`), and supports manual refresh. The card displays the active `command_id` and exposes a direct audit link that opens `/api/dc3s/audit/{commandId}`; the Next.js route encodes the ID and proxies to FastAPI `/dc3s/audit/{commandId}` with explicit error propagation.

## 3. Data Assets, Scope, and Feature Design

### 3.1 Dataset Policy Lock (Canonical)
This manuscript uses the dashboard profile lock generated at `2026-02-17T11:15:38.623283` (`data/dashboard/manifest.json`).

| Region | Dataset Label | Rows (hourly) | Columns | Engineered Features | Start (UTC) | End (UTC) | Approx Days |
|---|---|---:|---:|---:|---|---|---:|
| DE | Germany (OPSD) | 17,377 | 98 | 94 | 2018-10-07 23:00 | 2020-09-30 23:00 | 724.04 |
| US | USA (EIA-930 MISO) | 13,638 | 118 | 114 | 2024-07-01 06:00 | 2026-01-20 11:00 | 568.25 |

The lock is a governance boundary, not only a descriptive snapshot. Any value outside this scope may still be analytically useful, but it cannot be published as a canonical thesis claim without manifest-level reconciliation.

### 3.2 Signal and Target Definitions
Target variables in both regions are:
- `load_mw`
- `wind_mw`
- `solar_mw`

Region-specific context:
- DE includes `price_eur_mwh` and `carbon_kg_per_mwh` in the profile.
- US includes generation-mix channels (`coal_mw`, `gas_mw`, `nuclear_mw`, `hydro_mw`) and `price_usd_mwh` / `carbon_kg_per_mwh` features.

This target alignment enables cross-region architectural comparisons while still preserving dataset-specific feature context. In other words, model families and pipeline stages are shared, but feature spaces remain region-aware.

### 3.3 Feature Family Composition
| Region | Weather Features | Lag Features | Calendar Features | Total Features |
|---|---:|---:|---:|---:|
| DE | 40 | 36 | 6 | 94 |
| US | 56 | 42 | 6 | 114 |

The US profile has a broader weather feature footprint and generation-mix context, which can affect uncertainty calibration and optimization behavior. This is one structural reason cross-region magnitude differences should be interpreted cautiously.

### 3.4 Target Distribution Snapshot
Values below are from `targets_summary` in dashboard stats files.

| Region | Target | Mean (MW) | Std (MW) | Min (MW) | Max (MW) | Median (MW) | Non-zero (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| DE | load_mw | 55,156.42 | 9,998.06 | 31,923.0 | 76,925.0 | 54,752.0 | 100.0 |
| DE | wind_mw | 14,368.98 | 10,321.58 | 136.0 | 46,064.0 | 11,728.0 | 100.0 |
| DE | solar_mw | 5,013.16 | 7,665.92 | 0.0 | 32,947.0 | 156.0 | 56.8 |
| US | load_mw | 75,757.21 | 11,819.46 | 52,940.0 | 120,343.0 | 73,564.0 | 100.0 |
| US | wind_mw | 10,955.94 | 6,041.83 | 0.0 | 26,132.0 | 10,183.5 | 100.0 |
| US | solar_mw | 2,875.42 | 3,842.56 | 0.0 | 14,315.0 | 110.0 | 94.6 |

### 3.5 Missingness and Data Quality
`missing_pct` is empty in both dashboard stats JSON files, indicating no unresolved missingness in the locked feature profiles used for this manuscript snapshot.

Data quality interpretation in this thesis is therefore focused on distributional behavior and calibration quality rather than imputation-heavy preprocessing effects in the locked profile artifacts.

### 3.6 Scope Boundaries and Legacy Drift
This thesis explicitly excludes non-locked legacy profile claims. In particular, previously circulated row/count/date-range claims from older report families are treated as non-canonical unless reconciled to the dashboard profile lock above.

### 3.7 Leakage-Safe Temporal Framing
The training configuration defines a 24-hour forecast horizon and 168-hour lookback with time-aware cross-validation and an explicit gap (`cross_validation.n_folds = 10`, `cross_validation.gap = 24`) in both DE and US forecast configs. This enforces temporal ordering and reduces leakage risk when comparing model families.

Because this thesis is decision-oriented, leakage prevention is not only a forecasting requirement. Leakage in upstream forecasting would directly contaminate optimization and impact estimates, so temporal separation is treated as an end-to-end validity condition.

## 4. Methods

### 4.1 Forecasting Method

#### 4.1.1 Model Families
GridPulse trains three model families per target:
1. **GBM (LightGBM)**: operational primary model in current locked dashboards.
2. **LSTM**: sequence deep model for long temporal dependence.
3. **TCN**: convolutional sequence model with dilated temporal receptive fields.

#### 4.1.2 Training Setup
Configured horizon/lookback settings from `configs/train_forecast.yaml` and `configs/train_forecast_eia930.yaml`:
- Forecast horizon: 24 hours.
- Lookback: 168 hours.
- Time-aware cross-validation enabled in config (10 folds with 24-hour gap).
- Seed control and optional multi-seed training are defined in configs.

Implementation detail that matters for reproducibility:
1. DE and US both keep GBM as enabled baseline model with LightGBM hyperparameter blocks in config.
2. Deep model blocks are present for LSTM and TCN in both configs, but parameterization differs by region configuration profile.
3. Cross-validation behavior is explicitly controlled by config values rather than implicit defaults.
4. Training artifacts are persisted into region-specific output paths (`artifacts/models` and `artifacts/models_eia930` pathways through configs/scripts).

Runtime training controls are explicit in the CLI contract (`src/gridpulse/forecasting/train.py`, `scripts/train_dataset.py`). `--tune` force-enables Optuna tuning, `--no-tune` disables tuning even when YAML enables it, `--ensemble` trains multi-seed GBM members from config seeds, `--max-seeds` caps ensemble size, `--n-trials` overrides Optuna trial count, and `--top-pct` overrides top-trial aggregation percentile. In tuned runs, selected parameters and tuning metadata are serialized in saved bundles; in ensemble runs, per-seed metrics, seed list, and `ensemble_models` are persisted so inference and reporting remain run-reproducible.

#### 4.1.3 Forecast Metrics
Primary metrics: RMSE, MAE, sMAPE, R2. MAPE is recorded but can become numerically unstable when targets have many near-zero periods (especially solar and some wind segments), so interpretation prioritizes sMAPE and absolute metrics.

Metric interpretation policy in this thesis:
1. RMSE and MAE are primary for magnitude-sensitive operational quality.
2. sMAPE is used for relative comparability across targets with different scales.
3. R2 is included as explanatory fit context, not as a standalone decision metric.
4. MAPE is treated as supplementary and interpreted carefully for near-zero regimes.

#### 4.1.4 Forecast Inference Contract
API-level forecast inference (`services/api/routers/forecast.py`) follows a strict contract:
1. Load feature matrix from configured path.
2. Resolve model bundle by target through explicit/fallback mapping.
3. Emit next-horizon predictions with generated timestamp and metadata.
4. Surface missing-model states explicitly in API response metadata.

This contract is relevant for reproducibility because missing artifacts are not silently imputed; they are surfaced as explicit missing-target notes.
Resolution behavior is deterministic: the router first checks explicit per-target model paths from config, then applies ordered fallback search (`lstm`, `tcn`, `gbm`) under `artifacts/models`. For GBM bundles containing `ensemble_models`, inference averages member predictions; for single-model bundles, it uses the single estimator path. Quantiles are served from explicit quantile models when present, otherwise from residual quantile offsets embedded in the bundle.

### 4.2 Uncertainty: Conformal + Adaptive Intervals
Conformal calibration (alpha = 0.10 nominal) is implemented in `src/gridpulse/forecasting/uncertainty/conformal.py` with horizon-wise residual quantiles and rolling updates.

Core interval form:
- Lower bound: `y_hat - q`
- Upper bound: `y_hat + q`

Where `q` is a calibrated residual quantile (global or horizon-wise). Adaptive conformal logic updates alpha/scale behavior online while preserving bounded controls. Coverage is evaluated with PICP, and width with MPIW.

The implementation includes two linked mechanisms:
1. `ConformalInterval` computes interval widths from residual quantiles.
2. `AdaptiveConformal` updates effective alpha using bounded step logic (gamma-driven updates) when misses occur.

Default conformal config values in code include `alpha = 0.10`, `rolling = true`, and `rolling_window = 720`, which are operationally meaningful because interval behavior can evolve with recent residual behavior rather than remaining static.

### 4.3 Deterministic Dispatch Optimization
Deterministic dispatch in `src/gridpulse/optimizer/lp_dispatch.py` solves a mixed-integer linear objective with battery charge/discharge, grid import, curtailment, unmet load penalty, and SOC dynamics. Objective terms include:
1. Energy cost.
2. Carbon-weighted cost proxy.
3. Battery degradation throughput penalty.
4. Optional peak-shaving penalty.

Key constraints:
- Power balance.
- SOC recursion.
- SOC bounds.
- Charge/discharge power limits.
- Grid import capacity.

The deterministic objective can be summarized as:

```text
min sum_t [
  (cost_weight * price_t + carbon_weight * carbon_cost_t) * grid_t
  + degradation_cost * (charge_t + discharge_t)
  + curtailment_penalty * curtail_t
  + unmet_load_penalty * unmet_t
] + peak_penalty * peak
```

Key implementation choices from config and code:
1. Piecewise battery efficiency regimes (`efficiency_regime_a`, `efficiency_regime_b`) with SOC split gating.
2. Optional interval-aware risk binding through `risk.mode = worst_case_interval` and bound selectors.
3. Explicit high penalties for unmet load and curtailment to preserve physical realism in solved plans.

Default deterministic optimization config (`configs/optimization.yaml`) includes `objective.cost_weight = 1.0`, `objective.carbon_weight = 1.2`, battery capacity `20,000 MWh`, battery max power `5,000 MW`, and grid import cap `100,000 MW`.

### 4.4 Robust Dispatch Optimization
Robust dispatch in `src/gridpulse/optimizer/robust_dispatch.py` uses a two-scenario min-max LP over lower/upper load bounds with scenario-coupled epigraph objective. The objective mixes:
1. Worst-case scenario cost weight.
2. Average scenario cost weight.
3. Throughput degradation penalty.

Robust feasibility is explicitly returned with solver status, scenario costs, and operational plan outputs.

Robust objective form:

```text
min [
  risk_weight_worst_case * z
  + (1 - risk_weight_worst_case) * average_scenario_grid_cost
  + degradation_cost_per_mwh * throughput
]
subject to:
  z >= scenario_cost_lower
  z >= scenario_cost_upper
```

Important implementation behaviors:
1. Charge/discharge decisions are shared across scenarios, while grid import and SOC are scenario-indexed.
2. Non-optimal solve termination returns an explicit infeasible-safe payload instead of partial unsafe actions.
3. Solver availability is checked before solve, and status is returned in publication-facing artifacts.

### 4.5 DC3S: Drift-Calibrated Conformal Safety Shield
DC3S is implemented in `src/gridpulse/dc3s/` and exposed through `POST /dc3s/step` and `GET /dc3s/audit/{command_id}` in `services/api/routers/dc3s.py`. It composes telemetry quality weighting, drift detection, uncertainty inflation, safe action repair, and certificate-grade audit persistence.

The implemented interval inflation law is linear and bounded:
1. Base half-width is conformal quantile `q`.
2. Reliability floor is applied as `w_t <- max(w_t, min_w)`.
3. Inflation factor:
`infl = clip(1 + k_q * (1 - w_t) + k_d * 1[drift], 1, infl_max)`.
4. Final interval:
`lower = y_hat - q * infl`,
`upper = y_hat + q * infl`.

The reliability module (`quality.py`) computes `w_t` from missingness, delay, out-of-order behavior, and spike penalties. Drift logic (`drift.py`) uses Page-Hinkley updates on residual magnitude `r_t = |y_t - y_hat_t|` with configured warmup and cooldown controls.

Safety shielding (`shield.py`) supports two modes:
1. `projection`: deterministic clipping against SOC, power, and ramp constraints.
2. `robust_resolve`: attempt robust optimization under interval bounds, then project as final guard.

Certification (`certificate.py`) computes `model_hash` and `config_hash`, supports optional `prev_hash` chaining, and stores signed payload structure to DuckDB. Runtime online state (`state.py`) persists per `(zone, device, target)` keys including drift state, last inflation, latest telemetry/action context, and audit linkage pointers.

### 4.6 Monitoring and Retraining Controls
Monitoring logic (`services/api/routers/monitor.py`, `src/gridpulse/monitoring/`) includes:
- KS-based data drift checks (`ks_drift`).
- Model drift based on performance degradation thresholds.
- Retraining decision logic with cadence and new-data conditions.

Default monitoring thresholds are configured in `configs/monitoring.yaml`.

Configured defaults in the lock snapshot:
1. `data_drift.p_value_threshold = 0.01`.
2. `model_drift.metric = mape`.
3. `model_drift.degradation_threshold = 0.15`.
4. `retraining.cadence_days = 30`.
5. `retraining.min_new_data_days = 14`.

These controls establish decision rules for when to retrain and when to keep current models, preventing ad hoc retraining that can invalidate comparability across evaluation windows.

### 4.7 Safety and Operational Controls
Safety controls include:
1. BMS constraint checking for dispatch commands.
2. Watchdog heartbeat and islanding protection.
3. API scope verification for read/write operations.
4. Health and readiness endpoints for runtime gating.

`/control/dispatch` is protected by a three-step sequence in code:
1. API scope authorization (`verify_scope("write", api_key)`).
2. Watchdog islanding check (reject if isolated).
3. BMS validation of charge/discharge/SOC bounds.

This layered gate is part of the thesis contribution because it makes optimization outputs operationally enforceable rather than advisory.

### 4.8 Reproducibility Controls
Run manifests, pinned artifact paths, and non-mutating claim checks are core reproducibility controls for this thesis:
- `paper/metrics_manifest.json`
- `paper/claim_matrix.csv`
- `scripts/validate_paper_claims.py`

These controls are used as release gates, not passive documentation. A manuscript edit is considered incomplete unless the validator passes and claim rows remain traceable to canonical sources.

### 4.9 Method Assumptions and Boundary Conditions
This thesis assumes:
1. Hourly dispatch horizon and timestep consistency.
2. Price/carbon proxies are suitable for comparative decision analysis in the locked windows.
3. Forecast uncertainty is represented by interval bounds rather than full probabilistic distributions.
4. Battery and grid constraints in config are sufficient to produce physically plausible dispatch plans.

These are engineering assumptions, not universal truths; they should be revisited for different markets or finer control cadences.

### 4.10 Baselines and Comparator Policies
This manuscript uses three explicit baseline policies for decision comparisons:
1. **B1 (Deterministic LP, no uncertainty intervals)**: implemented in `src/gridpulse/optimizer/lp_dispatch.py` via `optimize_dispatch(...)` with point forecasts and without `load_interval` / `renewables_interval`. It uses battery + grid constraints and is the optimization policy reported as GridPulse in canonical impact artifacts.
2. **B2 (Grid-only, no battery)**: implemented in `src/gridpulse/optimizer/baselines.py` as `grid_only_dispatch(...)`. It sets `battery_charge_mw = 0` and `battery_discharge_mw = 0`, and serves net deficit from grid import. This is the denominator baseline for canonical DE/US impact percentages.
3. **B3 (Naive battery heuristic)**: implemented in `src/gridpulse/optimizer/baselines.py` as `naive_battery_dispatch(...)`, with a fixed rule that charges at 00-05 and discharges at 17-21.

Runtime disambiguation: while API `/optimize` may default to robust mode in service context, publication impact percentages in this thesis are produced from offline deterministic `optimize_dispatch(...)` (B1) versus grid-only baseline (B2).

## 5. Experimental Protocol

### 5.1 Evidence-Lock Policy
All publication-facing numeric claims in this draft must map to canonical values in `paper/metrics_manifest.json`. `paper/PAPER_DRAFT.md` is the authority file for writing, with synchronization rules defined in `paper/sync_rules.md`.

The evidence-lock policy is intentionally strict because this project has multiple report families and historical drafts. Without a lock, metric drift can occur even when each individual number is locally valid in some artifact.

### 5.2 Canonical Source and Run Selection
| Claim Domain | Source | Selection Rule | Locked ID / Time |
|---|---|---|---|
| DE decision impact | `reports/impact_summary.csv` | latest dataset-scoped record | snapshot used in manifest |
| US decision impact | `reports/eia930/impact_summary.csv` | latest dataset-scoped record | snapshot used in manifest |
| DE stochastic metrics | `reports/research_metrics_de.csv` | row_type = run_summary | run `20260217_165756`, `2026-02-17T16:57:56.736552` |
| US stochastic metrics | `reports/research_metrics_us.csv` | row_type = run_summary | run `20260217_182305`, `2026-02-17T18:23:06.287827` |
| Dataset profile | `data/dashboard/*_stats.json` + manifest | dashboard lock only | `2026-02-17T11:15:38.623283` |

### 5.3 Rounding and Formatting Rules
From `paper/metrics_manifest.json`:
1. Percent metrics: `round(value * 100, 2)`.
2. Monetary, mass, and power values: two-decimal display where reported.
3. Large stochastic values: thousands separators plus two decimals.

Rounding is part of the claim contract. If raw values change in source files, manuscript updates must be generated from raw values and then rounded by these explicit rules, never by manual ad hoc formatting.

### 5.4 Cross-Region Comparability Rules
1. Compare methods and governance processes across DE/US.
2. Do not assume equal effect sizes across regions.
3. Use region-scoped source artifacts and run IDs for all stochastic statements.

In practice, this means "same pipeline, different evidence contexts." The thesis compares architecture-level behavior while preserving region-specific dataset scope and run provenance.

### 5.5 Consistency Gates
The draft is considered publication-safe only when:
1. `python scripts/validate_paper_claims.py` passes.
2. Required run IDs appear exactly in markdown and LaTeX.
3. No banned legacy percentages or placeholder tokens appear.
4. All core claims are mapped in `paper/claim_matrix.csv`.

### 5.6 Reconciliation Procedure for Conflicts
When a conflicting value is found:
1. Locate claim text and claim ID in `paper/claim_matrix.csv`.
2. Identify canonical source and extraction rule from `paper/metrics_manifest.json`.
3. Replace manuscript value with canonical display value derived from rounding policy.
4. Re-run `scripts/validate_paper_claims.py`.
5. Mirror the same correction in LaTeX/DOCX sync outputs.

This procedure prevents "partial fixes" where one format is corrected while another remains stale.

## 6. Results

### 6.1 Forecast Quality Across Model Families
Source: `data/dashboard/de_metrics.json`, `data/dashboard/us_metrics.json`.

#### 6.1.1 Germany (DE)
| Target | Model | RMSE | MAE | sMAPE (%) | R2 |
|---|---|---:|---:|---:|---:|
| load_mw | GBM | 267.51 | 167.49 | 0.35 | 0.9991 |
| load_mw | LSTM | 3,474.78 | 2,722.50 | 5.38 | 0.8498 |
| load_mw | TCN | 2,668.05 | 2,031.12 | 3.94 | 0.9114 |
| wind_mw | GBM | 183.93 | 118.19 | 2.29 | 0.9993 |
| wind_mw | LSTM | 6,735.07 | 5,511.23 | 60.85 | 0.1545 |
| wind_mw | TCN | 9,196.16 | 7,167.41 | 76.67 | -0.5763 |
| solar_mw | GBM | 251.44 | 121.22 | 69.27 | 0.9992 |
| solar_mw | LSTM | 4,079.38 | 2,835.74 | 109.44 | 0.8007 |
| solar_mw | TCN | 2,702.34 | 1,583.05 | 93.76 | 0.9125 |

#### 6.1.2 United States (US)
| Target | Model | RMSE | MAE | sMAPE (%) | R2 |
|---|---|---:|---:|---:|---:|
| load_mw | GBM | 162.89 | 123.23 | 0.17 | 0.9996 |
| load_mw | LSTM | 4,767.14 | 3,835.32 | 5.24 | 0.6014 |
| load_mw | TCN | 3,850.49 | 2,877.31 | 3.82 | 0.7399 |
| wind_mw | GBM | 269.23 | 144.66 | 1.75 | 0.9982 |
| wind_mw | LSTM | 6,301.91 | 5,234.31 | 42.59 | 0.0627 |
| wind_mw | TCN | 7,187.54 | 5,930.74 | 47.65 | -0.2192 |
| solar_mw | GBM | 208.92 | 74.93 | 45.45 | 0.9962 |
| solar_mw | LSTM | 1,781.97 | 1,055.34 | 136.92 | 0.7135 |
| solar_mw | TCN | 1,743.66 | 965.08 | 131.50 | 0.7257 |

Interpretation: GBM is the strongest performer in the locked operational snapshots for both regions and all targets.

Thesis-level interpretation:
1. The GBM advantage is broad across load, wind, and solar in both regions for the locked evaluation artifacts.
2. Deep models remain analytically useful but are not operational leaders in this snapshot.
3. Because decisions are downstream of forecasts, model ranking should be interpreted together with optimization impact and not as an isolated leaderboard.

### 6.2 GBM Residual and Coverage Diagnostics (90% Intervals)
| Region | Target | Residual q10 | Residual q50 | Residual q90 | PICP (%) |
|---|---|---:|---:|---:|---:|
| DE | load_mw | -321.56 | -11.61 | 254.52 | 95.17 |
| DE | wind_mw | -293.49 | -42.68 | 221.95 | 88.15 |
| DE | solar_mw | -165.27 | 0.40 | 336.66 | 87.00 |
| US | load_mw | -240.08 | 1.00 | 223.77 | 87.39 |
| US | wind_mw | -146.14 | -3.96 | 122.64 | 80.50 |
| US | solar_mw | -105.40 | -0.14 | 186.94 | 90.47 |

Interpretation: coverage is close to nominal for some targets and lower for others, motivating region/target-specific uncertainty calibration tuning.

Coverage pattern notes:
1. DE load coverage exceeds nominal (conservative intervals in this slice).
2. US wind coverage is materially below nominal, indicating under-coverage risk.
3. Solar coverage differs by region and likely reflects different generation profiles and feature regimes.

### 6.2A Baselines and Comparator Mapping
| Result section/table | Numerator policy | Denominator/comparator policy | Source artifact |
|---|---|---|---|
| Decision Impact (Canonical Percent Outcomes) | B1 (deterministic LP, no uncertainty intervals) | B2 (grid-only, no battery) | `reports/impact_summary.csv`; `reports/eia930/impact_summary.csv` |
| Decision Impact (Absolute Values) | B1 (deterministic LP, no uncertainty intervals) | B2 (grid-only, no battery) | `reports/impact_summary.csv`; `reports/eia930/impact_summary.csv` |
| Stochastic Value Metrics (Canonical Run IDs) | robust vs deterministic stochastic formulations | deterministic/robust run-family comparator (not B2 denominator) | `reports/research_metrics_de.csv`; `reports/research_metrics_us.csv` |

### 6.2B Baseline Absolute Policy Outcomes (DE/US)
Source files:
- `reports/impact_comparison.json` (DE)
- `reports/eia930/impact_comparison.json` (US)

Table uses existing serialized artifacts only (`baseline` = B2, `naive` = B3, `optimized_forecast` = B1) and applies two-decimal display rounding.

| Region | Policy ID | Policy name | Cost (USD) | Carbon (kg) | Peak (MW) |
|---|---|---|---:|---:|---:|
| DE | B2 | Grid-only (no battery) | 154,773,462.72 | 1,461,641,243.28 | 52,165.00 |
| DE | B3 | Naive battery heuristic | 154,200,945.64 | 1,454,657,899.03 | 52,165.00 |
| DE | B1 | Deterministic LP (no uncertainty intervals) | 143,776,429.88 | 1,457,205,025.31 | 48,969.37 |
| US | B2 | Grid-only (no battery) | 461,364,890.00 | 3,907,803,618.32 | 74,741.00 |
| US | B3 | Naive battery heuristic | 462,821,035.03 | 3,911,194,390.28 | 70,929.00 |
| US | B1 | Deterministic LP (no uncertainty intervals) | 460,842,677.76 | 3,902,642,369.72 | 74,741.00 |

### 6.3 Decision Impact (Canonical Percent Outcomes)
| Region | Cost Savings | Carbon Reduction | Peak Shaving |
|---|---:|---:|---:|
| DE | **7.11%** | **0.30%** | **6.13%** |
| US | **0.11%** | **0.13%** | **0.00%** |

Source files:
- `reports/impact_summary.csv`
- `reports/eia930/impact_summary.csv`

All reported DE/US impact percentages are computed as B1 vs B2.

Decision-impact reading guidance:
1. Percent values are the canonical publication claims.
2. The absolute table below should be used to explain scale.
3. DE and US should not be treated as directly interchangeable market contexts.

### 6.4 Decision Impact (Absolute Values)
| Region | Baseline Cost (USD) | GridPulse Cost (USD) | Cost Delta (USD) | Baseline Carbon (kg) | GridPulse Carbon (kg) | Carbon Delta (kg) | Baseline Peak (MW) | GridPulse Peak (MW) | Peak Delta (MW) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DE | 154,773,462.72 | 143,776,429.88 | 10,997,032.84 | 1,461,641,243.28 | 1,457,205,025.31 | 4,436,217.97 | 52,165.00 | 48,969.37 | 3,195.63 |
| US | 461,364,890.00 | 460,842,677.76 | 522,212.24 | 3,907,803,618.32 | 3,902,642,369.72 | 5,161,248.60 | 74,741.00 | 74,741.00 | 0.00 |

Absolute impacts are critical for thesis writing because they show that a small percentage in a large system can still correspond to substantial absolute deltas.

### 6.5 Stochastic Value Metrics (Canonical Run IDs)
| Region | Run ID | Timestamp (UTC) | EVPI_robust | EVPI_deterministic | VSS | Robust Feasible | Solver Status |
|---|---|---|---:|---:|---:|---|---|
| DE | `20260217_165756` | 2026-02-17T16:57:56.736552 | 2.32 | -30.40 | 2,708.61 | True | summary |
| US | `20260217_182305` | 2026-02-17T18:23:06.287827 | 10,279,851.74 | 24,915,503.93 | 297,092.71 | True | summary |

Additional run-summary context:

| Region | Mean Dynamic Interval Width | Mean Base Interval Width | Mean Stressed Interval Width | FACI Scale Lower | FACI Scale Upper |
|---|---:|---:|---:|---:|---:|
| DE | 653.57 | 657.11 | not reported in DE schema | 0.0648 | 1.0000 |
| US | 649.20 | 415.60 | 1,233.49 | 0.1035 | 1.5965 |

US run summary also includes operational control fields (same run row):
- Operational grid cap (MW): 69,844.27
- Reserve SOC (MWh): 2,000.00
- Terminal SOC target (MWh): 4,450.43
- Load stress additive (MW): 1,257.02
- Stress interval multiplier: 1.90

Interpretation for writing:
1. Both runs are feasible and explicitly tied to run IDs and timestamps.
2. DE run shows modest EVPI magnitudes with positive VSS.
3. US run shows very large EVPI magnitudes and positive VSS, indicating strong sensitivity to uncertainty and scenario setup.
4. These values are run-scoped and should always be written with run IDs.

### 6.6 Robustness Perturbation Summary
Source: `reports/publication/tables/table6_robustness.csv`.

| Dataset | Perturbation (%) | Infeasible (%) | Mean Regret |
|---|---:|---:|---:|
| DE | 0.0 | 0.0 | 0.00 |
| DE | 5.0 | 0.0 | -1,509.47 |
| DE | 10.0 | 0.0 | -68,063.96 |
| DE | 20.0 | 0.0 | -183,810.02 |
| DE | 30.0 | 0.0 | -142,933.97 |
| US | 0.0 | 0.0 | 0.00 |
| US | 5.0 | 0.0 | 25,431.64 |
| US | 10.0 | 0.0 | -241,341.60 |
| US | 20.0 | 0.0 | -140,118.36 |
| US | 30.0 | 0.0 | 506,266.38 |

All listed perturbation points remain feasible in this summary table.

Regret sign caution:
1. Negative regret values can appear depending on comparator definitions and scenario realization.
2. Regret magnitude should be interpreted with consistent metric definitions from the source table, not redefined ad hoc in text.

### 6.7 Cross-Region Interpretation
1. Both regions show positive VSS under canonical runs, indicating stochastic/robust value exists in both settings.
2. Magnitude differs strongly by region, implying scenario design and market regime sensitivity.
3. DE shows stronger direct percent decision impact in the locked snapshot.
4. US shows smaller percent impact in cost/peak, but large stochastic value magnitudes in canonical run summary metrics.

This pattern supports a key thesis argument: forecast-to-dispatch systems should be evaluated as region-specific decision systems, not benchmarked only by aggregate forecast score transferability.

### 6.8 Copy-Ready Core Claims
Use these exact statements in publication sections to avoid drift:
1. DE cost savings: 7.11%; DE carbon reduction: 0.30%; DE peak shaving: 6.13%.
2. US cost savings: 0.11%; US carbon reduction: 0.13%; US peak shaving: 0.00%.
3. DE stochastic run: 20260217_165756 (EVPI_robust 2.32; EVPI_deterministic -30.40; VSS 2,708.61).
4. US stochastic run: 20260217_182305 (EVPI_robust 10,279,851.74; EVPI_deterministic 24,915,503.93; VSS 297,092.71).
5. Metric policy is dataset-scoped latest; this manuscript is not a single common-run freeze.

### 6.9 Negative and Neutral Findings (Do Not Omit)
To keep the thesis technically honest, the following points should remain explicit:
1. US peak shaving in the locked impact source is `0.00%`.
2. Several interval coverage values are below 90% nominal target.
3. Deep model families are not top-ranked in the locked dashboard snapshots.
4. Cross-region impact magnitude varies significantly despite shared architecture.

## 7. Metrics Source of Truth and Run IDs

### 7.1 Locked Sources
| Claim Group | Canonical Source | Selection |
|---|---|---|
| DE impact metrics | `reports/impact_summary.csv` | dataset-scoped latest |
| US impact metrics | `reports/eia930/impact_summary.csv` | dataset-scoped latest |
| DE stochastic metrics | `reports/research_metrics_de.csv` | `run_id = 20260217_165756`, `row_type = run_summary` |
| US stochastic metrics | `reports/research_metrics_us.csv` | `run_id = 20260217_182305`, `row_type = run_summary` |
| Dataset profile claims | `data/dashboard/de_stats.json`, `data/dashboard/us_stats.json`, `data/dashboard/manifest.json` | dashboard profile lock |

### 7.2 Rule for Any Future Metric Update
A metric can be changed in manuscript text only if all conditions are met:
1. Value is updated in `paper/metrics_manifest.json` with source path and run ID.
2. Claim row exists/updates in `paper/claim_matrix.csv`.
3. `scripts/validate_paper_claims.py` passes after edit.
4. Markdown and LaTeX versions retain matching core values.

This rule applies to percentages, absolute values, run IDs, timestamps, and any sentence that embeds quantitative evidence. Editorial convenience is not a valid reason to bypass the update chain.

### 7.3 Non-Policy Sources
Publication table exports and legacy markdown reports are secondary evidence only. They may be used for narrative context after explicit reconciliation, but not as primary numeric sources for locked claims.

### 7.4 Metrics Change-Control Checklist
Before approving any numeric edit:
1. Confirm source artifact path exists and is versioned.
2. Confirm row/field selection rule is unambiguous.
3. Regenerate display value from raw value using manifest rounding rules.
4. Verify the same metric is synchronized in abstract, results, and conclusion.
5. Re-run `scripts/validate_paper_claims.py` and archive pass output in revision notes.

## 8. Claim Traceability and Editorial Governance

### 8.1 Claim Status Definitions
- **Verified**: directly traceable to locked artifact(s).
- **Conflicting**: disagrees with canonical policy or other locked values.
- **Unsupported**: no in-repo evidence found.
- **Needs Citation**: plausible but requires external published source.

Claim statuses are operational controls. A status changes only when evidence or citation coverage changes; wording-only edits are insufficient.

### 8.2 Traceability Fields Required per Claim
Each claim row in `paper/claim_matrix.csv` should include:
1. `claim_id`
2. `status`
3. `category`
4. `claim_text`
5. `source_file`
6. `source_locator`
7. `run_id` when applicable
8. `timestamp_utc` when applicable
9. `rounding_rule`

### 8.3 Publication Rule
Publication-facing sections should retain only:
1. Verified claims.
2. External claims with explicit citations.

Conflicting/unsupported claims should be removed or rewritten before release export.

### 8.4 Editing Workflow
1. Edit `paper/PAPER_DRAFT.md` only.
2. Run claim validator.
3. Sync `paper/paper.tex` from markdown master rules.
4. Sync DOCX last and record any environment limitation.

### 8.5 Reviewer and Advisor Workflow
Recommended review loop for future revisions:
1. Reviewer flags text spans or claim IDs.
2. Author resolves each item against `claim_matrix` and manifest source.
3. Any unresolved numeric conflict blocks release.
4. Validator pass output is attached to the revision package.

## 9. Discussion

### 9.1 Strongly Supported Findings
1. The DE and US impact percentages above are directly reproducible from canonical impact CSVs.
2. The DE and US stochastic values above are directly reproducible from canonical run-summary rows.
3. Forecast model ranking in dashboard artifacts favors GBM for the locked snapshots.
4. Governance controls materially reduce manuscript drift risk.

The strongest supported thesis claim is the combination of measurable decision outcomes and explicit provenance enforcement. This manuscript treats those two properties as inseparable.

### 9.2 Why Cross-Region Results Differ
Likely drivers include:
1. Different data windows and distributions.
2. Different operational constraints and stress/control settings.
3. Different feature spaces and generation mix dynamics.
4. Different scale of system-level costs and uncertainty realizations.

This implies deployment should be region-tuned. A strategy that yields high percentage gains in one setting may remain valuable in another while showing smaller direct percentages.

### 9.3 Non-Overclaim Boundary
This manuscript does not claim universal superiority across all markets. It reports locked evidence for the stated DE/US windows and explicitly separates verified local findings from global claims requiring citation.

### 9.4 Engineering Significance
The main engineering contribution is integration plus governance:
- Integrated forecast-to-dispatch pipeline.
- Artifact-locked evidence policy.
- Automated cross-file consistency checks before publication.

### 9.5 Practical Operator Interpretation
Operationally, GridPulse behaves as a guarded decision-support system:
1. Healthy forecast and interval artifacts trigger optimization-led dispatch planning.
2. Infeasible or degraded states trigger explicit safe fallbacks.
3. Control-plane watchdog and BMS checks enforce dispatch safety before execution.

This framing links thesis evidence to practical operational deployment behavior.
Operationally, the stack is ready for software deployment pathways and controlled integration tests, but hardware field validation remains pending in the current evidence lock.

## 10. Threats to Validity

### 10.1 Internal Validity
Risk: mixed source families can reintroduce contradictory metrics.  
Mitigation: manifest lock + validator + claim matrix.

### 10.2 External Validity
Risk: findings may not generalize to other regions/time periods.  
Mitigation: keep claims region-scoped; extend datasets in future work.

### 10.3 Construct Validity
Risk: price/carbon proxies may not represent every market rule in detail.  
Mitigation: document proxy assumptions and avoid over-broad causal claims.

### 10.4 Conclusion Validity
Risk: stochastic value magnitude is sensitive to run/scenario specification.  
Mitigation: report explicit run IDs and discourage metric mixing across run families.

### 10.5 Residual Risk Prioritization
Priority order for next validity improvements:
1. Keep metric-source synchronization strict across markdown, LaTeX, and DOCX.
2. Improve interval calibration on under-covered targets.
3. Expand external validity with additional regions and time windows.
4. Close all citation gaps for non-repository claims.

## 11. Operational Safety and Failure Modes

### 11.1 Failure-Mode Table
| Failure Mode | Trigger | Detection | Mitigation | Fallback |
|---|---|---|---|---|
| Missing/stale feature data | upstream delay or schema break | `/ready`, data checks, missing file checks | block forecast/optimize calls until fresh data available | serve health warning, skip optimization |
| Forecast drift | distribution shift in live windows | KS/model drift checks in monitor pipeline | retraining decision logic | temporary conservative dispatch mode |
| Optimization infeasibility | extreme constraints or malformed intervals | solver status + feasibility flags | return safe infeasible payload, investigate constraints | grid-only baseline dispatch |
| Unsafe dispatch command | SOC/power constraint violation | BMS validation in control route | reject command with explicit error | keep current safe operating state |
| Control-plane degradation | missed heartbeat / watchdog timeout | watchdog islanding behavior | lock remote control | manual/local control only |
| API auth misuse | invalid scope/key | API security middleware | deny operation | read-only status endpoints |

### 11.2 Incident Response Sequence
1. Detect (health/monitoring/solver flags).
2. Contain (reject unsafe commands or switch to baseline).
3. Diagnose (artifact/log inspection with run IDs).
4. Recover (retrain, reconfigure, or rollback models).
5. Record (update governance artifacts and incident notes).

### 11.3 Governance Safeguards
1. Source-of-truth lock for manuscript claims.
2. Explicit banned legacy metrics in validator.
3. Claim matrix with status enforcement.

### 11.4 Config-Level Safety Snapshot
Selected defaults in locked configs:
1. KS drift threshold `p_value_threshold = 0.01`.
2. Model drift threshold `degradation_threshold = 0.15` on configured metric.
3. Retraining cadence `30` days with minimum new-data rule in config.
4. Deterministic objective weights: cost `1.0`, carbon `1.2`.
5. High unmet-load and curtailment penalties to discourage unsafe optimization shortcuts.

## 12. Limitations

### 12.1 Dataset and Coverage Limits
1. Evidence is restricted to the locked DE/US windows listed in Section 3.
2. Broader climate/market regimes remain outside this current lock.

These limits are intentional for traceability. Scope expansion should happen only with a corresponding manifest and claim-matrix update.

### 12.2 Modeling and Optimization Limits
1. Scenario and stress design choices influence stochastic magnitudes.
2. Proxy cost/carbon assumptions may differ from real settlement mechanisms.

Robust objective weighting and interval-construction choices can shift tradeoff behavior; therefore results should be interpreted as evidence for this configured system, not as universal constants.

### 12.3 Documentation Synchronization Limits
Programmatic DOCX sync is currently constrained in this shell environment:
1. system `python3` lacks `python-docx`.
2. the `.venv` Python launcher is not currently usable here.

### 12.4 Citation Limits
Some broader operational/societal statements require external literature or official source citation before publication.

Claims without internal artifact evidence should remain clearly marked as requiring citation until sources are added.

### 12.5 IoT Validation Boundary (Deployment-Readiness, Non-Field Evidence)
IoT-related validation reported in this manuscript is software-in-the-loop/API/streaming-smoke evidence, not completed hardware field commissioning evidence.
1. Streaming checks rely on replayed telemetry and broker-consumer pipeline execution.
2. DC3S live checks are API-level evaluations driven by simulated/replayed telemetry inputs.
3. No physical inverter/BMS edge commissioning logs or field-trial artifacts are claimed in this revision.

Therefore, deployment-readiness claims are limited to software stack correctness and observability, not hardware-operational certification.

## 13. Future Work

### 13.1 Near-Term (Next Revision Cycle)
1. Unify all report generators so dashboard and publication tables derive from one locked pipeline.
2. Add stricter section-level linting for unsupported claim phrases.
3. Improve uncertainty calibration for under-covered target/region combinations.

Near-term success condition: no unresolved claim conflicts and consistent core metrics across markdown and LaTeX outputs.

### 13.2 Mid-Term
1. Extend stochastic scenario design with explicit sensitivity studies.
2. Add more regions and tariff/carbon regimes for external-validity testing.
3. Improve model registry lineage and release metadata binding.

Mid-term success condition: cross-region comparisons with fully serialized scenario assumptions and reproducible run-level evidence.

### 13.3 Long-Term
1. Transition from artifact checks to signed evidence bundles.
2. Integrate richer operational economics and market constraints.
3. Deploy continuous publication-quality reporting with deterministic synchronization.

Long-term success condition: end-to-end signed provenance from data extraction to publication-ready manuscript artifacts.

## 14. Conclusion
GridPulse demonstrates a decision-grade ML system that connects forecasting, uncertainty, optimization, and governance. Under the dataset-scoped latest lock (February 17, 2026 artifacts), DE impact is 7.11% cost savings, 0.30% carbon reduction, and 6.13% peak shaving; US impact is 0.11% cost savings, 0.13% carbon reduction, and 0.00% peak shaving. Canonical stochastic values come from DE run `20260217_165756` and US run `20260217_182305`, with positive VSS in both regions (`2,708.61` and `297,092.71`, respectively). The thesis-level contribution is therefore both algorithmic and operational: measurable decision outcomes plus a reproducible governance framework that keeps claims stable across manuscript iterations.

The central practical message is that trustworthy decision systems require both technical performance and claim governance. GridPulse shows that forecast quality, uncertainty quantification, optimization feasibility, and publication traceability can be engineered as one coherent system.
Current IoT validation evidence in this manuscript is non-field and should be interpreted as software stack readiness rather than completed hardware-operational certification.

## 15. References

### 15.1 Internal Artifact References
- `paper/metrics_manifest.json`
- `paper/claim_matrix.csv`
- `paper/accuracy_audit.md`
- `paper/rewrite_pack.md`
- `paper/sync_rules.md`
- `scripts/validate_paper_claims.py`
- `frontend/src/app/api/chat/tool-executors.ts`
- `services/api/routers/monitor.py`
- `src/gridpulse/streaming/run_consumer.py`
- `src/gridpulse/streaming/consumer.py`
- `src/gridpulse/streaming/worker.py`
- `configs/dc3s.yaml`
- `src/gridpulse/dc3s/`
- `services/api/routers/dc3s.py`
- `data/dashboard/manifest.json`
- `data/dashboard/de_stats.json`
- `data/dashboard/us_stats.json`
- `data/dashboard/de_metrics.json`
- `data/dashboard/us_metrics.json`
- `reports/impact_summary.csv`
- `reports/eia930/impact_summary.csv`
- `reports/impact_comparison.json`
- `reports/eia930/impact_comparison.json`
- `reports/research_metrics_de.csv`
- `reports/research_metrics_us.csv`
- `reports/publication/tables/table6_robustness.csv`
- `docs/ARCHITECTURE.md`
- `docs/TRAINING_PIPELINE.md`
- `docs/EVALUATION.md`

### 15.2 External Method Literature (Maintain in Publication Export)
Keep explicit citations for:
1. Conformal prediction and adaptive conformal inference.
2. Robust/stochastic optimization for power systems.
3. Forecast model classes used (GBM, LSTM, TCN) where method background is discussed.

### 15.3 External Factual Claims Requiring Citation
Any non-repository factual statement (policy, workforce, market forecasts, environmental lifecycle values) must cite a reliable external source before publication.

## Appendix A. Replication Checklist
1. Validate manuscript claims against manifest and claim matrix:
```bash
python scripts/validate_paper_claims.py
```
2. Compile LaTeX manuscript:
```bash
cd paper
pdflatex paper.tex
```
3. Confirm core canonical strings exist in markdown and LaTeX:
- `7.11%`
- `0.11%`
- `20260217_165756`
- `20260217_182305`
- `2,708.61`
- `297,092.71`
4. Confirm no banned legacy percentages or placeholder tokens are present.
5. Confirm every publication numeric claim maps to a row in `paper/claim_matrix.csv`.

## Appendix B. Artifact Inventory
| Artifact | Purpose |
|---|---|
| `paper/PAPER_DRAFT.md` | Master manuscript source |
| `paper/paper.tex` | LaTeX export manuscript |
| `paper/metrics_manifest.json` | Canonical metric lock and validation regex |
| `paper/claim_matrix.csv` | Claim-level provenance and status |
| `paper/accuracy_audit.md` | Accuracy and conflict audit log |
| `paper/rewrite_pack.md` | Extended section guidance bank |
| `paper/sync_rules.md` | Markdown->LaTeX->DOCX synchronization contract |
| `scripts/validate_paper_claims.py` | Automated non-mutating consistency checker |
| `data/dashboard/*` | Dataset profiles and dashboard metrics |
| `reports/impact_summary.csv` | Canonical DE impact outcomes |
| `reports/eia930/impact_summary.csv` | Canonical US impact outcomes |
| `reports/research_metrics_de.csv` | DE stochastic runs |
| `reports/research_metrics_us.csv` | US stochastic runs |
| `reports/publication/tables/table6_robustness.csv` | Robustness perturbation summary |

## Appendix C. Section-by-Section Writing Bank (Copy-Ready)
Use this appendix when expanding or trimming sections.

### C1. Introduction
Required facts:
1. Forecast-only systems are insufficient for operations.
2. GridPulse decision loop and region scope.
3. Evidence-lock policy date and source concept.

Required figures/tables:
1. Architecture figure (`reports/figures/architecture.png` or `.svg`).

Optional details:
1. Production context and deployment pathways.

External citation required:
1. Industry-wide background claims about renewable volatility and operator workflows.

### C2. Data Assets and Scope
Required facts:
1. DE/US rows, date ranges, and feature counts from dashboard stats.
2. Distinction between columns and engineered features.
3. Scope boundary against legacy profile claims.

Required tables:
1. Dataset profile table.
2. Feature family composition table.

Optional details:
1. Target distribution table (mean/std/min/max/non-zero).

External citation required:
1. Dataset provider descriptions (OPSD, EIA-930) if discussed beyond internal files.

### C3. Methods
Required facts:
1. Model families and horizon setup.
2. Conformal interval logic and calibration target.
3. Deterministic and robust optimization intents and constraints.
4. Monitoring and safety pathways.

Required tables/figures:
1. Objective/constraint summary table.
2. Optional equation block for EVPI/VSS definitions.

Optional details:
1. Endpoint contracts and module-level implementation map.

External citation required:
1. Foundational method references (conformal, robust optimization, sequence models).

### C4. Results
Required facts:
1. Forecast metrics by target and model (DE and US).
2. Coverage and residual diagnostics.
3. Canonical DE/US impact percentages.
4. Canonical stochastic values and run IDs.
5. Robustness perturbation summary.

Required tables:
1. Forecast comparison tables.
2. Impact percent table.
3. Absolute impact table.
4. Stochastic run table.

Optional details:
1. Additional figures from `reports/figures/` (dispatch, SOC, tradeoff, interval width).

External citation required:
1. None for repository-derived numeric outcomes.

### C5. Governance Sections
Required facts:
1. Source-of-truth paths and run IDs.
2. Claim status model and publication rule.
3. Validation command and release gates.

Required tables:
1. Source-of-truth mapping table.
2. Claim-status summary snapshot.

Optional details:
1. Governance timeline and reviewer sign-off logs.

External citation required:
1. None unless claiming formal compliance against external standards.

### C6. Safety, Validity, and Limitations
Required facts:
1. Concrete failure modes and mitigations.
2. Internal/external/construct/conclusion validity boundaries.
3. Current environment limitations for DOCX sync.

Required table:
1. Failure-mode table (trigger, detection, mitigation, fallback).

Optional details:
1. SLA/SLO targets from production-readiness docs if included in thesis scope.

External citation required:
1. Policy/regulatory requirements if explicitly stated.

## Appendix D. Claims Requiring Citation Before Publication
Claims in these categories need explicit external source support or removal:
1. Global market-size forecasts and long-horizon policy projections.
2. Named-operator staffing/organizational metrics.
3. Environmental footprint values not derived by reproducible in-repo method.
4. Universal superiority claims beyond locked DE/US evidence.

## Appendix E. Synchronization and Release Notes
1. Manuscript authority remains `paper/PAPER_DRAFT.md`.
2. `paper/paper.tex` must preserve identical title, abstract core metrics, and conclusion core metrics.
3. DOCX synchronization is run last and must record tooling constraints in release notes when automation is unavailable.
4. Final release gate requires passing `scripts/validate_paper_claims.py` and no unresolved claim-status conflicts in publication sections.

## Appendix F. Full-Detail Authoring Blocks (Use/Trim as Needed)
This appendix gives longer thesis-writing blocks that can be pasted into chapter drafts and then edited for style.

### F1. Long-Form Intro Block
GridPulse is designed around a decision-first interpretation of machine learning in energy systems. Rather than ending at point prediction, the system carries uncertainty information into optimization, enforces dispatch feasibility under battery and grid constraints, and evaluates outcomes against explicit baselines. This architecture creates a measurable bridge between model behavior and operational value. The thesis therefore evaluates not only forecast quality, but the integrity of the full decision chain from feature engineering to published claims.

### F2. Long-Form Method Block
The forecasting layer trains LightGBM, LSTM, and TCN families under time-aware configurations with a 24-hour horizon and 168-hour lookback. Uncertainty is represented using conformal intervals with adaptive behavior in code-level implementations. Deterministic optimization uses a mixed-integer structure with cost, carbon, degradation, curtailment, unmet-load, and peak terms. Robust optimization uses a two-scenario epigraph formulation over lower and upper load bounds. Safety constraints are enforced in serving pathways before control actions can be accepted.

### F3. Long-Form Results Block
Under the locked artifact policy, DE impact is 7.11% cost savings, 0.30% carbon reduction, and 6.13% peak shaving. US impact is 0.11% cost savings, 0.13% carbon reduction, and 0.00% peak shaving. Canonical stochastic evidence is tied to DE run 20260217_165756 and US run 20260217_182305, both marked feasible in run-summary rows. Positive VSS is observed in both regions, with large magnitude differences across datasets. These differences motivate region-scoped interpretation rather than universal-effect claims.

### F4. Long-Form Governance Block
The manuscript is governed by a source-of-truth contract in which every publication-facing numeric claim must map to `paper/metrics_manifest.json` and have an entry in `paper/claim_matrix.csv`. Claim status categories distinguish verified evidence from conflicts and citation-risk statements. Validation is automated through `scripts/validate_paper_claims.py`, which checks required canonical strings and blocks known legacy contradictions. This governance layer is necessary because multiple historical report families coexist in the repository.

### F5. Long-Form Limitation Block
Evidence in this thesis is intentionally restricted to locked DE and US windows from dashboard profile artifacts. While this improves traceability, it limits direct external generalization. In addition, optimization outcomes depend on proxy price/carbon assumptions and scenario/interval settings that may differ across market designs. Therefore, results should be read as reproducible outcomes for the documented pipeline configuration and dataset scope, not as universal constants.

### F6. Long-Form Future Work Block
Future work should focus on unifying report-generation pathways, improving uncertainty calibration where coverage is below nominal, expanding region and tariff coverage, and strengthening release automation with signed evidence bundles. This progression preserves the thesis core principle: model advances and manuscript claims must evolve together under explicit provenance controls.

### F7. Keep/Trim Guidance for Final Submission
If page limits require reduction:
1. Keep Sections 6, 7, 8, and 14 unchanged.
2. Keep all run IDs and canonical percentages unchanged.
3. Trim descriptive implementation prose before trimming governance text.
4. Do not trim limitations that clarify scope and citation boundaries.

## Appendix G. Project-to-Paper Coverage Map
This appendix maps implemented project surfaces to manuscript coverage so review can distinguish fully documented areas from areas that are currently summarized.

Coverage status definitions:
1. **Full**: implementation paths and operational behavior are explicitly described in the paper.
2. **Partial**: core behavior is described, but implementation/runtime details are summarized.

| Project Surface | Primary Implementation Paths | Current Paper Coverage | Status | Notes |
|---|---|---|---|---|
| Forecast training orchestration, splits, tuning | `src/gridpulse/forecasting/train.py`, `configs/train_forecast.yaml`, `configs/train_forecast_eia930.yaml`, `scripts/train_dataset.py` | Sections 3.7, 4.1 | Full | Runtime control flags (`--tune`, `--no-tune`, `--ensemble`, `--max-seeds`, `--n-trials`, `--top-pct`), serialization of tuning metadata, and ensemble member persistence are now explicitly documented. |
| Forecast inference and bundle resolution | `services/api/routers/forecast.py`, `src/gridpulse/forecasting/predict.py` | Sections 2.3, 4.1.4 | Full | Explicit target-path resolution, ordered fallback search, ensemble averaging behavior, quantile serving order, and missing-target signaling are now documented. |
| Streaming ingestion and persistence | `src/gridpulse/streaming/run_consumer.py`, `src/gridpulse/streaming/consumer.py`, `src/gridpulse/streaming/worker.py`, `configs/streaming.yaml` | Sections 2.7, 12.5 | Full | CLI path, config load path, and DuckDB table contract are explicitly documented. |
| DC3S uncertainty shield and audit chain | `src/gridpulse/dc3s/`, `services/api/routers/dc3s.py`, `configs/dc3s.yaml` | Sections 2.3, 2.7, 4.5, 12.5 | Full | Endpoint contracts, implemented inflation law, shield modes, hash chain, and state/audit behavior are documented. |
| Monitoring and retraining | `services/api/routers/monitor.py`, `src/gridpulse/monitoring/`, `configs/monitoring.yaml` | Section 4.6 | Full | Drift thresholds, cadence, and decision logic are explicitly documented. |
| Safety and control-plane enforcement | `src/gridpulse/safety/`, `services/api/main.py` (`/control/dispatch`) | Sections 4.7, 11 | Full | Authorization, watchdog, and BMS gate sequence is documented. |
| Frontend operator workflow and live DC3S UX | `frontend/src/app/(dashboard)/page.tsx`, `frontend/src/components/dashboard/DC3SLiveCard.tsx`, `frontend/src/lib/api/dc3s-client.ts`, `frontend/src/app/api/dc3s/audit/[commandId]/route.ts` | Sections 2.2, 2.7 | Full | Dashboard polling default, operator-selectable auto-refresh cadence, manual refresh, command-id-linked audit quick link, and proxy/error behavior are explicitly documented. |
| Claim governance and release gates | `paper/metrics_manifest.json`, `paper/claim_matrix.csv`, `scripts/validate_paper_claims.py` | Sections 5, 7, 8 | Full | Source lock, reconciliation, and publication gating are explicitly documented. |

### G1. Full-Coverage Update (Completed)
This revision closes the previously partial surfaces:
1. Section 4.1 now documents runtime training controls and their reproducibility implications.
2. Section 4.1.4 now documents explicit fallback/ensemble/quantile inference resolution mechanics.
3. Section 2.7 now documents frontend live-ops controls, auto-refresh cadence selection, and command-linked audit navigation.
4. Baseline policy definitions and impact-table numerator/denominator mapping are now explicit for B1/B2/B3.
