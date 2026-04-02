# ORIUS / DC³S — System Architecture

> **Version**: March 2026 · Generated diagram: `reports/figures/architecture.png`
>
> **Scope note**: this document describes the energy-management instantiation of ORIUS. The canonical domain-agnostic framing now lives in `docs/UNIVERSAL_KERNEL_ARCHITECTURE.md`; use this file when you need the shared thesis-facing kernel, benchmark, and governance architecture.

## Overview

ORIUS implements a **Level-4 Decision System** architecture where forecasts feed directly into a distributionally-robust optimization engine and a runtime safety shield (DC³S) governs every dispatch action.  Outcomes are measured against baselines and recorded in auditable, release-scoped artifacts.  This document describes the nine-layer system, component responsibilities, data contracts, and cross-layer flows.

## Architectural Principles

1. **Decision-grade output** — every forecast produces actionable dispatch commands
2. **Uncertainty-aware** — CQR conformal intervals propagate through DRO optimization and DC³S inflation
3. **Safety-shielded** — DC³S scores telemetry reliability at each step and conditionally widens the uncertainty set, repairs infeasible actions, and emits per-step certificates
4. **Baseline-compared** — all impact metrics are computed against grid-only and naive battery policies
5. **Reproducible** — deterministic seeds, version locks, run manifests, and release-scoped governance
6. **Region-agnostic** — same pipeline processes OPSD (Germany) and EIA-930 (US, 3 BAs)
7. **Layered defense** — inspired by Sha's simplex architecture: optimizer (complex) → projection (simple safety net) → guarantee checks (deterministic guard)

---

## Layer Map

| Layer | Name | Key modules | Primary output |
|:-----:|------|-------------|----------------|
| L1 | Data Sources & Ingestion | `data_pipeline/`, `pipeline/` | `data/processed/*.parquet` |
| L2 | Feature Engineering & Splitting | `pipeline/`, `forecasting/datasets.py` | `features.parquet`, `splits/` |
| L3 | Forecasting Engine | `forecasting/` (6 families) | `artifacts/models/*.pkl, *.pt` |
| L4 | Uncertainty & Calibration | `forecasting/uncertainty/`, `dc3s/calibration.py`, `dc3s/rac_cert.py` | `artifacts/uncertainty/*.json` |
| L5 | Optimization & Dispatch | `optimizer/` | `artifacts/dispatch_plans/*.json` |
| L6 | DC³S Safety Shield | `dc3s/` | Certificates, safe dispatch actions |
| L7 | IoT · Edge · Streaming | `iot/`, `streaming/`, `cpsbench_iot/` | Validated events, ACK/NACK |
| L8 | Monitoring · Anomaly · Governance | `monitoring/`, `anomaly/`, `registry/` | `reports/monitoring_*.json`, release manifests |
| L9 | Serving (API + Dashboard) | `services/api/`, `frontend/` | REST endpoints, operator UI |

---

## L1 — Data Sources & Ingestion

### Sources

| Source | Granularity | Variables |
|--------|:-----------:|-----------|
| OPSD Germany | hourly | load, wind, solar, price (€/MWh) |
| EIA-930 USA | hourly | demand, generation mix (MISO · ERCOT · PJM) |
| Open-Meteo | hourly | temperature, wind speed, cloud cover, solar radiation |
| Carbon factors | static | DE-avg, MISO, ERCOT, PJM (kgCO₂/MWh) |

### Validation gates

- Schema enforcement: required columns and dtypes
- Timestamp alignment to hourly cadence
- Range checks (non-negative load, physical bounds)
- Missing-value imputation: forward-fill, then linear interpolation

**Output**: `data/processed/*.parquet`

---

## L2 — Feature Engineering & Splitting

### Feature families (98 DE / 118 US)

| Family | Examples |
|--------|----------|
| Temporal | `hour`, `day_of_week`, `month`, `is_weekend`, `is_holiday` |
| Lags | t−1, t−2, t−24, t−48, t−168 |
| Rolling statistics | `mean_24h`, `std_24h`, `mean_168h`, `std_168h` |
| Cyclical encoding | `sin_hour`, `cos_hour`, `sin_dow`, `cos_dow` |
| Weather | temperature, humidity, wind speed, solar irradiance |

### Time-series split

| Partition | Share | Note |
|-----------|:-----:|------|
| Train | 70% | chronological |
| Validation | 15% | used for conformal calibration |
| Test | 15% | held-out evaluation |
| Gap | 24 h | between each split to prevent leakage |

**Output**: `data/processed/features.parquet`, `data/processed/splits/`

---

## L3 — Forecasting Engine

Six model families, three targets (`load_mw`, `wind_mw`, `solar_mw`), 24-h horizon.

| Model | Type | Key hyperparameters | File format |
|-------|------|---------------------|:-----------:|
| **LightGBM** [PROD] | Gradient-boosted trees | 1 000 trees · depth 12 · lr 0.03 · 256 leaves | `.pkl` |
| LSTM | Recurrent (deep seq) | 3 layers · 256 hidden · dropout 0.3 · 100 epochs | `.pt` |
| TCN | Temporal CNN | 4 channels · kernel 5 · dropout 0.3 · 100 epochs | `.pt` |
| N-BEATS | Block-structured MLP | stacks · blocks · lookback 168 | `.pt` |
| TFT | Attention + gating | variable selection · multi-horizon | `.pt` |
| PatchTST | Patch-based Transformer | patch length · heads · channel-independent | `.pt` |

- **Walk-forward evaluation** with sMAPE, RMSE, MAE, R², daylight-MAPE (solar).
- Only the production-candidate model (GBM) is calibrated and enters the dispatch pipeline.
- Model registry: `src/orius/registry/model_store.py` — `promote()` performs atomic staging → production copy.

**Output**: `artifacts/models/`

---

## L4 — Uncertainty Quantification & Calibration

### Pipeline

1. **CQR calibration** — compute nonconformity scores on the validation set at α = 0.10 (90% nominal coverage)
2. **FACI online adaptation** — adjust interval width based on recent realised coverage
3. **Per-horizon metrics** — PICP and MPIW for each forecast hour h₁ … h₂₄
4. **RAC-Cert inflation** — map telemetry reliability score to monotone interval widening (conditional conservatism); bridges measurement quality → uncertainty set
5. **Mondrian binning** — group-conditional coverage audit across reliability quantile bins

### Theoretical grounding

- **Theorem (RAC coverage)**: monotone inflation preserves marginal coverage
- **Theorem (conditional coverage)**: Mondrian partitioning yields group-conditional guarantees
- **Theorem (safety margin)**: safety-margin monotonicity under inflation
- **Corollary (zero violation)**: sufficiency condition for zero constraint violations

**Output**: `artifacts/uncertainty/*_conformal.json`

---

## L5 — Optimization Engine (DRO via Pyomo + HiGHS)

### Formulation

| Component | Detail |
|-----------|--------|
| Decision variables | P_ch[t], P_dis[t], G[s,t], SoC[s,t], z (epigraph) |
| Objective | min z + λ_deg Σ(P_ch + P_dis) |
| Energy balance | P_dis − P_ch + G ≥ Load − Renewables |
| SoC dynamics | SoC[t+1] = SoC[t] + η_ch · P_ch − P_dis / η_dis |
| Bounds | SoC_min ≤ SoC ≤ SoC_max, 0 ≤ G ≤ G_max |
| Scenario coupling | z ≥ cost(lower), z ≥ cost(upper) |

### Baselines

- **Grid-only**: no battery, all demand from grid
- **Naive battery**: charge overnight, discharge during evening peak

### Impact metrics

- Cost savings %, carbon reduction %, peak shaving %
- EVPI (value of perfect information), VSS (value of stochastic solution)

**Output**: `artifacts/dispatch_plans/*.json`, `reports/impact_summary.csv`

---

## L6 — DC³S Safety Shield (online dispatch loop)

The core contribution.  Runs at each dispatch step in < 0.04 ms P95.

### Step pipeline

```
observe → quality-score → drift-detect → RAC-Cert inflate
       → robust solve → action repair → post-check → certify
```

| Stage | Module | Purpose |
|-------|--------|---------|
| Observe | `dc3s/state.py` | Ingest telemetry: load, renewables, SoC |
| Quality score | `dc3s/quality.py` | Detect missing, stale, spike, reorder events → reliability weight w_t |
| Drift detect | `dc3s/drift.py` | KS stat + EWM online update → drift risk flag |
| RAC-Cert inflate | `dc3s/rac_cert.py` | Reliability → interval width; conditional conservatism |
| Robust solve | `optimizer/robust_dispatch.py` | DRO against widened uncertainty set |
| Action repair | `dc3s/shield.py` | Project infeasible actions onto safe set (SoC, rate bounds) |
| Post-check | `dc3s/guarantee_checks.py` | SoC feasibility, charge-rate limits, FTIT tube containment |
| Certify | `dc3s/certificate.py` | Hash-chained per-step certificate with margins and reasons |

### Certificate schema

Each step emits an auditable certificate containing:
- `certificate_id`, `command_id`, `timestamp`
- `reliability_weight`, `drift_flag`, `inflation_factor`
- `proposed_action`, `safe_action`, `was_repaired`
- `soc_margin_mwh`, `rate_margin_mw`
- `guarantee_pass` (boolean), `reasons[]`
- `hash` (SHA-256 chain link)

---

## L7 — IoT · Edge Agent · Streaming

### Edge agent (`iot/edge_agent/`)

- Shadow mode: `shadow_mode=true`, `applied=false` — logs recommendations without actuating
- Device contract (`iot/DEVICE_CONTRACT.md`): cadence 1 event/h ± 120 s, TTL 30 s, ACK/NACK semantics
- Authentication: `X-ORIUS-Key` header with read/write scopes

### Streaming consumer (`src/orius/streaming/`)

| Component | Purpose |
|-----------|---------|
| Kafka/Redpanda consumer | Ingest JSON telemetry events |
| Pydantic schema | `OPSDTelemetryEvent` validation |
| Temporal + range checks | Cadence enforcement, dropout detection, delta outlier checks |
| Checkpoint | Exactly-once semantics every 200 messages |
| Sink | DuckDB or Parquet, validated and time-ordered |

### CPSBench-IoT (`src/orius/cpsbench_iot/`)

Repeatable stress-testing under five fault scenarios × five seeds:
1. `scenarios.generate_episode()` → deterministic `x_true`, faulted `x_obs`, `event_log`
2. Four baseline adapters: deterministic LP, robust fixed-interval, naive safe-clip, DC³S-wrapped
3. `metrics.compute_all_metrics()` → forecast, control, and trace metrics
4. Publication artifacts: `dc3s_main_table.csv`, `dc3s_fault_breakdown.csv`, `calibration_plot.png`

### IoT router surface (`/iot/*`)

| Endpoint | Method | Purpose |
|----------|:------:|---------|
| `/iot/telemetry` | POST | Persist telemetry, compute reliability w_t |
| `/iot/command/next` | GET | Dequeue queued command (or hold) |
| `/iot/ack` | POST | Persist ACK/NACK linked to command/certificate |
| `/iot/state` | GET | Latest telemetry + command + ACK |
| `/iot/audit/{id}` | GET | Certificate retrieval from DC³S audit store |
| `/iot/control/reset-hold` | POST | Clear timeout hold for a device |

### Closed-loop simulation

```
telemetry → /iot/telemetry → /dc3s/step(enqueue_iot=true)
          → /iot/command/next → apply → /iot/ack
```

Validates command traceability, safety gating, and certificate completeness without changing production endpoint contracts.

---

## L8 — Monitoring · Anomaly Detection · Governance

### Drift detection

| Method | Trigger |
|--------|---------|
| Kolmogorov-Smirnov per feature | p < 0.05 |
| Population Stability Index (PSI) | PSI > threshold |
| Rolling RMSE vs calibration baseline | degradation > 10% |

### DC³S health

- Intervention rate, low-reliability rate, drift-flag rate, inflation P95
- Sustained-window triggering (persisted in `reports/monitoring_state.json`)

### Anomaly detection

- Residual z-score: |residual| > 3σ → alert
- Isolation forest: multi-feature outlier detection
- Cadence check: missing or delayed data points

### Retraining triggers

| Type | Condition |
|------|-----------|
| Scheduled | Weekly full retrain |
| Drift-based | KS p-value < threshold |
| DC³S-based | Intervention spikes, reliability degradation, drift persistence |

### Governance

- Release manifests with model hashes and artifact checksums
- Deployment evidence map: code path → governed artifact → manuscript claim
- Four validation gates: artifact completeness, metric bounds, evidence map, promotion review

**Output**: `reports/monitoring_summary.json`, `reports/monitoring_report.md`

---

## L9 — Serving Layer (API + Dashboard)

### FastAPI backend (`services/api/`)

| Endpoint | Method | Purpose |
|----------|:------:|---------|
| `/health` | GET | Liveness check |
| `/ready` | GET | Readiness with model count |
| `/metrics` | GET | Prometheus metrics |
| `/forecast` | POST | Generate forecast for region/target |
| `/optimize` | POST | Run dispatch optimization |
| `/dc3s/step` | POST | Safety-gated dispatch + certificate issuance |
| `/monitor` | GET | Drift + DC³S health + retrain decision |
| `/monitor/dc3s` | GET | DC³S health-only view |
| `/anomaly/recent` | GET | Recent anomaly events |

### Next.js 15 dashboard (`frontend/`)

Eight pages: Overview · Forecasting · Optimization · Carbon · Anomalies · Monitoring · Reports · Data Explorer.  Region toggle (DE / US) in navigation bar.

### Infrastructure

| Mode | Components |
|------|-----------|
| `docker-compose.yml` | FastAPI + PostgreSQL + Redis |
| `docker-compose.streaming.yml` | + Kafka/Redpanda |
| `docker-compose.full.yml` | + Prometheus + Grafana + frontend |
| `deploy/k8s/` | K8s manifests with health checks, ConfigMap |
| `deploy/aws/` | ECS task definitions, EventBridge retrain |
| `deploy/systemd/` | Bare-metal units with retrain timers |

---

## Key Artifacts

| Directory | Contents |
|-----------|----------|
| `configs/` | YAML configuration for training, optimisation, monitoring |
| `data/processed/` | Feature-engineered Parquet files |
| `data/processed/splits/` | Train / validation / test splits |
| `paper/assets/data/` | Tracked manuscript-facing data identity artifacts |
| `data/audit/` | DuckDB audit stores (IoT loop, certificates) |
| `artifacts/models/` | Trained model files (GBM `.pkl`, DL `.pt`) |
| `artifacts/uncertainty/` | Conformal calibration parameters |
| `artifacts/dispatch_plans/` | Optimisation output schedules |
| `artifacts/registry/` | Promoted model copies |
| `reports/` | Evaluation reports, figures, model cards |
| `reports/figures/` | Publication-ready plots (300 DPI PNG) |
| `reports/publication/` | Governed deployment evidence artifacts |

Frontend-facing report and dashboard data is served from backend research
endpoints backed by tracked publication artifacts. Old `data/dashboard/`
snapshots remain legacy cache material only.

---

## Data Contracts

### Feature store schema
```
timestamp: datetime64[ns]
load_mw: float64
wind_mw: float64
solar_mw: float64
price_eur_mwh: float64
hour: int8
day_of_week: int8
is_weekend: int8
load_lag_1: float64
load_lag_24: float64
load_rolling_mean_24h: float64
… (98 features for DE, 118 for US)
```

### Forecast output schema
```json
{
  "timestamp": "2026-02-17T12:00:00Z",
  "target": "load_mw",
  "point_forecast": 45230.5,
  "lower_bound": 44102.3,
  "upper_bound": 46358.7,
  "confidence": 0.90,
  "model": "gbm",
  "region": "DE"
}
```

### Dispatch plan schema
```json
{
  "timestamp": "2026-02-17T12:00:00Z",
  "battery_charge_mw": [0, 0, 25],
  "battery_discharge_mw": [15, 20, 0],
  "grid_import_mw": [42100, 43200],
  "soc_mwh": [50, 35, 15, 40],
  "total_cost_eur": 128450.32,
  "feasible": true
}
```

---

## Regenerating the architecture diagram

```bash
.venv/bin/python3 scripts/generate_architecture_diagram.py
```

Writes:
- `reports/figures/architecture.png` + `.svg`
- `paper/assets/figures/fig01_architecture.png` + `.svg`
