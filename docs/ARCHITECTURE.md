# GridPulse System Architecture

## Overview

GridPulse implements a **Level-4 Decision System** architecture where predictions feed directly into an optimization engine, and outcomes are measured against baselines. This document describes the complete system flow, component responsibilities, and data contracts between layers.

## Architectural Principles

1. **Decision-Grade Output**: Every forecast produces actionable dispatch commands
2. **Uncertainty-Aware**: Conformal prediction intervals propagate through optimization
3. **Baseline-Compared**: All impact metrics are computed against grid-only and naive policies
4. **Reproducible**: Deterministic seeds, version locks, and run manifests
5. **Region-Agnostic**: Same pipeline processes OPSD (Germany) and EIA-930 (US) data

---

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  OPSD Germany          EIA-930 USA           Open-Meteo Weather             │
│  • Load (MW)           • MISO Demand         • Temperature                  │
│  • Wind (MW)           • Generation Mix      • Wind Speed                   │
│  • Solar (MW)          • Interchange         • Cloud Cover                  │
│  • Price (€/MWh)       • Balancing Area      • Solar Radiation              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE (Layer 1)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Ingestion & Validation                                                   │
│     • Schema enforcement (required columns, dtypes)                          │
│     • Timestamp alignment to hourly cadence                                  │
│     • Missing value imputation (forward-fill, interpolation)                 │
│                                                                              │
│  2. Feature Engineering                                                      │
│     • Temporal: hour, day_of_week, month, is_weekend, is_holiday            │
│     • Lags: t-1, t-2, t-24, t-48, t-168 (1 week)                            │
│     • Rolling: mean_24h, std_24h, mean_168h, std_168h                       │
│     • Cyclical: sin/cos encoding for hour and day_of_week                   │
│     • Weather: temperature, humidity, wind_speed (if available)              │
│                                                                              │
│  3. Time-Series Splits                                                       │
│     • Train: 70% (chronological)                                             │
│     • Validation: 15%                                                        │
│     • Test: 15%                                                              │
│     • Gap: 24h between splits to prevent leakage                             │
│                                                                              │
│  Output: data/processed/features.parquet, data/processed/splits/             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FORECASTING ENGINE (Layer 2)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Model Types:                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   LightGBM      │  │     LSTM        │  │      TCN        │              │
│  │   (Primary)     │  │   (Deep Seq)    │  │   (Deep Conv)   │              │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤              │
│  │ • 1000 trees    │  │ • 3 layers      │  │ • 4 channels    │              │
│  │ • lr=0.03       │  │ • 256 hidden    │  │ • kernel=5      │              │
│  │ • depth=12      │  │ • dropout=0.3   │  │ • dropout=0.3   │              │
│  │ • leaves=256    │  │ • 100 epochs    │  │ • 100 epochs    │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
│  Targets: load_mw, wind_mw, solar_mw, price_eur_mwh                         │
│  Horizon: 24 hours                                                           │
│  Metrics: RMSE, MAE, sMAPE, R², Daylight-MAPE (solar)                       │
│                                                                              │
│  Output: artifacts/models/*.txt (GBM), artifacts/models/*.pt (DL)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      UNCERTAINTY LAYER (Layer 3)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Conformal Prediction (ICP + FACI Adaptation):                               │
│  • Calibration: Compute nonconformity scores on validation set               │
│  • Prediction: Generate [lower, upper] bounds at α=0.10 (90% coverage)      │
│  • Adaptation: FACI updates interval width based on recent coverage          │
│                                                                              │
│  Per-Horizon Metrics:                                                        │
│  • PICP: Coverage probability per forecast hour (h1...h24)                   │
│  • MPIW: Mean interval width per forecast hour                               │
│                                                                              │
│  Output: artifacts/uncertainty/*_conformal.json                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ANOMALY DETECTION (Layer 4)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Detection Methods:                                                          │
│  • Residual Z-Score: |residual| > 3σ triggers alert                         │
│  • Isolation Forest: Multi-feature outlier detection                         │
│  • Cadence Check: Missing or delayed data points                             │
│                                                                              │
│  Actions:                                                                    │
│  • Flag unreliable intervals for operator review                             │
│  • Exclude anomalous periods from optimization input                         │
│  • Log events to reports/anomaly_report.md                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZATION ENGINE (Layer 5)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Robust Dispatch (DRO via Pyomo + HiGHS):                                    │
│                                                                              │
│  Decision Variables:                                                         │
│  • P_ch[t]: Battery charge power (MW)                                        │
│  • P_dis[t]: Battery discharge power (MW)                                    │
│  • G[s,t]: Grid import per scenario (MW)                                     │
│  • SoC[s,t]: State of charge per scenario (MWh)                              │
│  • z: Epigraph variable for worst-case cost                                  │
│                                                                              │
│  Objective: min z + λ_deg * Σ(P_ch + P_dis)                                  │
│                                                                              │
│  Constraints:                                                                │
│  • Energy balance: P_dis - P_ch + G >= Load - Renewables                    │
│  • SoC dynamics: SoC[t+1] = SoC[t] + η_ch*P_ch - P_dis/η_dis                │
│  • Bounds: SoC_min ≤ SoC ≤ SoC_max, 0 ≤ G ≤ G_max                           │
│  • Scenario coupling: z ≥ cost(lower), z ≥ cost(upper)                       │
│                                                                              │
│  Configuration: configs/optimization.yaml                                    │
│  Output: artifacts/dispatch_plans/*.json                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION LAYER (Layer 6)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Baseline Comparison:                                                        │
│  • Grid-Only: No battery, all demand from grid                               │
│  • Naive Battery: Charge overnight, discharge evening peak                   │
│                                                                              │
│  Impact Metrics:                                                             │
│  • Cost Savings %: (baseline_cost - optimized_cost) / baseline_cost         │
│  • Carbon Reduction %: Same formula for carbon emissions                     │
│  • Peak Shaving %: Reduction in max grid import                              │
│                                                                              │
│  Stochastic Metrics:                                                         │
│  • EVPI: Value of perfect load information                                   │
│  • VSS: Value of robust vs deterministic solution                            │
│                                                                              │
│  Output: reports/impact_summary.csv, reports/frozen_metrics_snapshot.json   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MONITORING LAYER (Layer 7)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Data Drift Detection:                                                       │
│  • Kolmogorov-Smirnov test per feature (p < 0.05 triggers alert)            │
│  • Population Stability Index (PSI)                                          │
│                                                                              │
│  Model Drift Detection:                                                      │
│  • Rolling RMSE comparison vs calibration baseline                           │
│  • Alert if degradation > 10%                                                │
│                                                                              │
│  Retraining Triggers:                                                        │
│  • Scheduled: Weekly full retrain                                            │
│  • Drift-based: Automatic if KS p-value < threshold                          │
│                                                                              │
│  Output: reports/monitoring_summary.json, reports/monitoring_report.md      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SERVING LAYER (Layer 8)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  FastAPI Backend (services/api/):                                            │
│  • GET  /health          - Liveness check                                    │
│  • GET  /ready           - Readiness with model count                        │
│  • GET  /metrics         - Prometheus metrics                                │
│  • POST /forecast        - Generate forecast for region/target               │
│  • POST /optimize        - Run dispatch optimization                         │
│  • GET  /monitor/drift   - Current drift status                              │
│  • GET  /anomaly/recent  - Recent anomaly events                             │
│                                                                              │
│  Next.js 15 Dashboard (frontend/):                                           │
│  • Overview: KPIs, dispatch chart, model registry                            │
│  • Forecasting: Forecast vs actual, model comparison                         │
│  • Optimization: Dispatch plan, battery SOC, cost impact                     │
│  • Carbon: Emissions breakdown, baseline comparison                          │
│  • Anomalies: Z-score timeline, event log                                    │
│  • Monitoring: Drift metrics, active model versions                          │
│  • Reports: Formal evaluation, publication figures                           │
│  • Data Explorer: Dataset statistics, time series                            │
│                                                                              │
│  Region Toggle: DE / US switch in navigation bar                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Artifacts

| Directory | Contents |
|-----------|----------|
| `configs/` | YAML configuration for training, optimization, monitoring |
| `data/processed/` | Feature-engineered Parquet files |
| `data/processed/splits/` | Train/val/test splits |
| `data/dashboard/` | Pre-computed JSON for frontend |
| `artifacts/models/` | Trained model files (GBM .txt, DL .pt) |
| `artifacts/uncertainty/` | Conformal calibration parameters |
| `artifacts/dispatch_plans/` | Optimization output schedules |
| `reports/` | Evaluation reports, figures, model cards |
| `reports/figures/` | Publication-ready plots (300 DPI PNG) |

---

## Data Contracts

### Feature Store Schema
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
... (98 features for DE, 118 for US)
```

### Forecast Output Schema
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

### Dispatch Plan Schema
```json
{
  "timestamp": "2026-02-17T12:00:00Z",
  "battery_charge_mw": [0, 0, 25, ...],
  "battery_discharge_mw": [15, 20, 0, ...],
  "grid_import_mw": [42100, 43200, ...],
  "soc_mwh": [50, 35, 15, 40, ...],
  "total_cost_eur": 128450.32,
  "feasible": true
}
```

---

## CPSBench-IoT Benchmark Block

GridPulse now includes a CPSBench-IoT harness under `src/gridpulse/cpsbench_iot/` for repeatable stress-testing of controller behavior under telemetry faults and drift.

### CPSBench Data Flow

1. `scenarios.generate_episode(...)` creates deterministic `x_true`, faulted `x_obs`, and `event_log`.
2. Baseline adapters execute:
   - deterministic LP dispatch,
   - robust fixed-interval dispatch,
   - naive safe-clip battery rule,
   - DC3S-wrapped controller.
3. `metrics.compute_all_metrics(...)` computes forecast, control, and trace metrics.
4. `runner.run_suite(...)` writes publication outputs to `reports/publication/`.

### CPSBench Publication Artifacts

- `reports/publication/dc3s_main_table.csv`
- `reports/publication/dc3s_fault_breakdown.csv`
- `reports/publication/calibration_plot.png`
- `reports/publication/violation_vs_cost_curve.png`
- `reports/publication/dc3s_run_summary.json`

Determinism guarantees:

- scenario generator uses local `numpy.random.Generator(seed)`,
- outputs are sorted by `(scenario, seed, controller)`,
- CSV float formatting is fixed to `%.6f`.

---

## IoT Closed-Loop Block

The closed-loop IoT validation path is additive and runs through in-process FastAPI endpoints.

### Router Surface (`/iot/*`)

- `POST /iot/telemetry`: persists telemetry, computes reliability weight `w_t`, updates device state.
- `GET /iot/command/next`: dequeues (or peeks) queued command; returns `status=hold` when timeout hold is active.
- `POST /iot/ack`: persists ACK/NACK linked to command/certificate.
- `GET /iot/state`: returns latest telemetry + last command + last ACK.
- `GET /iot/audit/{command_id}`: proxies certificate retrieval from DC3S audit store.
- `POST /iot/control/reset-hold`: clears timeout hold state for a device.

All `/iot/*` endpoints are API-key scoped (`read`/`write`) via `X-GridPulse-Key`.

### IoT Persistence

Default DuckDB file: `data/audit/iot_loop.duckdb`

Tables:

- `iot_telemetry`
- `iot_command_queue`
- `iot_ack`
- `iot_device_state`

Queue lifecycle extensions:

- queued commands include `expires_at` (TTL, default 30 seconds),
- stale queued/dispatched commands are marked `timeout`,
- timeout activates per-device hold (`hold_active`, `hold_reason`, `hold_since_utc`) in state table.

### Closed-Loop Simulation Path

Implemented in `iot/simulator/run_closed_loop.py`:

`telemetry -> /iot/telemetry -> /dc3s/step(enqueue_iot=true) -> /iot/command/next -> apply -> /iot/ack`

`/dc3s/step` now supports additive queue fields:

- request: `enqueue_iot`, `queue_ttl_seconds`
- response: `queued`, `queue_status`

This path validates command traceability, safety gating, and certificate completeness without changing existing production endpoint contracts.
