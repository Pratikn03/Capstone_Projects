# GridPulse: Autonomous Energy Intelligence & Optimization Platform  
*A decision‑grade system that turns energy forecasts into feasible, measurable dispatch actions.*

## 1) Executive Summary  
GridPulse is an end‑to‑end energy intelligence platform built to address a core weakness in modern grid operations: **forecasting alone does not create actionable decisions**. As renewable penetration grows, grid variability increases and operators must continuously choose how to balance demand, renewables, and storage under both **cost** and **carbon** constraints. GridPulse combines **time‑series forecasting**, **anomaly detection**, and **optimization** into a single pipeline that converts raw data into **dispatch plans** for battery energy storage systems (BESS).

Unlike traditional “predict‑only” pipelines, GridPulse closes the loop. It ingests power system data (and optional weather data), produces short‑horizon probabilistic forecasts, detects anomalous conditions, and **optimizes grid/battery actions** using linear programming. The platform also provides monitoring, retraining hooks, and a Next.js operator dashboard. The end result is a system that is not just technically sophisticated, but operationally meaningful: predictions → decisions → measurable outcomes.

GridPulse is designed as a modular, production‑ready architecture. It includes reproducible data pipelines, configurable model training, evaluation reports, and an API layer suitable for integration with real‑world operator tooling. The project demonstrates full‑stack applied ML engineering: from raw data to a decision product.

---

## 2) Problem Statement  
Electrical grids are becoming increasingly volatile. The integration of wind and solar introduces large, non‑dispatchable supply swings, while demand patterns are affected by time‑of‑day, seasonality, and extreme weather. Traditional grid management relies heavily on static rules or short‑term operational heuristics, which are insufficient for:

- **Intermittency** of renewables  
- **Uncertain demand spikes**  
- **Operational constraints** (battery SOC, charging limits, grid import caps)  
- **Cost and carbon trade‑offs**

Forecasting demand and supply is necessary but not sufficient. Grid operators also need to know **what to do** next: how much to draw from the grid, when to charge/discharge storage, and how to reduce carbon impact without violating constraints.

GridPulse solves this by **coupling forecasts with an optimization engine** that generates feasible dispatch plans. The optimization is cost‑ and carbon‑aware, and the platform provides baseline comparisons to measure impact.

---

## 3) Project Objectives  
GridPulse is structured around six core objectives:

1. **Build a reliable data pipeline** for power system time‑series.  
2. **Forecast load and renewables** using both ML and DL models.  
3. **Detect anomalies** to guard against unstable or unreliable inputs.  
4. **Optimize dispatch** using linear programming under constraints.  
5. **Support production operations** with monitoring and retraining.  
6. **Deliver a usable product** via API + dashboard.

---

## 4) Data Sources & Inputs  

GridPulse supports multiple datasets, each used to demonstrate robustness across regions.

### Primary Dataset: OPSD Germany (Hourly)  
**Source:** Open Power System Data (OPSD)  
**Signals:**
- Load (actual)  
- Wind generation  
- Solar generation  
- (Optional) day‑ahead price signal

This dataset is used for the main pipeline and baseline analysis.

### Secondary Dataset: EIA‑930 (USA)  
**Source:** U.S. EIA Form 930  
**Signals:**
- Balancing‑authority level load and generation  
- Optional regional selection (e.g., MISO, PJM, CAISO)

### Optional Weather Features  
Weather features can be integrated (e.g., Berlin temperature) to improve renewable and demand forecasting.

### Licensing & Data Hygiene  
Raw datasets are **not** committed to the repository. A dedicated `DATA.md` describes:
- the expected file locations  
- licensing requirements  
- download steps  
- attribution

This keeps the project open‑source compliant and professional.

---

## 5) System Architecture (Level‑4 Decision Loop)

GridPulse is built as a **Level‑4 decision system**: predictions → decisions → measurable impact → monitoring.

```
[ Raw Grid + Weather Data ]
              ↓
[ Data Validation & Time‑Aware Feature Store ]
              ↓
[ Forecasting Layer ]
  - Load forecast (GBM + DL)
  - Wind/Solar forecast
  - Prediction intervals
              ↓
[ Uncertainty & Reliability Layer ]
  - Residual‑based confidence
  - Anomaly flags (IForest / z‑score)
              ↓
[ Decision Layer ]
  - Optimization Engine (LP)
  - Cost objective
  - Carbon penalty objective
  - Battery + grid constraints
              ↓
[ Baseline Comparator ]
  - Grid‑only policy
  - Naive battery policy (optional)
              ↓
[ Impact Evaluation ]
  - Cost savings %
  - Carbon reduction %
  - Peak shaving %
              ↓
[ Monitoring & Governance ]
  - Data drift
  - Model drift
  - Retraining triggers
              ↓
[ Product Layer ]
  - API (forecast / optimize / monitor)
  - Next.js operator dashboard
```

This architecture is explicitly designed to show that the model is not just “accurate,” but **decision‑effective**.

---

## 6) Data Pipeline & Feature Engineering  

The data pipeline focuses on **time‑aware, leakage‑safe** processing.

### Key Steps  
1. **Ingestion & validation**  
   - Load raw CSV  
   - Validate required columns  
   - Align timestamps and enforce hourly cadence

2. **Feature engineering**  
   - Calendar features: hour‑of‑day, day‑of‑week  
   - Lag features: t‑1, t‑24, t‑168  
   - Rolling windows: 24h / 168h moving statistics  
   - Optional price/carbon proxies if explicit series missing

3. **Data splits**  
   - Time‑based train/val/test split  
   - Ensures no future leakage

Outputs are stored in Parquet for fast access.

---

## 7) Forecasting Layer  

GridPulse uses a **comparative modeling strategy**:

### Model A: Gradient Boosting (GBM)  
- LightGBM / XGBoost  
- Strong baseline for tabular, lag‑feature data  
- Fast to train, strong generalization on OPSD

### Model B: Deep Learning  
- LSTM and TCN  
- Captures temporal dependencies and nonlinearities  
- Uses scaled features and targets for stability  
- Supports sequence‑to‑sequence forecasting

### Forecast Outputs  
- 24‑hour horizon predictions  
- Residual quantiles for uncertainty  
- Stored model bundles include scalers + metadata

### Metrics  
- RMSE  
- MAE  
- sMAPE  
- MAPE  
- **Daylight MAPE** (solar‑only to avoid night‑time blow‑ups)

---

## 8) Anomaly Detection  

Forecasts are not always trustworthy. GridPulse includes a **safety layer**:

- **Residual‑based alerts:** if forecast error exceeds a dynamic threshold  
- **Isolation Forest:** detects multi‑feature anomalies  
- Flags “unreliable” intervals for operator review  
- Prevents feeding corrupted signals into optimization

This layer is lightweight but critical for operational reliability.

---

## 9) Optimization & Decision Engine  

The decision engine is the core of GridPulse. It solves a linear program that determines:

- Grid import (MW)  
- Battery charge/discharge (MW)  
- Battery state of charge (MWh)

### Objective  
Minimize:

```
cost + carbon_penalty + optional_peak_penalty
```

### Constraints  
- Battery SOC bounds  
- Charge/discharge limits  
- Efficiency loss  
- Grid import limits  
- Energy balance (load = renew + battery + grid)

This produces a **feasible dispatch plan** for each horizon.

---

## 10) Baseline Comparator & Impact Evaluation  

The system includes baseline policies to measure impact:

- **Grid‑only baseline**  
- **Naive battery** (optional)

Impact metrics are computed by comparing GridPulse vs baseline:

- Cost savings %  
- Carbon reduction %  
- Peak shaving %

Results are written to:

- `reports/impact_summary.csv`  
- `reports/impact_comparison.md`  
- `reports/figures/impact_savings.png`

Note: actual savings depend on **price and carbon signals**. If those are missing, cost/carbon improvements will be near zero. This is a design choice to prevent fake or unverifiable claims.

---

## 11) Monitoring & Governance  

GridPulse includes monitoring logic to track:

- **Data drift:** distribution changes vs training  
- **Model drift:** metric degradation  
- **Retraining triggers:** threshold‑based policy

This ensures models remain valid as grid conditions shift.

---

## 12) Product Layer  

### API  
FastAPI endpoints include:

- `/forecast`  
- `/optimize`  
- `/monitor`  
- `/health`

### Dashboard  
Next.js operator dashboard provides:

- Forecast plots  
- KPI tiles (cost/carbon/anomaly)  
- Dispatch visualizations  
- Auto‑refresh support

Together, these layers demonstrate real‑world usability.

---

## 13) Reproducibility  

GridPulse emphasizes reproducibility and auditability:

- Fixed seeds for Python/NumPy/PyTorch  
- Pipeline cache with hash tracking  
- Version locks in `requirements.lock.txt`  
- Run manifests with config + environment snapshots  
- End‑to‑end runbook notebook (`13_runbook_end_to_end.ipynb`)

A single script (`scripts/repro_run.sh`) can rebuild the pipeline from raw data to reports.

---

## 14) Evaluation & Reporting  

Generated reports include:

- Formal evaluation report (1 page)  
- Multi‑horizon backtest JSON + plot  
- Model cards for each target  
- Dispatch comparison plots  
- Arbitrage visualization (if price exists)

This is designed for both **technical audit** and **public presentation**.

---

## 15) Limitations & Honest Boundaries  

GridPulse is intentionally **honest about claims**:

- Impact metrics are only meaningful when price/carbon signals exist.  
- Forecasting performance varies by dataset and target.  
- LSTM/TCN models require careful scaling and tuning.  
- The current optimization is linear and does not model market bidding.

These limitations are documented to maintain credibility.

---

## 16) Why This Is “Level‑4”  

Level‑4 systems do not just predict. They **decide**, evaluate decisions against a baseline, and prove measurable improvements.

GridPulse qualifies because it includes:

- Prediction → optimization → measurable impact  
- Baseline comparison  
- Operational monitoring  
- Operator‑ready delivery

This is the difference between a research project and a decision‑grade system.

---

## 17) Future Work  

Potential extensions include:

- Dynamic pricing integration  
- Regional carbon intensity streams  
- Multi‑node grid topology  
- Dispatch under uncertainty (stochastic LP)  
- Reinforcement learning (optional, advanced)

These are optional and not required for the project’s current goals.

---

## 18) Conclusion  

GridPulse is a complete, decision‑grade energy intelligence system. It goes beyond forecasting to produce **feasible, measurable, and auditable dispatch actions**. It demonstrates end‑to‑end ML engineering skills: data pipelines, model training, evaluation, optimization, monitoring, and product delivery.

The project is structured to be **competition‑ready**, **hire‑ready**, and **publishable**, with clear documentation, reproducible scripts, and a professional product interface. Most importantly, it does not over‑claim results — it provides a rigorous path to generating verified impact metrics.
