# GridPulse: Autonomous Energy Intelligence & Optimization Platform

![CI](https://github.com/Pratikn03/Capstone_Projects/actions/workflows/ci.yml/badge.svg)
![Reproducible](https://img.shields.io/badge/reproducible-yes-success)

GridPulse is an end‑to‑end energy intelligence platform for modern electrical grids. It ingests power‑system and weather data, forecasts load and renewables, detects anomalies, and optimizes dispatch under cost and carbon objectives. The system includes MLOps monitoring and a Streamlit operator dashboard.

## Results Snapshot
The latest model comparison (mean across targets) is visualized below.

![Model Comparison](reports/figures/model_comparison.png)

## Latest Run Summary
- Run ID: `20260204_054609`
- Bundle: `artifacts/submission_bundle_20260204_054609/`
- Dataset: OPSD Germany (hourly)
- Carbon signal: SMARD hourly (2015–2020)
- Carbon weight: 20.0 (see `configs/optimization.yaml`)
- Carbon budget: 1% below baseline (see `configs/optimization.yaml`)

### Forecast Highlights (GBM, Test Split)
| Target | RMSE | MAE | sMAPE | MAPE |
|---|---:|---:|---:|---:|
| load_mw | 280.44 | 165.10 | 0.0035 | 0.0035 |
| wind_mw | 91.62 | 65.52 | 0.0163 | 0.0188 |
| solar_mw | 224.09 | 114.24 | 0.7028 | 0.1712 (daylight) |
| price_eur_mwh | 4.74 | 2.29 | 0.1001 | 1.4336 |

## Live Benchmark (Auto-Updated)
Metrics from the latest local run (`reports/impact_summary.csv`):

- Report: `reports/impact_comparison.md`
- Figure: `reports/figures/dispatch_compare.png`
- Summary CSV: `reports/impact_summary.csv`
- Savings Plot: `reports/figures/impact_savings.png`

| Metric | Value |
|---|---:|
| Cost savings | -105.69% |
| Carbon reduction | 1.00% |
| Peak shaving | 0.00% |

### Submission Bundle
- Manifest: `artifacts/submission_bundle_20260204_054609/manifest.json`
- Reports: `artifacts/submission_bundle_20260204_054609/reports/`
- Impact summary: `artifacts/submission_bundle_20260204_054609/reports/impact_comparison.md`

**Figure 1: GridPulse arbitrage decision vs grid-only baseline.**
![Arbitrage Optimization](reports/figures/arbitrage_optimization.png)

## Demo
Short preview of the forecasting inputs (sample run):

![GridPulse Demo](reports/figures/demo.gif)

![Dashboard Snapshot](reports/figures/dashboard_snapshot.png)

## Architecture

```mermaid
flowchart TD
  A["OPSD and Weather Inputs"] --> B["Data Ingestion and Validation"]
  B --> C["Feature Store - Time Aware"]
  C --> D["Forecasting Engine - GBM and LSTM/TCN"]
  D --> E["Residual and IsolationForest Anomaly Detection"]
  D --> F["Optimization Engine (LP) - Cost and Carbon"]
  E --> G["API - forecast, anomaly, optimize, monitor"]
  F --> G
  G --> H["Streamlit Operator Dashboard"]
  C --> I["Monitoring - Data Drift and Model Drift"]
  D --> I
  I --> J["Retraining Rules and Model Promotion"]
```

## Core Capabilities
- **Forecasting:** Gradient boosting and deep learning (LSTM/TCN) with 24‑hour horizons and intervals.
- **Anomaly Detection:** Residual z‑scores + Isolation Forest.
- **Optimization:** Linear programming dispatch with battery and grid constraints.
- **MLOps:** Drift monitoring and retraining signals.
- **Product:** Streamlit dashboard for operators.

## Technology Stack
- **Python 3.9+**
- **Data:** Pandas, NumPy, Scikit‑learn, PyArrow
- **Forecasting:** LightGBM/XGBoost, PyTorch
- **API:** FastAPI, Uvicorn
- **UI:** Streamlit
- **Ops:** Docker, Git

## Data Sources
- **Power system data:** Open Power System Data (OPSD) — Germany load/wind/solar time‑series.
- **Weather data (optional):** Open‑Meteo for Berlin hourly features.
- **USA dataset (optional):** EIA Form 930 (hourly balancing‑authority demand + generation).

## Data & Licensing
This repo does **not** store raw datasets. See `DATA.md` for:
- dataset inventory (OPSD, EIA‑930, optional weather),
- expected file locations,
- licensing/attribution notes,
- reproducible download/processing steps.

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```
Exact versions are captured in `requirements.lock.txt` after installation.

### 1b) (Optional) Secrets + runtime settings
Copy `.env.example` to `.env` and fill in tokens if you plan to pull external carbon data (Electricity Maps, WattTime) or enable alerting.

### 2) Run the data pipeline
```bash
python -m gridpulse.data_pipeline.download_opsd --out data/raw
python -m gridpulse.data_pipeline.validate_schema --in data/raw --report reports/data_quality_report.md
python -m gridpulse.data_pipeline.build_features --in data/raw --out data/processed
python -m gridpulse.data_pipeline.split_time_series --in data/processed/features.parquet --out data/processed/splits
```
Or run the full pipeline with caching:
```bash
python -m gridpulse.pipeline.run --all
```

Optional weather ingestion + SQL storage:
```bash
python -m gridpulse.data_pipeline.download_weather --out data/raw --start 2017-01-01 --end 2020-12-31
python -m gridpulse.data_pipeline.build_features --in data/raw --out data/processed \
  --weather data/raw/weather_berlin_hourly.csv \
  --sql-out data/processed/gridpulse.duckdb --sql-engine duckdb
```

Optional price + carbon signals (recommended for realistic impact metrics):
```bash
python -m gridpulse.data_pipeline.build_features --in data/raw --out data/processed \
  --signals data/raw/price_carbon_signals.csv
```

Electricity Maps carbon intensity (real data):
```bash
export ELECTRICITYMAPS_TOKEN=...
python scripts/download_emaps_carbon.py \
  --zone <zone_code> \
  --start 2020-01-01T00:00:00Z \
  --end 2020-12-31T23:00:00Z \
  --out data/raw/carbon_signals.csv
# Or prepare from a CSV export:
python scripts/prepare_emaps_carbon.py \
  --in data/raw/electricitymaps_carbon.csv \
  --timestamp-col <timestamp_col_from_csv> \
  --carbon-col <carbon_col_from_csv> \
  --unit gco2_kwh \
  --zone-col <zone_col> --zone <zone_code> \
  --out data/raw/carbon_signals.csv
```

SMARD hourly carbon intensity (Germany, no token):
```bash
python scripts/download_smard_carbon.py \
  --region DE \
  --start 2015-01-01T00:00:00Z \
  --end 2020-09-30T23:00:00Z \
  --out data/raw/carbon_signals.csv \
  --factors configs/carbon_factors.yaml
```

WattTime marginal emissions (MOER):
```bash
export WATTTIME_USERNAME=... WATTTIME_PASSWORD=...
python scripts/download_watttime_moer.py \
  --region CAISO_NORTH \
  --start 2020-01-01T00:00:00Z \
  --end 2020-12-31T23:00:00Z \
  --out data/raw/moer_signals.csv
python scripts/merge_signals.py \
  --carbon data/raw/carbon_signals.csv \
  --moer data/raw/moer_signals.csv \
  --out data/raw/price_carbon_signals.csv
```

### 3) Train forecasting models (GBM + LSTM + TCN)
```bash
python -m gridpulse.forecasting.train --config configs/train_forecast.yaml
```

### Optional: Train both OPSD + USA EIA930
```bash
python scripts/train_multi_dataset.py --ba MISO
```

### Optional: Generate reports for EIA930
```bash
python scripts/build_reports.py \
  --features data/processed/us_eia930/features.parquet \
  --splits data/processed/us_eia930/splits \
  --models-dir artifacts/models_eia930 \
  --reports-dir reports/eia930
```

Training outputs include RMSE, MAE, MAPE, sMAPE, and daylight‑MAPE for solar. A walk‑forward report is optionally generated at `reports/walk_forward_report.json`.

### Impact benchmark (evidence‑backed claims)
Impact metrics are generated by `scripts/build_reports.py` using the optimization config. If price signals are missing, cost savings will be near zero. To update the README table:
```bash
python scripts/build_reports.py
python scripts/update_readme_impact.py
```

## Reproducibility
- **Fixed seed:** `configs/train_forecast.yaml` includes `seed: 42` (override as needed).
- **Deterministic training:** seeds are applied to Python, NumPy, and PyTorch.
- **Exact steps:** use `notebooks/13_runbook_end_to_end.ipynb` for a full end‑to‑end run.
- **Version locks:** `requirements.lock.txt` captures installed versions.
- **Pipeline cache:** `.cache/pipeline.json` tracks hashes to skip unchanged steps.
- **Scaled training:** LSTM/TCN training applies feature + target scaling for stability; scalers are stored in model bundles and used at inference time.
- **Run snapshot:** each pipeline run writes `artifacts/runs/<run_id>/manifest.json`, config copies, and `pip_freeze.txt`.

Reproducible one‑command run:
```bash
./scripts/repro_run.sh
```

## Reports
- `reports/formal_evaluation_report.md` — 1‑page evaluation summary with plots.
- `reports/model_cards/` — per‑target model cards.
- `reports/multi_horizon_backtest.json` — multi‑horizon backtest results.
- `reports/rolling_backtest.md` — rolling weekly backtest with confidence intervals.
- `reports/impact_comparison.md` — baseline vs optimized dispatch impact (cost + carbon).
- `reports/impact_summary.csv` — summary metrics for README (auto‑updated).
- `reports/case_study.md` — end‑to‑end case study with failure‑day analysis.
- `scripts/build_reports.py` — regenerate reports/figures after training (supports dataset-specific paths).

### 4) Start API
```bash
uvicorn services.api.main:app --reload --port 8000
```

### 5) Start dashboard
```bash
streamlit run services/dashboard/app.py
```

### 5b) Start API + dashboard together (optional)
```bash
./scripts/run_all.sh
```

### 6) Monitor + Optimize (API)
```bash
curl http://localhost:8000/monitor
curl -X POST http://localhost:8000/optimize \
  -H 'Content-Type: application/json' \
  -d '{"forecast_load_mw":[8000,8200],"forecast_renewables_mw":[3200,3100]}'
```

## Notebooks
- `notebooks/01_eda.ipynb` — dataset inspection
- `notebooks/02_baselines.ipynb` — baseline evaluation
- `notebooks/03_feature_pipeline.ipynb` — data pipeline
- `notebooks/04_train_models.ipynb` — GBM/LSTM/TCN training
- `notebooks/05_inference_intervals.ipynb` — forecast + intervals
- `notebooks/06_error_analysis.ipynb` — residual analysis
- `notebooks/07_production_run.ipynb` — end‑to‑end runbook
- `notebooks/08_weather_features.ipynb` — optional weather features
- `notebooks/09_walk_forward_report.ipynb` — backtest visualization
- `notebooks/10_optimization_engine.ipynb` — optimization + dispatch plots
- `notebooks/11_monitoring_drift.ipynb` — drift checks + retraining decision
- `notebooks/12_api_dashboard_smoke_test.ipynb` — API health + endpoint checks
- `notebooks/13_runbook_end_to_end.ipynb` — full pipeline runbook

## Repo Layout
- `src/gridpulse/` — core library (data pipeline, forecasting, anomaly, optimizer, monitoring)
- `services/api/` — FastAPI service
- `services/dashboard/` — Streamlit app
- `configs/` — YAML configs
- `notebooks/` — EDA and training notebooks
- `data/` — raw/interim/processed datasets (git‑ignored)
- `artifacts/` — models and backtests (git‑ignored)
- `reports/` — reports (git‑ignored)

## Configs
- `configs/train_forecast.yaml` — training configuration (seed, models, horizons).
- `configs/forecast.yaml` — inference model bundle paths.
- `configs/optimization.yaml` — dispatch cost/carbon/battery constraints.

## License
MIT (edit if your program requires otherwise).
