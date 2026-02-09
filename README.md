# GridPulse: Autonomous Energy Intelligence & Optimization Platform

![CI](https://github.com/Pratikn03/Capstone_Projects/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Next.js 15](https://img.shields.io/badge/next.js-15-black)
![Models](https://img.shields.io/badge/models-21-green)
![Reproducible](https://img.shields.io/badge/reproducible-yes-success)

GridPulse is a production-grade energy intelligence platform that forecasts load, wind, and solar generation across **two countries** (Germany & USA), detects anomalies, optimizes dispatch under cost and carbon objectives, and serves results through a real-time Next.js operator dashboard — all backed by **21 trained ML models** and real-world data.

---

## Key Results

### Germany (OPSD) — 17,377 hourly observations × 98 features

| Model | Target | RMSE (MW) | MAE (MW) | R² |
|---|---|---:|---:|---:|
| **GBM** | load_mw | 270.3 | 168.5 | 0.998 |
| **GBM** | wind_mw | 47.3 | 29.8 | 0.999 |
| **GBM** | solar_mw | 24.7 | 12.2 | 0.999 |
| **GBM** | price_eur_mwh | 3.1 | 1.9 | 0.999 |
| LSTM | load_mw | 4,975.2 | 3,633.8 | — |
| TCN | load_mw | 6,157.1 | 5,172.1 | — |

**Impact:** 2.89% cost savings · 0.58% carbon reduction · 7.09% peak shaving

### USA (EIA-930 / MISO) — 13,638 hourly observations × 118 features

| Model | Target | RMSE (MW) | MAE (MW) | R² |
|---|---|---:|---:|---:|
| **GBM** | load_mw | 145.4 | 88.7 | 0.999 |
| **GBM** | wind_mw | 52.9 | 31.2 | 0.999 |
| **GBM** | solar_mw | 24.3 | 10.6 | 0.999 |
| LSTM | load_mw | 2,471.0 | — | — |
| TCN | load_mw | 3,105.8 | — | — |

**Impact:** 0.03% cost savings (no real price signal)

---

## Architecture

```mermaid
flowchart TD
  subgraph Data Sources
    A1["OPSD Germany\n(load, wind, solar, price)"]
    A2["EIA-930 USA\n(MISO demand + generation)"]
    A3["Open-Meteo Weather\n(Berlin & Chicago)"]
  end

  subgraph Pipeline
    B["Data Ingestion & Validation"]
    C["Feature Engineering\n(lags, calendar, weather, holidays)"]
    D["Time-Series Splits\n(train / val / test)"]
  end

  subgraph ML Engine
    E["GBM (LightGBM)\n12 DE + 9 US models"]
    F["Deep Learning\nLSTM + TCN per target"]
    G["Conformal Prediction\n90% intervals"]
  end

  subgraph Operations
    H["Anomaly Detection\n(residual z-score + IsolationForest)"]
    I["LP Dispatch Optimizer\n(cost + carbon + battery)"]
    J["MLOps Monitoring\n(drift, retraining triggers)"]
  end

  subgraph Serving
    K["FastAPI\n/forecast · /optimize · /monitor"]
    L["Next.js 15 Dashboard\n8 pages · region toggle"]
  end

  A1 & A2 & A3 --> B --> C --> D
  D --> E & F
  E & F --> G
  G --> H & I
  H & I & J --> K --> L
```

---

## Dashboard

The Next.js 15 operator dashboard shows **real ML results** — no mock data on production pages.

| Page | Description |
|---|---|
| **Overview** | KPIs, dispatch chart, model registry, dataset stats |
| **Forecasting** | Forecast vs actual, model comparison table, hourly profiles |
| **Optimization** | Dispatch plan, battery SOC, cost impact panel |
| **Carbon** | Carbon impact breakdown, baseline vs optimized emissions |
| **Anomalies** | Z-score timeline, event log, data quality overview |
| **Monitoring** | Model drift, active model versions, artifact sizes |
| **Reports** | Formal evaluation, model cards, publication figures |
| **Data Explorer** | Dataset statistics, time series, hourly profiles, forecast comparison |

All pages support a **DE / US region toggle** in the top bar.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **ML/Data** | Python 3.11, LightGBM 4.6, PyTorch 2.8, scikit-learn, Pandas, NumPy |
| **API** | FastAPI 0.110+, Uvicorn, Pydantic v2 |
| **Frontend** | Next.js 15.3, React 19, TypeScript 5.8, Tailwind v4, Recharts, Framer Motion |
| **Ops** | Docker, GitHub Actions CI/CD, DuckDB |
| **Optimization** | Pyomo + GLPK (LP dispatch) |

---

## Quickstart

### 1. Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Run Data Pipeline

```bash
# Germany (OPSD)
python -m gridpulse.pipeline.run --all

# USA (EIA-930)
python scripts/build_features_eia930.py
```

### 3. Train All Models (21 total)

```bash
# Germany: 12 models (GBM×4 + LSTM×4 + TCN×4)
python -m gridpulse.forecasting.train --config configs/train_forecast.yaml

# USA: 9 models (GBM×3 + LSTM×3 + TCN×3)
python -m gridpulse.forecasting.train --config configs/train_forecast_eia930.yaml
```

Or train both at once:
```bash
python scripts/train_multi_dataset.py --ba MISO
```

### 4. Extract Dashboard Data

```bash
make extract-data
# → generates data/dashboard/*.json (17 files)
```

### 5. Generate Reports & Figures

```bash
python scripts/build_reports.py                     # DE reports
python scripts/build_reports.py \
  --features data/processed/us_eia930/features.parquet \
  --splits data/processed/us_eia930/splits \
  --models-dir artifacts/models_eia930 \
  --reports-dir reports/eia930                       # US reports
```

### 6. Start Services

```bash
# Terminal 1: API
make api

# Terminal 2: Dashboard
make frontend
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

### Docker Deployment

```bash
cd docker && docker compose up --build
```

---

## 21 Trained Models

| # | Region | Target | Model | Artifact |
|---|---|---|---|---|
| 1 | DE | load_mw | GBM | `artifacts/models/gbm_load_mw.txt` |
| 2 | DE | wind_mw | GBM | `artifacts/models/gbm_wind_mw.txt` |
| 3 | DE | solar_mw | GBM | `artifacts/models/gbm_solar_mw.txt` |
| 4 | DE | price_eur_mwh | GBM | `artifacts/models/gbm_price_eur_mwh.txt` |
| 5 | DE | load_mw | LSTM | `artifacts/models/lstm_load_mw.pt` |
| 6 | DE | wind_mw | LSTM | `artifacts/models/lstm_wind_mw.pt` |
| 7 | DE | solar_mw | LSTM | `artifacts/models/lstm_solar_mw.pt` |
| 8 | DE | price_eur_mwh | LSTM | `artifacts/models/lstm_price_eur_mwh.pt` |
| 9 | DE | load_mw | TCN | `artifacts/models/tcn_load_mw.pt` |
| 10 | DE | wind_mw | TCN | `artifacts/models/tcn_wind_mw.pt` |
| 11 | DE | solar_mw | TCN | `artifacts/models/tcn_solar_mw.pt` |
| 12 | DE | price_eur_mwh | TCN | `artifacts/models/tcn_price_eur_mwh.pt` |
| 13 | US | load_mw | GBM | `artifacts/models_eia930/gbm_load_mw.txt` |
| 14 | US | wind_mw | GBM | `artifacts/models_eia930/gbm_wind_mw.txt` |
| 15 | US | solar_mw | GBM | `artifacts/models_eia930/gbm_solar_mw.txt` |
| 16 | US | load_mw | LSTM | `artifacts/models_eia930/lstm_load_mw.pt` |
| 17 | US | wind_mw | LSTM | `artifacts/models_eia930/lstm_wind_mw.pt` |
| 18 | US | solar_mw | LSTM | `artifacts/models_eia930/lstm_solar_mw.pt` |
| 19 | US | load_mw | TCN | `artifacts/models_eia930/tcn_load_mw.pt` |
| 20 | US | wind_mw | TCN | `artifacts/models_eia930/tcn_wind_mw.pt` |
| 21 | US | solar_mw | TCN | `artifacts/models_eia930/tcn_solar_mw.pt` |

---

## Publication Figures

### Forecasting Performance

| Load | Wind | Solar |
|:---:|:---:|:---:|
| ![](reports/figures/forecast_vs_actual_load.png) | ![](reports/figures/forecast_vs_actual_wind.png) | ![](reports/figures/forecast_vs_actual_solar.png) |

| Rolling Backtest RMSE | Residual Distribution | Seasonality Heatmap |
|:---:|:---:|:---:|
| ![](reports/figures/rolling_backtest_rmse_by_week.png) | ![](reports/figures/error_distribution_residuals.png) | ![](reports/figures/seasonality_error_heatmap.png) |

### Uncertainty Quantification

| PI — Load (90%) | Coverage by Horizon | Interval Width |
|:---:|:---:|:---:|
| ![](reports/figures/prediction_intervals_load.png) | ![](reports/figures/coverage_by_horizon.png) | ![](reports/figures/interval_width_by_horizon.png) |

### Dispatch & Impact

| Dispatch Comparison | Battery SOC | Impact Savings | Cost vs Carbon |
|:---:|:---:|:---:|:---:|
| ![](reports/figures/dispatch_compare.png) | ![](reports/figures/soc_trajectory.png) | ![](reports/figures/impact_savings.png) | ![](reports/figures/cost_vs_carbon_tradeoff.png) |

### Monitoring & Anomalies

| Anomaly Timeline | Z-Score | Data Drift | Model Drift |
|:---:|:---:|:---:|:---:|
| ![](reports/figures/anomaly_timeline.png) | ![](reports/figures/residual_zscore_timeline.png) | ![](reports/figures/data_drift_ks_over_time.png) | ![](reports/figures/model_drift_metric_over_time.png) |

---

## Data Sources

| Dataset | Region | Rows | Features | Period |
|---|---|---:|---:|---|
| **OPSD** | Germany | 17,377 | 98 | 2015–2020 |
| **EIA-930** | USA (MISO) | 13,638 | 118 | 2019–2024 |
| **Open-Meteo** | Berlin + Chicago | — | 7 weather vars | aligned |
| **SMARD** | Germany | — | carbon intensity | 2015–2020 |

See `DATA.md` for download instructions and licensing.

---

## Project Structure

```
gridpulse/
├── src/gridpulse/          # Core library
│   ├── data_pipeline/      #   ingest, validate, features, splits
│   ├── forecasting/        #   GBM, LSTM, TCN training & inference
│   ├── anomaly/            #   residual z-scores + IsolationForest
│   ├── optimizer/          #   LP dispatch (Pyomo + GLPK)
│   └── monitoring/         #   drift detection, retraining triggers
├── services/api/           # FastAPI service (forecast, optimize, monitor)
├── frontend/               # Next.js 15 dashboard (8 pages)
│   ├── src/app/(dashboard)/   Overview, Forecasting, Optimization, ...
│   ├── src/components/        Charts, panels, sidebar, top bar
│   └── src/lib/               API hooks, server loaders, utilities
├── scripts/                # Training, reports, data extraction
├── configs/                # YAML configs (training, optimization, etc.)
├── notebooks/              # 14 Jupyter notebooks (EDA → production)
├── data/                   # raw / processed / dashboard JSON
├── artifacts/              # models, scalers, backtests, runs
├── reports/                # figures, model cards, evaluation reports
├── docker/                 # Dockerfiles + compose
└── .github/workflows/      # CI (backend + frontend) + deploy
```

---

## Reproducibility

```bash
# One-command reproducible run
./scripts/repro_run.sh
```

- **Fixed seed:** `seed: 42` in all training configs
- **Deterministic:** seeds applied to Python, NumPy, PyTorch
- **Version locks:** `requirements.lock.txt`
- **Pipeline cache:** `.cache/pipeline.json` skips unchanged steps
- **Run snapshots:** `artifacts/runs/<run_id>/manifest.json`

---

## Makefile Targets

| Target | Description |
|---|---|
| `make setup` | Create venv + install Python & Node deps |
| `make test` | Run pytest suite |
| `make api` | Start FastAPI on :8000 |
| `make frontend` | Start Next.js on :3000 |
| `make extract-data` | Extract dashboard JSON from parquets |
| `make train` | Train all DE models |
| `make reports` | Generate reports & figures |
| `make production` | Full pipeline → train → extract |

---

## Reports & Notebooks

**Reports:** `reports/formal_evaluation_report.md` · `reports/model_cards/` · `reports/case_study.md` · `reports/impact_comparison.md` · 19 publication figures

**Notebooks:** 14 notebooks from EDA through production deployment — see `notebooks/` directory.

---

## License

MIT
