# ORIUS Battery Framework — Phase 3: Environment & Reproducibility

**Status**: Environment fully specified from confirmed repo files. Reproducibility anchor locked.

---

## 1. Environment Specification

### Language and runtime
| Item | Requirement | Source |
|------|-------------|--------|
| Python | `>=3.11, <3.12` | `pyproject.toml` |
| OS | macOS (darwin) or Linux | confirmed: macOS darwin 25.2.0 |
| Node.js | Any LTS (for frontend only) | `frontend/` Next.js 14 |

### Core Python dependencies (from `requirements.lock.txt`)

| Package | Version | Role |
|---------|---------|------|
| `lightgbm` | 4.6.0 | GBM forecasting (primary model) |
| `torch` / `pytorch` | (DL models) | LSTM, TCN, N-BEATS, TFT, PatchTST |
| `numpy` | 2.0.2 | Numerical core |
| `pandas` | 2.3.3 | Dataframes, feature engineering |
| `pyarrow` | 21.0.0 | Parquet I/O |
| `scikit-learn` | (transitive) | Preprocessing, baselines |
| `fastapi` | 0.128.0 | REST API service |
| `uvicorn` / `httptools` | 0.7.1 | ASGI server |
| `pydantic` | 2.12.5 | Data validation / schemas |
| `duckdb` | 1.4.4 | Audit ledger, state persistence |
| `highspy` | 1.8.1 | HiGHS LP solver |
| `pyomo` | 6.8.2 | Optimization modeling |
| `matplotlib` | 3.9.4 | Figures |
| `plotly` | 6.5.2 | Interactive plots |
| `altair` | 5.5.0 | Declarative charts |
| `ngboost` | 0.5.8 | Distributional forecasting |
| `PyYAML` | 6.0.3 | Config loading |
| `python-dotenv` | 1.2.1 | Environment variables |
| `psutil` | 7.2.2 | System metrics |
| `GitPython` | 3.1.46 | Git hash in manifests |
| `jupyter_core` | 5.8.1 | Notebook support |
| `ipykernel` | 6.31.0 | Kernel for notebooks |

> **Canonical dependency authority**: `requirements.lock.txt`. `requirements.txt` is a shim that points to it.

---

## 2. Setup Options

### Option A — Python venv (recommended for scripts-only work)

```bash
cd <repo-root>

# Create venv and install all locked deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt

# Install the orius package in editable mode
pip install -e .

# Verify
python -c "import orius; print('orius OK')"
python -c "import lightgbm; print(lightgbm.__version__)"
python -c "import duckdb; print(duckdb.__version__)"
python -c "import fastapi; print(fastapi.__version__)"
```

> Or use the Makefile shortcut:
> ```bash
> make setup   # runs: python -m venv .venv && pip install -r requirements.lock.txt
> ```

### Option B — Docker (recommended for full stack with Kafka + Prometheus)

```bash
cd <repo-root>

# Dev stack (API + DB only)
docker compose -f docker/docker-compose.yml up -d

# Full stack (API + Kafka + Prometheus + AlertManager + Grafana)
docker compose -f docker/docker-compose.full.yml up -d

# Streaming stack
docker compose -f docker/docker-compose.streaming.yml up -d
```

Available images:
- `docker/Dockerfile.api` — Python API service
- `docker/Dockerfile.frontend` — Next.js dashboard

### Option C — Frontend only

```bash
cd <repo-root>/frontend
npm install
npm run dev   # starts Next.js dev server
```

---

## 3. Reproducibility Contract

### Anchor
`reports/publish/reproducibility_lock.json` is the canonical reproducibility anchor.

Current locked state (as of 2026-03-15T20:32:24Z):
```json
{
  "run_id": "publish_20260315_203224",
  "seeds": {
    "DE": [42, 123, 456, 789, 2024, 1337, 7777, 9999],
    "US": [42, 123, 456, 789, 2024]
  }
}
```

### Seed management
Seeds are managed in `src/orius/utils/seed.py`. Every run that generates locked outputs must use these seeds. Any update to seeds must update `reproducibility_lock.json` and all downstream locked CSVs.

### Run ID convention
Format: `publish_YYYYMMDD_HHMMSS` — always write this to the lock file when generating new locked results.

### Hash verification
`dc3s/certificate.py` computes `compute_model_hash()` and `compute_config_hash()` at each DC3S step. These hashes are stored in the audit ledger (`data/audit/dc3s_audit.duckdb`) and allow post-hoc verification that a certificate was generated with the exact model + config that was locked.

### Manifest files
| File | Content |
|------|---------|
| `paper/metrics_manifest.json` | Metric values linked to paper tables |
| `paper/manifest.yaml` | Paper asset inventory |
| `paper/assets/data/metrics_snapshot.json` | Frozen metrics snapshot for paper |
| `paper/claim_matrix.csv` | Each paper claim mapped to evidence |

---

## 4. Configuration Management

### Config hierarchy
All configs live in `configs/`. Loaded by `src/orius/utils/config.py`.

```
configs/
├── dc3s.yaml                    # DC3S pipeline params (OQE, RUI, SAF, FTIT, drift, shield, audit)
├── dc3s_calibrated.yaml         # Post-calibration locked DC3S params
├── dc3s_ablation.yaml           # Ablation sweep variant
├── dc3s_ablations.yaml          # Multi-ablation config
├── cpsbench_r1_severity.yaml    # CPSBench severity sweep
├── forecast.yaml                # Forecast training (DE)
├── forecast_eia930.yaml         # Forecast training (US MISO)
├── train_forecast_eia930.yaml   # US training template
├── train_forecast_ercot.yaml    # ERCOT variant
├── train_forecast_pjm.yaml      # PJM variant
├── train_forecast_template.yaml # Base training template
├── uncertainty.yaml             # CQR / conformal calibration
├── optimization.yaml            # LP/robust dispatch params
├── streaming.yaml               # Kafka consumer config
├── monitoring.yaml              # Drift detectors, alert thresholds
├── iot.yaml                     # IoT edge agent (modbus, HTTP gateway)
├── anomaly.yaml                 # Anomaly detection
├── serving.yaml                 # API serving config
├── ablations.yaml               # General ablation config
├── data.yaml                    # Dataset registry
├── carbon_factors.yaml          # Carbon intensity factors
└── publish_audit.yaml           # Pre-publication audit config
```

### Loading a config in Python
```python
from orius.utils.config import load_config

cfg = load_config("configs/dc3s.yaml")
alpha = cfg.dc3s.alpha0       # → 0.10
k_quality = cfg.dc3s.k_quality  # → 0.2
```

### Key config values to know
| Config key | File | Value | Effect |
|-----------|------|-------|--------|
| `dc3s.alpha0` | `dc3s.yaml` | 0.10 | Base coverage level |
| `dc3s.k_quality` | `dc3s.yaml` | 0.2 | κ_r in RAC-Cert |
| `dc3s.k_sensitivity` | `dc3s.yaml` | 0.4 | κ_s in RAC-Cert |
| `dc3s.infl_max` | `dc3s.yaml` | 2.0 | Max inflation cap |
| `dc3s.reliability.min_w` | `dc3s.yaml` | 0.05 | Floor for w_t |
| `dc3s.shield.mode` | `dc3s.yaml` | `projection` | Repair strategy |
| `dc3s.shield.reserve_soc_pct_drift` | `dc3s.yaml` | 0.08 | SOC buffer in drift |
| `dc3s.audit.duckdb_path` | `dc3s.yaml` | `data/audit/dc3s_audit.duckdb` | Audit ledger path |
| `dc3s.drift.ph_lambda` | `dc3s.yaml` | 5.0 | PH detector threshold |
| `dc3s.ftit.decay` | `dc3s.yaml` | 0.98 | Fault evidence EMA |

---

## 5. CLI Entry Points (Makefile Targets)

All commands run from `<repo-root>/` with `.venv` activated.

### Setup & verification
```bash
make setup             # create venv, install deps
make lint              # compile check all Python
make lint-release      # ruff check for release-critical files
make test              # pytest -q --no-cov
make test-quick        # exclude slow/integration tests
make test-cov          # pytest with coverage (target: 80%)
```

### Training
```bash
make train             # train DE forecast models
make train-us          # train US EIA-930 forecast models
make train-dataset     # train on specific dataset
make train-all         # train all datasets
```

### Data pipeline
```bash
make data              # download + feature engineering
make refresh-data      # re-download raw data
make extract-data      # extract features only
```

### Benchmarking
```bash
make cpsbench          # run CPSBench battery track
make dc3s-demo         # run DC3S demo
make ablations         # run DC3S ablations
```

### Reports & publication
```bash
make reports           # generate DE reports
make reports-us        # generate US reports
make stats-tables      # generate statistics tables
make paper-assets      # sync paper assets from reports/
make paper-compile     # compile paper.tex → paper.pdf
make paper-verify      # verify paper claims vs locked metrics
make paper-freeze      # freeze all paper assets
make paper-refresh     # full rebuild: assets + compile
make publication-artifact  # build final publication zip
```

### Audits
```bash
make na-audit          # NA/missing value audit
make leakage-audit     # data leakage audit
make code-health-audit # code quality audit
make publish-audit     # pre-publication check
make publish-audit-isolated  # isolated audit environment
make figure-inventory-audit  # check all figures present
```

### IoT / HIL
```bash
make iot-sim           # run closed-loop IoT simulation
```

### Monitoring
```bash
make monitor           # run drift + health monitoring
make observability     # start Prometheus + Grafana
make down-observability # stop observability stack
```

---

## 6. Environment Verification Checklist

Run these after setup to confirm everything works:

```bash
# 1. Python package import check
python -c "
from orius.dc3s import run_dc3s_step
from orius.forecasting.ml_gbm import GBMForecaster
from orius.optimizer.lp_dispatch import LPDispatcher
from orius.cpsbench_iot.runner import CPSBenchRunner
print('All core imports OK')
"

# 2. DuckDB audit DB check
python -c "
import duckdb
import os
db_path = 'data/audit/dc3s_audit.duckdb'
if os.path.exists(db_path):
    con = duckdb.connect(db_path)
    tables = con.execute('SHOW TABLES').fetchall()
    print('Audit DB tables:', tables)
else:
    print('Audit DB does not exist yet — will be created on first run')
"

# 3. Config loading check
python -c "
from orius.utils.config import load_config
cfg = load_config('configs/dc3s.yaml')
print('DC3S alpha0:', cfg.dc3s.alpha0)
print('DC3S k_quality:', cfg.dc3s.k_quality)
"

# 4. Locked results check
python - <<'PY'
import pandas as pd, json, os
checks = [
    ('reports/impact_summary.csv', 'DE impact'),
    ('reports/publication/dc3s_main_table_ci.csv', 'DC3S main results'),
    ('reports/publication/dc3s_latency_summary.csv', 'DC3S latency'),
    ('reports/publication/reliability_group_coverage.csv', 'Group coverage'),
    ('reports/publication/transfer_stress.csv', 'Transfer stress'),
    ('reports/publish/reproducibility_lock.json', 'Repro lock'),
]
for path, label in checks:
    if os.path.exists(path):
        print(f'[OK] {label}: {path}')
    else:
        print(f'[MISSING] {label}: {path}')
PY

# 5. Quick test suite
make test-quick
```

---

## 7. Environment for Manuscript Build

LaTeX compilation requires a TeX distribution installed separately.

```bash
# macOS — install MacTeX or BasicTeX
brew install --cask mactex-no-gui   # ~3GB full install
# or
brew install --cask basictex        # minimal, add packages as needed

# Verify
pdflatex --version

# Build paper
cd <repo-root>/paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex

# Full refresh (sync assets + compile)
cd <repo-root>
make paper-refresh
```

---

*Next: see `04-core-apis.md` for runtime object definitions and Python type specifications.*
