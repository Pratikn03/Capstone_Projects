# ORIUS Battery Framework — Phase 8: Operations SOP

**This is the primary human-facing SOP.** Read this to set up, run, generate missing outputs, extend, or debug any part of the ORIUS battery system.

---

## Quick Reference

| Task | Command |
|------|---------|
| Set up environment | `make setup` |
| **ORIUS full pipeline check** | `make orius-check` or `make orius-check-quick` (includes universal framework) |
| **Run universal framework (5 domains)** | `python scripts/run_multi_domain_framework.py --out reports/multi_domain` |
| **Analyze agent artifact zip** | `make analyze-artifact` |
| **Download AV datasets** | `make av-datasets` |
| **Download Industrial datasets** | `make industrial-datasets` |
| **Download Healthcare datasets** | `make healthcare-datasets` |
| **Download all multi-domain datasets** | `make multi-domain-datasets` |
| Verify locked results | See §2 |
| Run CPSBench battery track | `make cpsbench` |
| Run DC3S demo | `make dc3s-demo` |
| Generate missing fault-perf table | See §5.1 |
| Generate 48h trace | See §5.2 |
| Lock latency table | See §5.3 |
| Run all tests | `make test` |
| Build paper PDF | `make paper-compile` |
| Pre-publication audit | `make publish-audit` |
| Add a new fault type | See §6.1 |
| Add a new forecast model | See §6.2 |
| Add a new region | See §6.3 |

---

## 1. Environment Setup

### Step 1 — Clone / navigate to repo

```bash
cd <repo-root>
```

### Step 2 — Create virtual environment and install dependencies

```bash
make setup
# This runs:
# python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.lock.txt
# cd frontend && npm install
```

### Step 3 — Activate environment

```bash
source .venv/bin/activate
```

### Step 4 — Verify installation

```bash
python -c "
from orius.dc3s import compute_reliability, repair_action, make_certificate
from orius.forecasting.ml_gbm import train_gbm, predict_gbm
from orius.optimizer import optimize_dispatch
from orius.cpsbench_iot.runner import run_suite
print('All core imports OK')
"
```

### Step 5 — Verify configs load

```bash
python -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path('configs/dc3s.yaml').read_text())
dc3s = cfg.get('dc3s', {})
print('DC3S alpha0:', dc3s.get('alpha0'))
print('k_quality:', dc3s.get('k_quality'))
print('min_w:', dc3s.get('reliability', {}).get('min_w'))
print('shield mode:', dc3s.get('shield', {}).get('mode'))
"
```

---

## 2. Verify Locked Evidence

Run after any code change to ensure locked results are not broken:

```bash
python - <<'PY'
import pandas as pd, json, os, sys

LOCKED_FILES = {
    'reports/impact_summary.csv': {
        'desc': 'DE dispatch impact',
        'check_cols': ['cost_savings_pct', 'carbon_reduction_pct', 'peak_shaving_pct'],
    },
    'reports/publication/dc3s_main_table_ci.csv': {
        'desc': 'DC3S main results',
        'check_cols': ['scenario', 'controller', 'violation_rate_mean'],
    },
    'reports/publication/dc3s_latency_summary.csv': {
        'desc': 'DC3S latency',
        'check_cols': ['component', 'mean_ms', 'p95_ms'],
    },
    'reports/publication/reliability_group_coverage.csv': {
        'desc': 'Group coverage',
        'check_cols': ['bin_id', 'picp', 'mean_interval_width'],
    },
    'reports/publish/reproducibility_lock.json': {
        'desc': 'Reproducibility lock',
        'check_cols': None,
    },
}

all_ok = True
for path, meta in LOCKED_FILES.items():
    if not os.path.exists(path):
        print(f'[MISSING] {meta["desc"]}: {path}')
        all_ok = False
        continue
    if path.endswith('.json'):
        with open(path) as f:
            data = json.load(f)
        print(f'[OK] {meta["desc"]}: run_id={data.get("run_id", "?")}')
    else:
        df = pd.read_csv(path)
        print(f'[OK] {meta["desc"]}: {len(df)} rows, cols: {list(df.columns[:4])}')

# Critical metric checks
print('\n=== Critical metric checks ===')
df = pd.read_csv('reports/publication/dc3s_main_table_ci.csv')
dc3s_rows = df[df['controller'] == 'dc3s_wrapped']
for _, row in dc3s_rows.iterrows():
    vr = row['violation_rate_mean']
    assert vr == 0.0, f'FAIL: dc3s_wrapped has TSVR={vr} in scenario {row["scenario"]}'
    print(f'[OK] dc3s_wrapped TSVR=0 in scenario {row["scenario"]}')

det_nominal = df[(df['controller']=='deterministic_lp') & (df['scenario']=='nominal')]
tsvr = det_nominal['violation_rate_mean'].values[0]
assert tsvr > 0, f'FAIL: deterministic_lp should have TSVR>0'
print(f'[OK] deterministic_lp TSVR={tsvr:.4f} in nominal (expected >0)')

if all_ok:
    print('\nAll locked files present and core metrics verified.')
else:
    print('\nWARNING: Some files are missing.')
PY
```

---

## 3. Run Core Experiments

### 3.1 DC3S Demo (quickest end-to-end check)

```bash
python scripts/run_dc3s_demo.py
# or:
make dc3s-demo
```

This runs a single DC3S episode and prints per-step output: `w_t`, RAC interval, `a_safe`, `intervened`, certificate hash.

### 3.2 CPSBench Battery Track (full benchmark)

```bash
# Full sweep: all scenarios × all controllers × 5 seeds
make cpsbench
# or:
python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml

# Outputs written to:
#   reports/publication/cpsbench_merged_sweep.csv
#   reports/publication/dc3s_main_table_ci.csv
```

Runtime: ~5–15 minutes depending on hardware.

### 3.3 DC3S Ablations

```bash
make ablations
# or:
python scripts/run_dc3s_ablations_cpsbench.py
# Output: reports/publication/dc3s_ablation_table.csv
```

### 3.4 Walk-Forward Forecast Backtest

```bash
# DE models
make train
# US models
make train-us
# or:
python scripts/train_dataset.py --config configs/forecast.yaml
python scripts/train_multi_dataset.py --config configs/train_forecast_eia930.yaml
```

### 3.5 Latency Benchmark

```bash
python scripts/benchmark_dc3s_steps.py
# Output: dc3s_latency_summary.csv (root), then copy to reports/publication/
```

### 3.6 Reliability Group Coverage Audit

```bash
python scripts/compute_reliability_group_coverage.py
# Output: reports/publication/reliability_group_coverage.csv
```

### 3.7 Transfer Stress Study

```bash
python scripts/run_transfer_stress.py
python scripts/cross_region_transfer.py
# Output: reports/publication/transfer_stress.csv, cross_region_transfer.csv
```

### 3.8 Sensitivity Sweeps (hyperparameter surfaces)

```bash
python scripts/run_sensitivity_sweeps.py
# Output: figures for violation rate vs dropout rate
```

### 3.9 Monitoring Run (drift + health check)

```bash
make monitor
# or:
python scripts/run_monitoring.py
# Output: reports/monitoring_report.md
```

---

## 4. Generate Missing Priority-1 Outputs

### 4.1 Fault-Performance Table (HIGHEST PRIORITY)

**Target**: `reports/publication/fault_performance_table.csv`

```bash
# Step 1: Run full CPSBench sweep
python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml

# Step 2: Generate fault-performance pivot table
python - <<'PY'
import pandas as pd
from pathlib import Path

df = pd.read_csv('reports/publication/cpsbench_merged_sweep.csv')
print(f'Loaded {len(df)} rows from cpsbench_merged_sweep.csv')
print('Scenarios:', df['scenario'].unique().tolist())
print('Controllers:', df['controller'].unique().tolist())

metrics = ['violation_rate_mean', 'intervention_rate_mean', 'expected_cost_usd_mean']
available = [c for c in metrics if c in df.columns]

pivot = df.pivot_table(
    index='scenario',
    columns='controller',
    values=available,
    aggfunc='first'
)
pivot.columns = ['_'.join(c).strip() for c in pivot.columns.values]
pivot = pivot.round(6)

out = Path('reports/publication/fault_performance_table.csv')
pivot.to_csv(out)
print(f'\nSaved fault-performance table ({pivot.shape[0]} rows × {pivot.shape[1]} cols) to {out}')
print(pivot.to_string())
PY
```

### 4.2 48-Hour Operational Trace (SECOND PRIORITY)

**Target**: `reports/publication/48h_trace.csv` + `paper/assets/figures/fig_48h_trace.pdf`

Step 1 — Build the script (paste into `scripts/generate_48h_trace.py`):

See `07-evaluation-and-audits.md` §6 for the full script listing.

Step 2 — Run:
```bash
python scripts/generate_48h_trace.py \
  --region DE \
  --fault stale_sensor \
  --window 48 \
  --seed 42
```

### 4.3 Latency Table — Lock with Full Percentiles (THIRD PRIORITY)

**Target**: `reports/publication/dc3s_latency_summary.csv` updated with p99 and max.

```bash
# Re-run with more trials and all percentiles
python scripts/benchmark_dc3s_steps.py --n-trials 10000 --percentiles 50,95,99,max

# Then verify and overwrite locked file
python - <<'PY'
import pandas as pd, shutil
df = pd.read_csv('dc3s_latency_summary.csv')  # root-level output
print(df.to_string(index=False))
if 'p99_ms' in df.columns and 'max_ms' in df.columns:
    shutil.copy('dc3s_latency_summary.csv', 'reports/publication/dc3s_latency_summary.csv')
    print('Locked to reports/publication/dc3s_latency_summary.csv')
else:
    print('WARNING: p99 or max column missing — check benchmark_dc3s_steps.py output')
PY
```

### 4.4 HIL Evidence Package (FOURTH PRIORITY — software mode)

```bash
# Start API service
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 &

# Run closed-loop simulation (software HIL)
python iot/simulator/run_closed_loop.py \
  --api-url http://localhost:8000 \
  --scenario stale_sensor \
  --horizon 48 \
  --seed 42 \
  --output-dir reports/hil/

# Stop API
kill %1
```

---

## 5. Paper Build

### Quick paper compile

```bash
cd <repo-root>/paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
# Output: paper/paper.pdf
```

### Full refresh (sync assets + compile)

```bash
cd <repo-root>
make paper-refresh
# Runs: paper-assets + paper-compile
```

### Sync paper assets from locked reports

```bash
make paper-assets
# Copies locked CSVs/JSONs from reports/publication/ to paper/assets/
```

### Verify paper claims

```bash
make paper-verify
# Checks that paper/metrics_manifest.json matches reports/publication/
```

### Freeze paper (lock all assets)

```bash
make paper-freeze
# Freezes paper/assets/data/metrics_snapshot.json
```

### Build publication artifact package

```bash
make publication-artifact
# Outputs: dist/orius_publication_artifact.zip
# Contains: source zip, paper PDF, locked tables, figures
```

**Theorem alignment.** For artifact zips that bundle thesis, paper, and proofs (e.g., agent-generated ORIUS artifact packages), include `orius-plan/THEOREM_REGISTER_MAPPING.md` so theorem labels align with the canonical battery-8 register. See `orius-plan/theorem_to_evidence_map.md` for full evidence mapping.

### Analyze agent artifact zip

Before accepting agent-generated content, compare the artifact zip against the repo thesis:

```bash
make analyze-artifact
# Or: python scripts/analyze_agent_artifact.py [path-to-zip]
```

This extracts theorem names from `ORIUS_COMPLETE_MATHEMATICAL_PROOFS.md` in the zip, compares against the canonical battery-8 (Appendix M), and reports mismatches and missing alignment. **Source of truth:** repo thesis. See `orius-plan/SOURCE_OF_TRUTH_POLICY.md`.

---

## 6. Extending the ORIUS Framework

### 6.1 Adding a New Fault Type

```python
# src/orius/cpsbench_iot/scenarios.py

# Step 1: Add fault column name
FAULT_COLUMNS = (
    "dropout", "delay_jitter", "out_of_order",
    "spikes", "stale_sensor", "covariate_drift",
    "label_drift",
    "MY_NEW_FAULT",  # add here
)

# Step 2: Define injection function
def _apply_my_new_fault(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Applies my_new_fault injection to the dataframe."""
    # Modify df['value'] or df['timestamp'] as appropriate
    df['MY_NEW_FAULT'] = 0.0
    # Set fault active rows
    fault_mask = rng.random(len(df)) < 0.1  # 10% fault rate
    df.loc[fault_mask, 'MY_NEW_FAULT'] = 1.0
    # Apply the fault effect
    df.loc[fault_mask, 'value'] = ...
    return df

# Step 3: Add to generate_episode() dispatcher
def generate_episode(scenario: str, horizon: int, seed: int) -> EpisodeArtifacts:
    ...
    if scenario == 'my_new_fault':
        x_obs = _apply_my_new_fault(x_obs, rng)
    ...
```

After adding:
```bash
# Re-run CPSBench to include the new fault
python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml
```

### 6.2 Adding a New Forecast Model

```python
# src/orius/forecasting/dl_mymodel.py

class MyModelForecaster:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Train your model
        ...

    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        # Return point forecasts
        ...
```

Register in `forecasting/train.py`:
```python
MODEL_REGISTRY = {
    'gbm': GBMForecaster,
    'lstm': LSTMForecaster,
    ...,
    'mymodel': MyModelForecaster,  # add here
}
```

Add config to `configs/forecast.yaml`:
```yaml
models:
  - mymodel
mymodel:
  # model-specific hyperparameters
```

### 6.3 Adding a New Region

```bash
# Step 1: Copy config template
cp configs/train_forecast_template.yaml configs/train_forecast_<region>.yaml

# Step 2: Edit region config
# - Set zone_id, data_path, target_cols, seeds

# Step 3: Create feature builder (if needed)
# src/orius/data_pipeline/build_features_<region>.py

# Step 4: Download data
python src/orius/data_pipeline/download_<region>.py

# Step 5: Train models
python scripts/train_dataset.py --config configs/train_forecast_<region>.yaml

# Step 6: Run CPSBench for the new region
python scripts/run_cpsbench.py \
  --config configs/cpsbench_r1_severity.yaml \
  --region <region>

# Step 7: Update impact comparison
python - <<'PY'
import json
with open('reports/impact_comparison.json') as f:
    data = json.load(f)
data['<region>'] = {
    'cost_savings_pct': ...,
    'carbon_reduction_pct': ...,
    'peak_shaving_pct': ...,
}
with open('reports/impact_comparison.json', 'w') as f:
    json.dump(data, f, indent=2)
PY
```

---

## 7. CI / Regression Checks

### Quick test suite (runs in <30s)

```bash
make test-quick
# pytest -q -m "not slow and not integration" --no-cov
```

### Full test suite

```bash
make test
# pytest -q --no-cov
```

### Test with coverage (target ≥ 80%)

```bash
make test-cov
# pytest (reads .coveragerc)
```

### Integration tests only

```bash
pytest -q -m "integration" --no-cov
```

### DC3S-specific tests

```bash
pytest -q tests/test_dc3s*.py tests/test_rac_cert*.py tests/test_shield*.py --no-cov
```

### CPSBench smoke test

```bash
pytest -q tests/test_cpsbench_smoke.py --no-cov
```

### Pre-release lint check

```bash
make lint-release
# ruff check on release-critical files (F401, F841)
```

---

## 8. Debugging Guide

### DC3S step returns no intervention but TSVR is non-zero

Check:
1. Is `w_t` being computed correctly? → `print(compute_reliability(event, last_event, cfg))`
2. Is the shield projecting correctly? → Check `shield.mode = projection` in `dc3s.yaml`
3. Is the uncertainty set being inflated? → Check `k_quality > 0` in config
4. Is the plant using the correct SOC bounds? → Check `min_soc_mwh`, `max_soc_mwh` in plant config

### Coverage below 90%

Check:
1. Did the time-aware split leak future data? → Run `make leakage-audit`
2. Is qhat computed on the calibration set (not training set)? → Review `forecasting/uncertainty/cqr.py`
3. Has distribution shifted? → Run `make monitor` to check drift

### Audit DB not updating

Check:
1. Does `data/audit/` exist? → `mkdir -p data/audit/`
2. Is the DuckDB path correct? → `configs/dc3s.yaml` → `dc3s.audit.duckdb_path`
3. Is `store_certificate()` being called? → Check `dc3s/__init__.py` pipeline exit

### Paper fails to compile

Check:
1. Is TeX installed? → `pdflatex --version`
2. Are all figure files present? → `make figure-inventory-audit`
3. Are all table `.tex` files generated? → `ls paper/assets/tables/generated/`
4. Re-generate paper assets: → `make paper-assets`

---

## 9. Observability Stack (Optional)

For production-like monitoring with Prometheus + Grafana:

```bash
# Start observability stack
make observability
# or:
docker compose -f docker/docker-compose.full.yml up -d

# Access:
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
# AlertManager: http://localhost:9093

# Stop
make down-observability
```

Metrics exposed by:
- `src/orius/monitoring/prometheus_metrics.py` — DC3S metrics (w_t, inflation, interventions, TSVR)
- `services/api/main.py` — API request metrics

---

## 10. SOP for New Contributors

Follow this order when onboarding to the ORIUS battery system:

1. Read `orius-plan/00-orientation.md` — understand what the system claims and where the files are
2. Read `orius-plan/01-codex-plan-extracted.md` — this is the extracted
   implementation plan; use it to understand what is done vs what is missing
3. Do §1 (environment setup) and §2 (verify locked results) above
4. Run `make dc3s-demo` and inspect the output
5. Read `orius-plan/04-core-apis.md` — understand the 12 runtime objects
6. Read `orius-plan/06-certificates-and-forecasting.md` — understand the certificate chain
7. Run `make cpsbench` and verify the 0% TSVR result for DC3S
8. Work on any item from the Priority-1 missing list in §4 above

---

*Next: see `09-framework-roadmap.md` for the gap audit, extension plans, and final output checklist.*
