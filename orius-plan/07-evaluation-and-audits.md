# ORIUS Battery Framework — Phase 7: Evaluation, Fault Stress & Audits

**Status**: Locked results documented. Missing outputs specified with exact run commands.

---

## 1. All Locked Result Locations

### Primary locked results (do not overwrite)

| File | Content | Status |
|------|---------|--------|
| `reports/impact_summary.csv` | DE dispatch savings: cost, carbon, peak shaving | **LOCKED** |
| `reports/publication/dc3s_main_table_ci.csv` | DC3S main results with CIs: TSVR, IR, cost — all 6 scenarios × 5 controllers | **LOCKED** |
| `reports/publication/dc3s_ablation_table.csv` | DC3S ablation breakdown (no-wt, no-drift, linear, kappa) | **LOCKED** |
| `reports/publication/dc3s_latency_summary.csv` | DC3S per-step latency (mean, p95 only — p99 missing) | **LOCKED (partial)** |
| `reports/publication/reliability_group_coverage.csv` | Reliability-stratified CQR group coverage (10 bins) | **LOCKED** |
| `reports/publication/cpsbench_merged_sweep.csv` | Full CPSBench sweep merged output | **LOCKED** |
| `reports/publication/cost_safety_pareto.csv` | Cost/safety Pareto frontier | **LOCKED** |
| `reports/publication/cross_region_transfer.csv` | Cross-region transfer coverage | **LOCKED** |
| `reports/publication/cqr_group_coverage.csv` | CQR group coverage (paper Table 3) | **LOCKED** |
| `reports/publication/table3_group_coverage.csv` | Paper Table 3 | **LOCKED** |
| `reports/publication/table4_region_compare.csv` | Paper Table 4 | **LOCKED** |
| `reports/publication/table5_transfer.csv` | Paper Table 5 | **LOCKED** |
| `reports/walk_forward_report.json` | Walk-forward backtest results | **LOCKED** |
| `reports/multi_horizon_backtest.json` | Multi-horizon (1h–24h) results | **LOCKED** |
| `reports/publish/reproducibility_lock.json` | Reproducibility anchor | **LOCKED** |

### Locked metrics to never overwrite

| Metric | Value | Source row |
|--------|-------|-----------|
| Deterministic LP TSVR (nominal) | 3.93% | `dc3s_main_table_ci.csv`: `nominal,deterministic_lp,violation_rate_mean=0.039286` |
| DC3S wrapped TSVR (all scenarios) | 0.00% | `dc3s_main_table_ci.csv`: `*,dc3s_wrapped,violation_rate_mean=0.0` |
| DE cost savings | 12.13% (from `impact_summary.csv`) | Note: this is gross savings vs baseline; 7.11% is net — confirm which is in paper |
| Reliability group coverage (all bins) | 0.90 PICP | `reliability_group_coverage.csv` all bins |

---

## 2. DC3S Main Results Summary

From `reports/publication/dc3s_main_table_ci.csv` — key rows:

| Scenario | Controller | TSVR (mean) | IR (mean) | PICP-90 |
|----------|-----------|-------------|-----------|---------|
| nominal | deterministic_lp | 3.93% | 0.0% | 1.00 |
| nominal | robust_fixed_interval | 22.14% | 0.0% | 1.00 |
| nominal | dc3s_wrapped | 0.00% | 0.0% | 1.00 |
| dropout | deterministic_lp | 3.69% | 0.0% | 0.927 |
| dropout | dc3s_wrapped | 0.00% | 0.0% | 0.927 |
| dropout | dc3s_ftit | 0.00% | 2.62% | 0.937 |
| drift_combo | deterministic_lp | 4.40% | 0.0% | 0.425 |
| drift_combo | robust_fixed_interval | 34.29% | 0.0% | 0.425 |
| drift_combo | dc3s_wrapped | 0.00% | 0.0% | 0.429 |
| drift_combo | dc3s_ftit | 0.00% | 4.88% | 0.932 |
| spikes | deterministic_lp | 3.57% | 0.0% | 0.956 |
| spikes | dc3s_wrapped | 0.00% | 0.0% | 0.956 |

**Key finding**: DC3S achieves 0% TSVR across all fault scenarios. Quality-ignorant controllers (det-LP, robust-fixed) have non-zero TSVR.

---

## 3. DC3S Ablation Results

From `reports/publication/dc3s_ablation_table.csv` — drift_combo scenario:

| Policy | PICP-90 | IR | TSVR | Mean width |
|--------|---------|-----|------|-----------|
| dc3s_no_wt | 0.898 | 1.46% | 0.0% | 14,747 |
| dc3s_no_drift | 0.923 | 1.46% | 0.0% | 19,591 |
| dc3s_linear | 0.923 | 1.46% | 0.0% | 19,812 |
| dc3s_kappa | 0.977 | 1.46% | 0.0% | 25,538 |

All ablation variants maintain 0% TSVR but with different interval widths — showing the tradeoff between coverage and useful work.

---

## 4. CPSBench Run Procedure

### Full CPSBench sweep

```bash
cd <repo-root>
source .venv/bin/activate

# Full battery track sweep (all scenarios, all controllers)
python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml

# Or via Makefile
make cpsbench
```

**Output files**:
- `reports/publication/cpsbench_merged_sweep.csv` — raw results (all scenarios × controllers)
- `reports/publication/dc3s_main_table_ci.csv` — summary with CIs (5 seeds × scenarios)

### Ablation sweep

```bash
python scripts/run_dc3s_ablations_cpsbench.py
# Output: reports/publication/dc3s_ablation_table.csv
```

### Sensitivity sweep (violation rate vs dropout rate)

```bash
python scripts/run_sensitivity_sweeps.py
# Output: figures for paper fig03/fig04
```

---

## 5. Fault-Performance Table — Target Specification

**Status**: MISSING — must generate before manuscript lock.

**Target file**: `reports/publication/fault_performance_table.csv`

### Table structure (7 faults × 4 controllers = 28 core cells)

| Scenario | det-LP | robust-LP | CVaR | DC3S |
|----------|--------|-----------|------|------|
| nominal | TSVR, IR, cost | ... | ... | 0%, IR, cost |
| dropout | ... | ... | ... | ... |
| delay_jitter | ... | ... | ... | ... |
| out_of_order | ... | ... | ... | ... |
| spikes | ... | ... | ... | ... |
| drift_combo | ... | ... | ... | ... |
| stale_sensor | ... | ... | ... | ... |

Metrics per cell: `violation_rate_mean`, `intervention_rate_mean`, `expected_cost_usd_mean`, `severity_p95` (if available).

### Generation command

```bash
# Step 1: Run full CPSBench sweep (includes stale_sensor if configured)
python scripts/run_cpsbench.py --config configs/cpsbench_r1_severity.yaml

# Step 2: Pivot into fault-performance table
python - <<'PY'
import pandas as pd

df = pd.read_csv('reports/publication/cpsbench_merged_sweep.csv')
cols = ['violation_rate_mean', 'intervention_rate_mean', 'expected_cost_usd_mean']
pivot = df.pivot_table(
    index='scenario',
    columns='controller',
    values=cols,
    aggfunc='first'
)
# Flatten multi-level columns
pivot.columns = ['_'.join(c).strip() for c in pivot.columns.values]
pivot.to_csv('reports/publication/fault_performance_table.csv')
print(pivot.to_string())
print('\nSaved to reports/publication/fault_performance_table.csv')
PY
```

---

## 6. 48-Hour Operational Trace — Target Specification

**Status**: MISSING — script must be built.

**Target file**: `reports/publication/48h_trace.csv` + figure `paper/assets/figures/fig_48h_trace.pdf`

### Required columns

| Column | Description |
|--------|-------------|
| `timestamp` | Hourly timestamps |
| `observed_soc_mwh` | SOC as seen by controller (`o_t`) |
| `true_soc_mwh` | True physical SOC (`x_t`) |
| `soc_divergence_mwh` | `|true_soc − observed_soc|` |
| `w_t` | Observation quality score |
| `interval_lower_mwh` | RAC-Cert lower bound |
| `interval_upper_mwh` | RAC-Cert upper bound |
| `interval_width_mwh` | `upper − lower` |
| `candidate_action_mw` | `a_t^*` from optimizer |
| `safe_action_mw` | `a_t^safe` after shield |
| `intervened` | Boolean: was action repaired? |
| `fault_active` | Boolean: is fault scenario active? |
| `price_per_mwh` | Optional: price signal |
| `reliability_w` | Same as `w_t` |
| `drift_flag` | `d_t` Page-Hinkley flag |

### Script to build: `scripts/generate_48h_trace.py`

```python
#!/usr/bin/env python3
"""Generate 48-hour operational trace for a specific fault episode."""
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def generate_48h_trace(region: str, fault: str, window: int, seed: int):
    from orius.utils.config import load_config
    from orius.cpsbench_iot.scenarios import generate_episode
    from orius.cpsbench_iot.plant import BatteryPlant
    from orius.cpsbench_iot.runner import CPSBenchRunner

    cfg = load_config('configs/dc3s.yaml')
    cpsbench_cfg = load_config('configs/cpsbench_r1_severity.yaml')

    # Generate episode with specified fault
    artifacts = generate_episode(scenario=fault, horizon=window, seed=seed)

    # Run DC3S pipeline on each step
    runner = CPSBenchRunner(config=cpsbench_cfg)
    trace = runner.run_single_episode(
        x_obs=artifacts.x_obs,
        x_true=artifacts.x_true,
        scenario=fault,
        controller='dc3s_wrapped',
    )

    # Save trace
    out_path = Path('reports/publication/48h_trace.csv')
    trace.to_csv(out_path, index=False)
    print(f'Saved {len(trace)} rows to {out_path}')
    return trace

def plot_48h_trace(trace: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: SOC trajectories
    axes[0].plot(trace['timestamp'], trace['true_soc_mwh'], label='True SOC', linewidth=2)
    axes[0].plot(trace['timestamp'], trace['observed_soc_mwh'], label='Observed SOC', linestyle='--')
    axes[0].fill_between(trace['timestamp'],
                          trace['interval_lower_mwh'], trace['interval_upper_mwh'],
                          alpha=0.2, label='RAC-Cert interval')
    # Shade fault-active periods
    fault_mask = trace['fault_active'].astype(bool)
    for _, group in trace[fault_mask].groupby((~fault_mask).cumsum()):
        axes[0].axvspan(group['timestamp'].iloc[0], group['timestamp'].iloc[-1], alpha=0.1, color='red')
    axes[0].set_ylabel('SOC (MWh)')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Battery SOC Trajectories with RAC-Cert Interval')

    # Panel 2: Observation quality
    axes[1].plot(trace['timestamp'], trace['w_t'], color='orange')
    axes[1].axhline(0.05, color='red', linestyle=':', label='w_min')
    axes[1].set_ylabel('w_t quality')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    # Panel 3: Candidate vs safe action
    axes[2].plot(trace['timestamp'], trace['candidate_action_mw'], label='a_t*', alpha=0.7)
    axes[2].plot(trace['timestamp'], trace['safe_action_mw'], label='a_t safe', linewidth=2)
    intervened = trace['intervened'].astype(bool)
    axes[2].scatter(trace.loc[intervened, 'timestamp'],
                     trace.loc[intervened, 'safe_action_mw'],
                     color='red', s=20, zorder=5, label='Intervention')
    axes[2].set_ylabel('Action (MW)')
    axes[2].legend()

    # Panel 4: Interval width
    axes[3].plot(trace['timestamp'], trace['interval_width_mwh'], color='purple')
    axes[3].set_ylabel('Interval width (MWh)')
    axes[3].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved figure to {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', default='DE')
    parser.add_argument('--fault', default='stale_sensor')
    parser.add_argument('--window', type=int, default=48)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    trace = generate_48h_trace(args.region, args.fault, args.window, args.seed)
    plot_48h_trace(trace, 'paper/assets/figures/fig_48h_trace.pdf')
```

**To run** (after building the script):
```bash
python scripts/generate_48h_trace.py --region DE --fault stale_sensor --window 48 --seed 42
```

---

## 7. Latency Benchmark

**Status**: PARTIAL — mean and p95 locked; p99 and max missing.

**Current locked values** (`reports/publication/dc3s_latency_summary.csv`):

| Stage | Mean (ms) | P95 (ms) | P99 | Max |
|-------|-----------|----------|-----|-----|
| Reliability scoring (OQE) | 0.0196 | 0.0239 | MISSING | MISSING |
| Drift update (Page-Hinkley) | 0.0004 | 0.0004 | MISSING | MISSING |
| Uncertainty set build | 0.0100 | 0.0132 | MISSING | MISSING |
| Action repair (SAF) | 0.0019 | 0.0020 | MISSING | MISSING |
| Full DC3S step | 0.0329 | 0.0354 | MISSING | MISSING |

**To generate complete table**:
```bash
# Run with extended percentiles
python scripts/benchmark_dc3s_steps.py --percentiles 50,95,99,max --n-trials 10000

# Lock the output
cp dc3s_latency_summary.csv reports/publication/dc3s_latency_summary.csv
```

**Target**: Full table with mean, median, p95, p99, max for all 5 stages.

---

## 8. Subgroup Coverage Audit

**Status**: DONE — `reports/publication/reliability_group_coverage.csv` locked.

```bash
# Re-run to verify (should reproduce same numbers)
python scripts/compute_reliability_group_coverage.py
```

**Additional audit scripts**:
```bash
# Walk-forward and subgroup diagnostics
python - <<'PY'
import os, glob
for f in glob.glob('reports/**/*walk_forward*', recursive=True) + \
         glob.glob('reports/**/*coverage*', recursive=True) + \
         glob.glob('reports/**/*reliability*', recursive=True):
    if os.path.exists(f):
        print(f)
PY
```

---

## 9. HIL Evidence Package Specification

**Status**: MISSING (hardware not run). Software HIL available via simulator.

### Option A: Software HIL (immediate — use simulator)

```bash
# Run closed-loop simulation against in-process FastAPI
python iot/simulator/run_closed_loop.py

# This exercises:
# - iot/edge_agent/drivers/sim.py (SimBatteryDriver)
# - services/api/routers/dc3s.py (DC3S REST endpoint)
# - iot/edge_agent/agent.py (telemetry send → command receive loop)
```

Outputs: latency measurements, safety outcomes, fault injection results.

### Option B: Real hardware HIL

**Setup requirements** (from app_i: HIL BOM and safety):

| Component | Purpose |
|-----------|---------|
| Battery emulator (e.g., BT2000 or similar) | Protected benchtop battery |
| Modbus TCP adapter | Hardware telemetry path |
| Controller host (this machine) | Runs DC3S pipeline |
| Actuation cut-off relay | Emergency stop hardware |
| Fault injector (network emulator) | Deterministic fault injection |

**Driver**: `iot/edge_agent/drivers/modbus_tcp.py` → `ModbusTCPDriver`

**Minimum HIL evidence package** (required for ch27):
1. Setup diagram (export from `docs/`)
2. Hardware table (BOM)
3. Timing table: telemetry latency, controller latency, actuation latency
4. Real fault-response plot: SOC trajectory under injected fault
5. Safety outcome table: TSVR = 0 under DC3S, TSVR > 0 under det-LP

### HIL run command (simulator mode)
```bash
# Terminal 1: Start the API service
source .venv/bin/activate
uvicorn services.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Run closed-loop simulation
source .venv/bin/activate
python iot/simulator/run_closed_loop.py \
  --api-url http://localhost:8000 \
  --scenario stale_sensor \
  --horizon 48 \
  --seed 42
```

---

## 10. Backtest Procedure

### Walk-forward backtest (primary)

```bash
# DE models
python scripts/train_dataset.py --config configs/forecast.yaml

# US models
python scripts/train_multi_dataset.py --config configs/train_forecast_eia930.yaml

# Output: reports/walk_forward_report.json, reports/eia930/walk_forward_report.json
```

### Multi-horizon backtest

```bash
python - <<'PY'
import json
with open('reports/multi_horizon_backtest.json') as f:
    data = json.load(f)
# Shows RMSE/MAE/MAPE for horizons 1h through 24h
PY
```

---

## 11. Pre-Publication Audit

Run this before any manuscript submission:

```bash
# NA audit (no missing values in locked tables)
make na-audit
# Output: reports/publish/na_audit.csv

# Data leakage audit
make leakage-audit

# Code health audit
make code-health-audit

# Full publication audit
make publish-audit

# Verify paper claims vs locked metrics
make paper-verify

# Check all figures present
make figure-inventory-audit
```

---

## 12. Evaluation Invariants

These must never be broken during any run:

1. **Truth/observed separation**: `BatteryPlant.step()` never clamps SOC — allows violations to be measured
2. **Deterministic seeds**: All CPSBench seeds are fixed (`reports/publish/reproducibility_lock.json`)
3. **Metric schema stability**: `dc3s_main_table_ci.csv` column names are frozen — do not rename
4. **Certificate chain integrity**: `prev_hash` links must be intact — no retroactive modification
5. **Evaluation time order**: test set is always after training set in time — no shuffle

---

*Next: see `08-operations-sop.md` for the complete step-by-step operational procedures.*
