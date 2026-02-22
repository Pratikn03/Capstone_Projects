# GridPulse Evaluation Framework

## Overview

This document describes the comprehensive evaluation framework used to assess GridPulse forecasting accuracy, optimization effectiveness, and decision impact. All metrics are computed on held-out test data using time-based splits to prevent data leakage.

---

## 1. Forecasting Metrics

### Primary Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **RMSE** | √(Σ(yᵢ - ŷᵢ)² / n) | Overall error magnitude |
| **MAE** | Σ\|yᵢ - ŷᵢ\| / n | Robust to outliers |
| **sMAPE** | 200% × Σ\|yᵢ - ŷᵢ\| / (\|yᵢ\| + \|ŷᵢ\|) / n | Symmetric percentage error |
| **R²** | 1 - SS_res / SS_tot | Explained variance |
| **MAPE** | 100% × Σ\|yᵢ - ŷᵢ\| / yᵢ / n | Percentage error |

### Solar-Specific Metrics

Standard MAPE is undefined when actual values are zero (nighttime for solar). We use:

**Daylight MAPE:** Computed only for hours where actual solar generation > 0.

```python
def daylight_mape(y_true, y_pred):
    """MAPE computed only during daylight hours."""
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
```

### Per-Horizon Metrics

Forecast error typically increases with horizon. We decompose metrics by forecast hour:

| Horizon | RMSE (MW) | MAE (MW) | Interpretation |
|---------|-----------|----------|----------------|
| h=1 | 125.3 | 89.2 | Near-term (most accurate) |
| h=6 | 152.1 | 108.4 | Short-term |
| h=12 | 178.9 | 132.1 | Medium-term |
| h=24 | 198.4 | 142.3 | Day-ahead (least accurate) |

This decomposition identifies when forecast quality degrades and informs operational decisions.

---

## 2. Uncertainty Quantification Metrics

### Prediction Interval Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **PICP** | Fraction of true values within [lower, upper] | ≥ (1 - α) |
| **MPIW** | Mean width of prediction intervals | Minimize (while maintaining PICP) |
| **NMPIW** | MPIW / range(y_true) | Normalized width for comparison |

### Coverage Analysis

```
PICP = (1/n) × Σ I(lower_i ≤ y_true_i ≤ upper_i)
```

Where I(·) is the indicator function.

**Interpretation:**
- PICP > (1-α): Intervals are conservative (too wide)
- PICP < (1-α): Intervals are overconfident (too narrow)
- PICP ≈ (1-α): Properly calibrated

### Per-Horizon Coverage

| Horizon | PICP (%) | MPIW (MW) |
|---------|----------|-----------|
| h=1 | 91.5 | 198.3 |
| h=12 | 89.2 | 245.7 |
| h=24 | 88.7 | 289.4 |

Coverage degradation at longer horizons indicates increasing uncertainty.

---

## 3. Statistical Significance Testing

### Diebold-Mariano Test

Compares forecast accuracy between two models:

```
DM = d̄ / σ̂(d̄)
```

Where d = loss(model_A) - loss(model_B)

| Comparison | DM Stat | p-value | Interpretation |
|------------|---------|---------|----------------|
| GBM vs LSTM | -9.42 | <0.001 | GBM significantly better |
| GBM vs TCN | -7.46 | <0.001 | GBM significantly better |
| GBM vs Persistence | -12.31 | <0.001 | GBM significantly better |

### Bootstrap Confidence Intervals

For metric M, compute:
1. Resample (with replacement) test set B=10,000 times
2. Compute M on each bootstrap sample
3. Report 2.5th and 97.5th percentiles as 95% CI

Example: RMSE = 271.2 MW, 95% CI [265.8, 276.4]

### Effect Size (Cohen's d)

```
d = (μ_A - μ_B) / σ_pooled
```

| d | Interpretation |
|---|----------------|
| |d| < 0.2 | Negligible |
| 0.2 ≤ |d| < 0.5 | Small |
| 0.5 ≤ |d| < 0.8 | Medium |
| |d| ≥ 0.8 | Large |

---

## 4. Backtesting Framework

### Walk-Forward Evaluation

```
Time →
──────────────────────────────────────────
[Train_1    ][Test_1]
[Train_2        ][Test_2]
[Train_3            ][Test_3]
[Train_4                ][Test_4]
[Train_5                    ][Test_5]
──────────────────────────────────────────
```

**Configuration:**
```yaml
backtest:
  n_folds: 5
  train_size_pct: 0.7
  test_size_hours: 720  # 30 days
  step_hours: 720       # Non-overlapping test windows
```

### Multi-Horizon Backtest

Evaluate at multiple forecast horizons simultaneously:

```bash
python scripts/build_reports.py --backtest-horizons 1,6,12,24
```

Output: `reports/multi_horizon_backtest.json`

---

## 5. Impact Evaluation (Level-4 Decision Metrics)

### Baseline Policies

**Grid-Only Baseline:**
- No battery storage used
- All demand met from grid import
- Represents do-nothing scenario

**Naive Battery Baseline:**
- Charge during overnight off-peak (00:00-06:00)
- Discharge during evening peak (17:00-21:00)
- Fixed schedule regardless of forecast

### Impact Metrics

| Metric | Formula | Definition |
|--------|---------|------------|
| **Cost Savings %** | (C_base - C_opt) / C_base × 100 | Reduction in total energy cost |
| **Carbon Reduction %** | (E_base - E_opt) / E_base × 100 | Reduction in CO₂ emissions |
| **Peak Shaving %** | (P_base_max - P_opt_max) / P_base_max × 100 | Reduction in peak grid import |

### Current Results (Frozen Run 20260217_165756)

| Dataset | Cost Savings | Carbon Reduction | Peak Shaving |
|---------|--------------|------------------|--------------|
| Germany | **7.11%** | 0.30% | 6.13% |
| USA | 0.11% | 0.13% | 0.00% |

**Note:** USA results show minimal improvement. This is an honest limitation, likely due to:
- Flat price proxy (no real TOU tariff)
- Different grid characteristics
- Battery sizing mismatch

---

## 6. Stochastic Value Metrics

### Expected Value of Perfect Information (EVPI)

Measures cost gap between forecast-based and oracle (perfect information) dispatch:

```
EVPI = Cost(forecast-based dispatch, actual load)
     - Cost(oracle dispatch, actual load)
```

- EVPI ≈ 0: Forecast is nearly perfect
- EVPI > 0: Perfect information would save EVPI euros

### Value of Stochastic Solution (VSS)

Measures benefit of using robust optimization vs deterministic:

```
VSS = Cost(deterministic dispatch, actual load)
    - Cost(robust dispatch, actual load)
```

- VSS > 0: Robust optimization outperforms deterministic
- VSS = 0: No benefit from uncertainty modeling
- VSS < 0: Deterministic performs better (unusual)

### Current VSS Results

| Dataset | VSS (€) | Interpretation |
|---------|---------|----------------|
| Germany | **2,708.61** | Robust optimization provides clear value |
| USA | 0.00 | No benefit observed (requires investigation) |

---

## 7. Ablation Study Framework

### Scenarios

| Scenario | Uncertainty | Carbon | Optimization |
|----------|-------------|--------|--------------|
| **Full System** | ✓ | ✓ | ✓ |
| No Uncertainty | ✗ | ✓ | ✓ |
| No Carbon | ✓ | ✗ | ✓ |
| Forecast Only | ✗ | ✗ | ✗ |

### Running Ablations

```bash
python scripts/run_ablations.py \
    --data data/processed/splits/test.parquet \
    --output reports/ablations \
    --n-runs 10 \
    --noise 0.10
```

### Output Files

- `ablation_results.csv`: Raw results per scenario × run
- `ablation_summary.csv`: Aggregated with bootstrap CI
- `ablation_stats.json`: Pairwise statistical comparisons
- `ablation_comparison.png`: Publication figure

---

## 8. Price and Carbon Signal Requirements

### Impact of Missing Signals

| Scenario | Expected Cost Savings | Expected Carbon Reduction |
|----------|----------------------|--------------------------|
| Real price + real carbon | 5-15% | 1-5% |
| Price only | 3-10% | ~0% |
| Carbon only | ~0% | 1-5% |
| Neither (constant) | **~0%** | **~0%** |

**This is intentional design:** We do not inflate impact metrics when realistic signals are unavailable.

### Signal Sources

| Signal | Germany | USA |
|--------|---------|-----|
| Price | OPSD day-ahead (€/MWh) | Time-of-use proxy ($/MWh) |
| Carbon | SMARD generation mix | EPA eGRID factors |

To use real carbon data:
```bash
python scripts/download_smard_carbon.py --out data/raw/
python scripts/merge_signals.py --price --carbon
```

---

## 9. Reports Generated

| Report | Path | Content |
|--------|------|---------|
| Formal Evaluation | `reports/formal_evaluation_report.md` | All model metrics |
| Impact Summary | `reports/impact_summary.csv` | Cost/carbon/peak metrics |
| Model Cards | `reports/model_cards/` | Per-model documentation |
| Backtest | `reports/walk_forward_report.json` | Temporal decomposition |
| Multi-Horizon | `reports/multi_horizon_backtest.json` | Per-horizon metrics |
| Figures | `reports/figures/` | 19 publication plots |

### Updating README

After generating reports:
```bash
python scripts/update_readme_impact.py
```

This keeps README tables consistent with latest metrics.

---

## 10. Validation Checklist

Before releasing results:

- [ ] All metrics computed on test split only
- [ ] No future information leakage
- [ ] Statistical significance tests passed
- [ ] Confidence intervals computed
- [ ] Ablation study demonstrates component value
- [ ] Impact metrics have realistic signals
- [ ] Figures are publication-quality (300 DPI)
- [ ] Results reproducible with fixed seeds

---

## 11. CPSBench-IoT Evaluation Path

The CPSBench-IoT harness extends evaluation beyond static offline metrics by injecting telemetry faults and drift, then measuring control safety and trace completeness.

### Run Commands

```bash
make cpsbench
```

This executes `python scripts/run_cpsbench.py` with default suite:

- Scenarios: `nominal`, `dropout`, `delay_jitter`, `out_of_order`, `spikes`, `drift_combo`
- Seeds: `[11, 22, 33, 44, 55]`

### CPSBench Metrics

Forecast metrics:

- `MAE`, `RMSE`
- `PICP_90`, `PICP_95`
- `mean_interval_width`

Control metrics:

- `violation_rate`
- `violation_severity`
- `recovery_time`
- `intervention_rate`

Trace metrics:

- `certificate_presence_rate`
- `certificate_missing_fields`

### CPSBench Artifacts

- `reports/publication/dc3s_main_table.csv`
- `reports/publication/dc3s_fault_breakdown.csv`
- `reports/publication/calibration_plot.png`
- `reports/publication/violation_vs_cost_curve.png`
- `reports/publication/dc3s_run_summary.json`

Determinism note:

- fixed seeds + deterministic row ordering + stable float formatting (`%.6f`) are used for reproducible artifact diffs.

---

## 12. IoT Closed-Loop Validation Path

Run command:

```bash
make iot-sim
```

This runs `python iot/simulator/run_closed_loop.py` and exercises:

1. `POST /iot/telemetry`
2. `POST /dc3s/step` with `enqueue_iot=true`
3. `GET /iot/command/next`
4. command apply via simulator driver
5. `POST /iot/ack`
6. `GET /iot/audit/{command_id}`

All `/iot/*` calls are authenticated with scoped API keys (`X-GridPulse-Key`).
Queue timeout defaults to 30 seconds; expired commands activate per-device hold until reset via `POST /iot/control/reset-hold`.

Expected nominal behavior:

- zero safety violations,
- non-zero intervention count when shield repairs unsafe proposals,
- certificate completeness rate near `1.0`.

Quick API sanity command:

```bash
make dc3s-demo
```

This executes one `/dc3s/step` request and validates audit retrieval using the returned `command_id`.

For real-device shadow pilot (HTTP gateway):

```bash
export GRIDPULSE_IOT_API_KEY='<gridpulse_rw_key>'
python iot/edge_agent/run_agent.py --config configs/iot.yaml --mode shadow --iterations 24
```

Shadow ACK semantics:

- `status=acked`
- `payload.shadow_mode=true`
- `payload.applied=false`
