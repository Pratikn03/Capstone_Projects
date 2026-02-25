# GridPulse Scripts

Command-line scripts for training, evaluation, deployment, and maintenance tasks.

## Script Categories

### Training & Model Management
| Script | Purpose |
|--------|---------|
| `train_multi_dataset.py` | Train models on DE/US datasets |
| `run_ablations.py` | Run feature ablation studies |
| `register_models.py` | Register models + conformal artifacts in local registry JSON |
| `promote_model.py` | Promote model to production |
| `retrain_if_needed.py` | Conditional retraining based on drift |
| `train_dataset.py` | Unified dataset training with standard/aggressive profiles |

### Evaluation & Reports
| Script | Purpose |
|--------|---------|
| `build_reports.py` | Generate evaluation reports |
| `build_stats_tables.py` | Statistical significance tables |
| `build_forecast_interval_report.py` | Uncertainty quantification report |
| `statistical_tests.py` | Diebold-Mariano tests |
| `shap_importance.py` | SHAP feature importance analysis |

### Data Processing
| Script | Purpose |
|--------|---------|
| `download_smard_carbon.py` | Download German carbon data |
| `download_emaps_carbon.py` | Download EMAPS carbon factors |
| `download_watttime_moer.py` | Download WattTime MOER data |
| `prepare_emaps_carbon.py` | Process EMAPS data |
| `generate_holidays.py` | Generate holiday features |
| `merge_signals.py` | Merge forecasts with carbon signals |

### Deployment & Operations
| Script | Purpose |
|--------|---------|
| `build_release_bundle.py` | Create deployment bundle |
| `release_check.py` | Pre-release quality gates |
| `final_publish_audit.py` | Full publish orchestrator with GO/NO-GO decision |
| `audit_na_tables.py` | Strict NA-quality audit for parquet/csv/duckdb |
| `audit_leakage.py` | Temporal/feature leakage audit |
| `audit_code_health.py` | Static code-health audit (complexity/except patterns) |
| `audit_git_delta.py` | GitHub old-vs-new delta audit against baseline ref |
| `audit_figure_inventory.py` | Figure/publication inventory + freshness + hash audit |
| `refresh_data_delta.py` | Hybrid delta refresh and dedup checks |
| `backfill_dc3s_typed_columns.py` | Backfill typed DC3S audit columns from payload JSON |
| `run_publish_audit_isolated.sh` | Run full publish audit in isolated `/tmp` workspace clone |
| `check_api_health.py` | API health verification |
| `validate_configs.py` | Configuration validation |
| `validate_dispatch.py` | Dispatch plan validation |

### Visualization
| Script | Purpose |
|--------|---------|
| `generate_publication_figures.py` | Paper-ready figures |
| `generate_arbitrage_plot.py` | Arbitrage opportunity plots |
| `generate_architecture_diagram.py` | System architecture diagram |
| `extract_dashboard_data.py` | Dashboard JSON export |

## Usage

```bash
# Run with Python
python scripts/train_multi_dataset.py

# Or via Makefile targets
make train
make ablations
make stats-tables
```

## Environment

Scripts expect:
- Virtual environment activated (`.venv`)
- Working directory at project root
- Required data in `data/` directory

## Publication Package Runbook

Canonical training configs:
- DE: `configs/train_forecast.yaml`
- US: `configs/train_forecast_eia930.yaml`

Suggested end-to-end sequence:

```bash
python scripts/train_dataset.py --dataset DE --profile aggressive --max-runtime-hours 6
python scripts/train_dataset.py --dataset US --profile aggressive --max-runtime-hours 6
python scripts/run_cpsbench.py --out-dir reports/publication
python scripts/run_ablations.py --dc3s --output reports/publication --scenario drift_combo --horizon 96 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/build_reports.py --features data/processed/features.parquet --splits data/processed/splits --models-dir artifacts/models --reports-dir reports
python scripts/validate_paper_claims.py
python scripts/final_publish_audit.py --config configs/publish_audit.yaml --max-runtime-hours 6 --iot-steps 72 --baseline-ref origin/main
```

## One-Command Reproducibility

Build the full admissions artifact package (Tasks 1-4) with one command:

```bash
make publication-artifact
```

This writes/refreshes:
- `reports/publication/dc3s_main_table.csv`
- `reports/publication/dc3s_fault_breakdown.csv`
- `reports/publication/fig_true_soc_violation_vs_dropout.png`
- `reports/publication/fig_true_soc_severity_p95_vs_dropout.png`
- `reports/publication/table2_ablations.csv`
- `reports/publication/stats_summary.json`
- `reports/publication/cqr_group_coverage.csv`
- `reports/publication/table3_group_coverage.csv` (compat alias)
- `reports/publication/fig_coverage_width_tradeoff.png`
- `reports/publication/fig_coverage_width.png` (compat alias)
- `reports/publication/transfer_stress.csv`
- `reports/publication/table5_transfer.csv` (compat alias)
- `reports/publication/fig_transfer_coverage.png`
