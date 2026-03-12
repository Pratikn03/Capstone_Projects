# GridPulse Scripts

Command-line scripts for training, evaluation, publication packaging, and release maintenance.

## Canonical Workflow

The normalized research workflow is anchored to these entrypoints:
- `train_dataset.py`: canonical per-dataset training and candidate-run acceptance
- `run_r1_release.py`: canonical release-family orchestrator (`diagnostic`, `full`, `cpsbench`, `verify`, `promote`)
- `build_baseline_comparison_table.py --release-id <id>`: canonical forecasting-table rebuild for the thesis headline regions
- `build_publication_artifact.py --release-id <id>`: canonical publication artifact build
- `post_training_paper_update.py --release-id <id>`: canonical final paper-freeze flow for one verified release family
- `sync_paper_assets.py --check`: paper-asset freshness and provenance audit
- `validate_paper_claims.py`: hard manuscript claim gate
- `final_publish_audit.py`: final release GO/NO-GO audit

Legacy/internal scripts may still exist for compatibility, but they should not be used as the primary publication workflow.

## Script Categories

### Training & Model Management
| Script | Purpose |
|--------|---------|
| `train_multi_dataset.py` | Legacy/internal multi-dataset helper retained for transition compatibility |
| `run_ablations.py` | Run feature ablation studies |
| `register_models.py` | Register models + conformal artifacts in local registry JSON |
| `promote_model.py` | Promote model to production |
| `retrain_if_needed.py` | Conditional retraining based on drift |
| `train_dataset.py` | Unified dataset training with standard/aggressive profiles |
| `run_r1_release.py` | Release-family orchestration across DE + US balancing authorities |

### Evaluation & Reports
| Script | Purpose |
|--------|---------|
| `build_reports.py` | Generate evaluation reports |
| `build_baseline_comparison_table.py` | Rebuild the promoted six-model forecasting table from one release family |
| `build_publication_artifact.py` | Build a manifest-locked publication package for one `release_id` |
| `compute_reliability_group_coverage.py` | Build reliability-binned conformal coverage artifacts for research analysis |
| `build_stats_tables.py` | Statistical significance tables |
| `build_paper_table_tex.py` | Convert manifest-mapped CSV tables into LaTeX table fragments + token lookup |
| `post_training_paper_update.py` | Freeze thesis/conference PDFs, render review PNGs, and record PDF hashes in the publication manifest |
| `build_forecast_interval_report.py` | Uncertainty quantification report |
| `statistical_tests.py` | Diebold-Mariano tests |
| `shap_importance.py` | SHAP feature importance analysis |
| `update_paper_metrics.py` | Materialize `\\PaperMetric` values from `reports/publication/stats_summary.json` |
| `verify_paper_manifest.py` | Verify manifest paths, section contract, and LaTeX token resolution |

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
| `export_paper_assets.sh` | Export curated paper assets from raw publication outputs |
| `sync_paper_assets.py` | Check paper assets against the locked release manifest and stale-value rules |
| `validate_paper_claims.py` | Validate locked manuscript numbers, run IDs, and claim-matrix status |
| `summarize_chil_run.py` | Summarize IoT/API/certificate outputs into a CHIL-style JSON report |
| `export_pilot_bundle.py` | Export a future-pilot telemetry/decision/ACK/certificate bundle |
| `validate_pilot_bundle.py` | Validate bundle schema, linkage, and timestamp ordering |
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
# Canonical install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
pip install -e .

# Canonical release workflow
export RELEASE_ID=R1_20260312T000000Z
make r1-diagnostic RELEASE_ID=$RELEASE_ID
make r1-full RELEASE_ID=$RELEASE_ID PROFILE=standard
make r1-cpsbench RELEASE_ID=$RELEASE_ID
make r1-verify RELEASE_ID=$RELEASE_ID
make publication-artifact RELEASE_ID=$RELEASE_ID
make paper-sync
```

## Environment

Scripts expect:
- Virtual environment activated (`.venv`)
- Working directory at project root
- Required data in `data/` directory

## Code Quality Lock

Use release-grade static hygiene checks before publication freeze:

```bash
.venv/bin/pip install ruff
make lint-release
```

## Publication Package Runbook

Canonical training configs:
- DE: `configs/train_forecast.yaml`
- US: `configs/train_forecast_eia930.yaml`

Suggested end-to-end sequence:

```bash
export RELEASE_ID=R1_20260312T000000Z
python scripts/run_r1_release.py --stage diagnostic --release-id "$RELEASE_ID"
python scripts/run_r1_release.py --stage full --release-id "$RELEASE_ID" --profile aggressive
python scripts/run_r1_release.py --stage cpsbench --release-id "$RELEASE_ID"
python scripts/run_r1_release.py --stage verify --release-id "$RELEASE_ID"
python scripts/build_publication_artifact.py --release-id "$RELEASE_ID" --out-dir reports/publication
python scripts/post_training_paper_update.py --release-id "$RELEASE_ID"
python scripts/validate_paper_claims.py
python scripts/sync_paper_assets.py --check
python scripts/final_publish_audit.py --config configs/publish_audit.yaml --skip-retrain
```

## One-Command Reproducibility

Build the full admissions artifact package (Tasks 1-4) with one command:

```bash
make publication-artifact RELEASE_ID=R1_20260312T000000Z
```

This writes/refreshes:
- `reports/publication/dc3s_main_table.csv`
- `reports/publication/dc3s_fault_breakdown.csv`
- `reports/publication/fig_true_soc_violation_vs_dropout.png`
- `reports/publication/fig_true_soc_severity_p95_vs_dropout.png`
- `reports/publication/table2_ablations.csv`
- `reports/publication/stats_summary.json`
- `reports/publication/cqr_group_coverage.csv`
- `reports/publication/cqr_calibration_summary.json`
- `reports/publication/table3_group_coverage.csv` (compat alias)
- `reports/publication/fig_cqr_group_coverage.png`
- `reports/publication/rac_cert_summary.json`
- `reports/publication/fig_rac_sensitivity_vs_width.png`
- `reports/publication/fig_coverage_width_tradeoff.png`
- `reports/publication/fig_coverage_width.png` (compat alias)
- `reports/publication/table_cqr_distributional_compare.csv`
- `reports/publication/fig_distributional_vs_cqr.png`
- `reports/publication/ambiguity_calibration_summary.json`
- `reports/publication/transfer_stress.csv`
- `reports/publication/table5_transfer.csv` (compat alias)
- `reports/publication/fig_transfer_coverage.png`
- `reports/publication/cost_safety_pareto.csv`
- `reports/publication/fig_cost_safety_pareto.png`
- `reports/publication/release_manifest.json`

## Paper Asset Refresh Workflow

Manifest-driven paper refresh (run only after a successful `build_publication_artifact.py --release-id ...`):

```bash
make paper-refresh
```

This runs:
1. `scripts/export_paper_assets.sh`
2. `scripts/build_paper_table_tex.py`
3. `scripts/update_paper_metrics.py`
4. `scripts/verify_paper_manifest.py`
5. 2-pass `pdflatex` compile for `paper/paper.tex`

For the final submission freeze, use the release-scoped canonical flow instead of `make paper-refresh`:

```bash
make paper-freeze RELEASE_ID=FINAL_20260312T000000Z
```

This performs the paper-refresh steps, compiles both `paper.tex` and `paper_r1.tex`, renders review PNGs, copies immutable frozen PDFs under `reports/publication/frozen/<RELEASE_ID>/`, and records PDF hashes in `reports/publication/release_manifest.json`.
