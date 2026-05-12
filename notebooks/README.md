# ORIUS Research Notebooks

These notebooks are analysis surfaces over locked artifacts. They are not the source of truth for release claims; the CSV/JSON/TeX/PDF artifacts under `reports/publication/` and the freeze artifacts remain authoritative.

| Notebook | Claim surface | Primary artifacts | Status |
|---|---|---|---|
| `notebooks/01_eda.ipynb` | legacy_battery_eda | `battery data exploration` | active_legacy |
| `notebooks/02_baselines.ipynb` | legacy_battery_forecasting | `battery baselines` | active_legacy |
| `notebooks/03_feature_pipeline.ipynb` | legacy_feature_pipeline | `feature construction` | active_legacy |
| `notebooks/04_train_models.ipynb` | legacy_training | `battery model training` | active_legacy |
| `notebooks/05_inference_intervals.ipynb` | legacy_uncertainty | `interval analysis` | active_legacy |
| `notebooks/06_error_analysis.ipynb` | legacy_error_analysis | `forecast residual analysis` | active_legacy |
| `notebooks/07_production_run.ipynb` | legacy_runbook | `production runbook` | active_legacy |
| `notebooks/08_weather_features.ipynb` | legacy_weather_features | `optional weather features` | active_legacy |
| `notebooks/09_walk_forward_report.ipynb` | legacy_backtest | `walk-forward report` | active_legacy |
| `notebooks/10_optimization_engine.ipynb` | legacy_optimization | `dispatch optimization` | active_legacy |
| `notebooks/11_monitoring_drift.ipynb` | legacy_monitoring | `drift monitoring` | active_legacy |
| `notebooks/12_api_dashboard_smoke_test.ipynb` | legacy_ui_api | `API/dashboard smoke test` | active_legacy |
| `notebooks/13_runbook_end_to_end.ipynb` | legacy_reproducibility | `end-to-end runbook` | active_legacy |
| `notebooks/14_de_us_gap_analysis.ipynb` | battery_data_gap | `DE/US battery data gap` | active |
| `notebooks/15_av_domain_validation.ipynb` | av_domain_validation | `AV runtime validation` | active |
| `notebooks/17_healthcare_domain_validation.ipynb` | healthcare_domain_validation | `Healthcare runtime validation` | active |
| `notebooks/19_universal_theorem_visualization.ipynb` | theorem_visualization | `universal theorem visualization` | active |
| `notebooks/00_orius_research_notebook_index.ipynb` | ORIUS Research Notebook Index | `reports/publication/orius_research_notebook_inventory.csv` | active |
| `notebooks/20_final_release_results_analysis.ipynb` | Final Release Results Analysis | `reports/publication/final_*_for_paper.csv | reports/publication/tbl_final_*.tex` | active |
| `notebooks/21_utility_preserving_safety_analysis.ipynb` | Utility-Preserving Safety Analysis | `reports/publication/utility_preserving_safety_scorecard.csv` | active |
| `notebooks/22_theorem_audit_traceability.ipynb` | Theorem Audit and Traceability | `reports/publication/active_theorem_audit.csv | reports/publication/theorem_result_cards/*.json` | active |
| `notebooks/23_freeze_release_reproducibility_audit.ipynb` | Freeze and Release Reproducibility Audit | `reports/split_training/latest_release_id.txt | reports/predeployment_freeze/<release>/*.json` | active |
| `notebooks/24_publication_package_quality_audit.ipynb` | Publication Package Quality Audit | `reports/publication/tbl_final_*.tex | reports/publication/fig_final_*.png | paper/paper.pdf` | active |
