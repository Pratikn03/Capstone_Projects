# GridPulse Scripts

Command-line scripts for training, evaluation, deployment, and maintenance tasks.

## Script Categories

### Training & Model Management
| Script | Purpose |
|--------|---------|
| `train_multi_dataset.py` | Train models on DE/US datasets |
| `run_ablations.py` | Run feature ablation studies |
| `register_models.py` | Register models in MLflow registry |
| `promote_model.py` | Promote model to production |
| `retrain_if_needed.py` | Conditional retraining based on drift |

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
