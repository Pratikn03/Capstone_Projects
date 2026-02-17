.PHONY: setup lint test api dashboard frontend frontend-build pipeline data train production reports monitor release_check release_check_full extract-data shap-importance stat-tests train-us reports-us verify-training cv-eval ablations stats-tables verify-novelty robustness-analysis train-dataset train-all

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install

lint:
	python -m compileall src services scripts

test:
	pytest -q

# Training verification
verify-training:
	python scripts/verify_training_outputs.py --verbose

# Run CV evaluation (time-series cross-validation)
cv-eval:
	python -m gridpulse.forecasting.train --config configs/train_forecast.yaml --enable-cv

api:
	PYTHONPATH=src python3 -m uvicorn services.api.main:app --reload --port 8000

dashboard: frontend

frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

extract-data:
	python scripts/extract_dashboard_data.py

pipeline:
	python -m gridpulse.pipeline.run --all

data: pipeline

train:
	python -m gridpulse.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target load_mw
	python -m gridpulse.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target wind_mw
	python -m gridpulse.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target solar_mw
	python -m gridpulse.forecasting.train --config configs/train_forecast.yaml

reports:
	python scripts/build_reports.py

reports-us:
	python scripts/build_reports.py --features data/processed/us_eia930/features.parquet --splits data/processed/us_eia930/splits --models-dir artifacts/models_eia930 --reports-dir reports/eia930

train-us:
	python -m gridpulse.forecasting.train --config configs/train_forecast_eia930.yaml

shap-importance:
	python scripts/shap_importance.py

stat-tests:
	python scripts/statistical_tests.py

monitor:
	python scripts/run_monitoring.py

release_check:
	bash scripts/release_check.sh

release_check_full:
	bash scripts/release_check.sh --full

production: pipeline train extract-data

# ============================================================
# Unified Dataset Training (Recommended)
# ============================================================

# Train a specific dataset: make train-dataset DATASET=DE (or US)
train-dataset:
ifndef DATASET
	$(error DATASET is not set. Usage: make train-dataset DATASET=DE)
endif
	python scripts/train_dataset.py --dataset $(DATASET)

# Train all registered datasets
train-all:
	python scripts/train_dataset.py --all

# Train with hyperparameter tuning: make train-tune DATASET=DE
train-tune:
ifndef DATASET
	$(error DATASET is not set. Usage: make train-tune DATASET=DE)
endif
	python scripts/train_dataset.py --dataset $(DATASET) --tune

# Generate reports only (no training): make reports-dataset DATASET=US
reports-dataset:
ifndef DATASET
	$(error DATASET is not set. Usage: make reports-dataset DATASET=DE)
endif
	python scripts/train_dataset.py --dataset $(DATASET) --reports-only

# List available datasets
list-datasets:
	python scripts/train_dataset.py --list

# ============================================================
# Advanced Features
# ============================================================

# Run ablation study (4 scenarios: Full, No Uncertainty, No Carbon, Forecast Only)
ablations:
	python scripts/run_ablations.py --data data/processed/splits/test.parquet --output reports/ablations --n-runs 5 -v

# Build publication-quality LaTeX tables
stats-tables:
	python scripts/build_stats_tables.py --ablation-dir reports/ablations --output-dir reports/tables

# Verify novelty outputs (ablations, stats, figures)
verify-novelty:
	python scripts/verify_novelty_outputs.py --verbose

# Run robustness analysis with noise perturbations
robustness-analysis:
	python scripts/run_ablations.py --data data/processed/splits/test.parquet --output reports/robustness --n-runs 10 --noise 0.15 -v

# Complete novelty workflow (ablations + tables + verification)
novelty-full: ablations stats-tables verify-novelty
	@echo "✅ Advanced features workflow complete"
