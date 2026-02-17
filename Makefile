.PHONY: setup lint test test-cov test-quick api dashboard frontend frontend-build pipeline data train production reports monitor release_check release_check_full extract-data shap-importance stat-tests train-us reports-us verify-training cv-eval ablations stats-tables verify-novelty robustness-analysis train-dataset train-all k6-load locust-load observability down-observability

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install

lint:
	python -m compileall src services scripts

test:
	pytest -q

# Test with coverage (minimum 80%)
test-cov:
	pytest --cov=gridpulse --cov-report=html --cov-report=term-missing --cov-fail-under=80

# Quick tests (exclude slow markers)
test-quick:
	pytest -q -m "not slow and not integration"

# Integration tests only
test-integration:
	pytest -q -m "integration"

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

# ============================================================
# Load Testing (98/100 Enhancement)
# ============================================================

# Run k6 load tests (requires k6 installed: brew install k6)
k6-load:
	@which k6 > /dev/null || (echo "k6 not installed. Install with: brew install k6" && exit 1)
	k6 run tests/load/k6_load_test.js

# Run k6 with custom VUs and duration
k6-stress:
	@which k6 > /dev/null || (echo "k6 not installed. Install with: brew install k6" && exit 1)
	k6 run --vus 200 --duration 5m tests/load/k6_load_test.js

# Run locust load tests (web UI at http://localhost:8089)
locust-load:
	locust -f tests/load/locustfile.py --host=http://localhost:8000

# Run locust headless (CI mode)
locust-ci:
	locust -f tests/load/locustfile.py --host=http://localhost:8000 --headless -u 50 -r 10 --run-time 2m

# ============================================================
# Observability Stack (95/100 Enhancement)
# ============================================================

# Start full observability stack (Prometheus + Grafana + Alertmanager)
observability:
	docker compose -f docker/docker-compose.full.yml up -d prometheus grafana alertmanager node-exporter cadvisor

# Start complete production stack (API + DB + Kafka + Observability)
stack-up:
	docker compose -f docker/docker-compose.full.yml up -d

# Stop full stack
stack-down:
	docker compose -f docker/docker-compose.full.yml down

# View logs from full stack
stack-logs:
	docker compose -f docker/docker-compose.full.yml logs -f

# View observability dashboards
grafana:
	@echo "Opening Grafana at http://localhost:3001 (admin/gridpulse)"
	open http://localhost:3001 || xdg-open http://localhost:3001 || echo "Visit: http://localhost:3001"

# View Prometheus targets
prometheus:
	@echo "Opening Prometheus at http://localhost:9090"
	open http://localhost:9090 || xdg-open http://localhost:9090 || echo "Visit: http://localhost:9090"

# Run streaming worker (requires Kafka running)
streaming-worker:
	PYTHONPATH=src python -m gridpulse.streaming.worker

# ============================================================
# Advanced Baselines (93/100 Enhancement)
# ============================================================

# Train all advanced baselines (Prophet, N-BEATS, AutoML)
train-baselines:
	PYTHONPATH=src python -c "from gridpulse.forecasting.advanced_baselines import train_all_baselines; train_all_baselines('data/processed/features.parquet')"

# Evaluate baselines against production models
eval-baselines:
	PYTHONPATH=src python -c "from gridpulse.forecasting.advanced_baselines import evaluate_baselines; print(evaluate_baselines('data/processed/splits/test.parquet').to_markdown())"

# ============================================================
# Production Release Workflow
# ============================================================

# Full CI check (coverage + lint + integration)
ci: lint test-cov test-integration
	@echo "✅ CI checks passed"

# Pre-release validation
pre-release: ci release_check
	@echo "✅ Pre-release validation complete"
