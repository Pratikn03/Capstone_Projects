.PHONY: setup lint lint-release test test-cov test-quick api dashboard frontend frontend-build pipeline data train production reports monitor release_check release_check_full extract-data shap-importance stat-tests train-us reports-us verify-training cv-eval ablations stats-tables verify-novelty robustness-analysis train-dataset train-all k6-load locust-load observability down-observability cpsbench dc3s-demo orius-check orius-check-quick framework-proof thesis-pipeline-verify thesis-train thesis-bench thesis-artifacts thesis-manuscript thesis-freeze thesis-full iot-sim refresh-data na-audit leakage-audit code-health-audit git-delta-audit figure-inventory-audit backfill-dc3s publish-audit publish-audit-isolated publication-artifact analyze-artifact av-datasets industrial-datasets healthcare-datasets aerospace-datasets multi-domain-datasets multi-domain-build universal-framework-figure paper-assets paper-verify paper-compile paper-refresh paper-freeze paper2-blackout-benchmark

PYTHON ?= $(if $(wildcard .venv/bin/python3),.venv/bin/python3,python3)
PROFILE ?= standard
PROOF_SEEDS ?= 1
PROOF_HORIZON ?= 24

setup:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && .venv/bin/pip install -r requirements.lock.txt
	cd frontend && npm install

lint:
	PYTHONPYCACHEPREFIX=/tmp/orius_pycache $(PYTHON) -m compileall src services scripts

# Release-focused static hygiene checks (unused imports/locals)
lint-release:
	@if [ -x .venv/bin/ruff ]; then \
		./.venv/bin/ruff check \
			src/orius/dc3s/rac_cert.py \
			src/orius/dc3s/calibration.py \
			src/orius/cpsbench_iot/runner.py \
			scripts/build_publication_artifact.py \
			scripts/run_ablations.py \
			scripts/train_regime_cqr.py \
			tests/test_rac_cert.py \
			tests/test_rac_bounds_integration.py \
			tests/test_publication_rac_required.py \
			tests/test_cpsbench_smoke.py \
			tests/test_run_ablations_dc3s.py \
			--select F401,F841; \
	else \
		echo "ruff not found in .venv. Install with: .venv/bin/pip install ruff"; \
		exit 1; \
	fi

test:
	pytest -q --no-cov

# Test with coverage (minimum 80%)
test-cov:
	pytest

# Quick tests (exclude slow markers)
test-quick:
	pytest -q -m "not slow and not integration" --no-cov

# Integration tests only
test-integration:
	pytest -q -m "integration" --no-cov

# Training verification
verify-training:
	$(PYTHON) scripts/verify_training_outputs.py --verbose

# Run CV evaluation (time-series cross-validation)
cv-eval:
	$(PYTHON) -m orius.forecasting.train --config configs/train_forecast.yaml --enable-cv

api:
	PYTHONPATH=src $(PYTHON) -m uvicorn services.api.main:app --reload --port 8000

dashboard: frontend

frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

extract-data:
	$(PYTHON) scripts/extract_dashboard_data.py

pipeline:
	$(PYTHON) -m orius.pipeline.run --all

data: pipeline

train:
	$(PYTHON) -m orius.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target load_mw
	$(PYTHON) -m orius.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target wind_mw
	$(PYTHON) -m orius.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target solar_mw
	$(PYTHON) -m orius.forecasting.train --config configs/train_forecast.yaml

reports:
	$(PYTHON) scripts/build_reports.py

reports-us:
	$(PYTHON) scripts/build_reports.py --features data/processed/us_eia930/features.parquet --splits data/processed/us_eia930/splits --models-dir artifacts/models_eia930 --reports-dir reports/eia930

train-us:
	$(PYTHON) -m orius.forecasting.train --config configs/train_forecast_eia930.yaml

shap-importance:
	$(PYTHON) scripts/shap_importance.py

stat-tests:
	$(PYTHON) scripts/statistical_tests.py

monitor:
	$(PYTHON) scripts/run_monitoring.py

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
	$(PYTHON) scripts/train_dataset.py --dataset $(DATASET)

# Train all registered datasets
train-all:
	$(PYTHON) scripts/train_dataset.py --all

# Train with hyperparameter tuning: make train-tune DATASET=DE
train-tune:
ifndef DATASET
	$(error DATASET is not set. Usage: make train-tune DATASET=DE)
endif
	$(PYTHON) scripts/train_dataset.py --dataset $(DATASET) --tune

# Generate reports only (no training): make reports-dataset DATASET=US
reports-dataset:
ifndef DATASET
	$(error DATASET is not set. Usage: make reports-dataset DATASET=DE)
endif
	$(PYTHON) scripts/train_dataset.py --dataset $(DATASET) --reports-only

# List available datasets
list-datasets:
	$(PYTHON) scripts/train_dataset.py --list

# ============================================================
# Advanced Features
# ============================================================

# Run ablation study (4 scenarios: Full, No Uncertainty, No Carbon, Forecast Only)
ablations:
	$(PYTHON) scripts/run_ablations.py --data data/processed/splits/test.parquet --output reports/ablations --n-runs 5 -v

# Build publication-quality LaTeX tables
stats-tables:
	$(PYTHON) scripts/build_stats_tables.py --ablation-dir reports/ablations --output-dir reports/tables

# Verify novelty outputs (ablations, stats, figures)
verify-novelty:
	$(PYTHON) scripts/verify_novelty_outputs.py --verbose

# Run robustness analysis with noise perturbations
robustness-analysis:
	$(PYTHON) scripts/run_ablations.py --data data/processed/splits/test.parquet --output reports/robustness --n-runs 10 --noise 0.15 -v

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
	@echo "Opening Grafana at http://localhost:3001 (admin/orius)"
	open http://localhost:3001 || xdg-open http://localhost:3001 || echo "Visit: http://localhost:3001"

# View Prometheus targets
prometheus:
	@echo "Opening Prometheus at http://localhost:9090"
	open http://localhost:9090 || xdg-open http://localhost:9090 || echo "Visit: http://localhost:9090"

# Run streaming worker (requires Kafka running)
streaming-worker:
	PYTHONPATH=src $(PYTHON) -m orius.streaming.worker

# ============================================================
# Advanced Baselines (93/100 Enhancement)
# ============================================================

# Train all advanced baselines (Prophet, N-BEATS, AutoML)
train-baselines:
	PYTHONPATH=src $(PYTHON) -c "from orius.forecasting.advanced_baselines import train_all_baselines; train_all_baselines('data/processed/features.parquet')"

# Evaluate baselines against production models
eval-baselines:
	PYTHONPATH=src $(PYTHON) -c "from orius.forecasting.advanced_baselines import evaluate_baselines; print(evaluate_baselines('data/processed/splits/test.parquet').to_markdown())"

# ============================================================
# Production Release Workflow
# ============================================================

# CPSBench-IoT + closed-loop IoT validation
cpsbench:
	PYTHONPATH=src $(PYTHON) scripts/run_cpsbench.py

publication-artifact:
ifndef RELEASE_ID
	$(error RELEASE_ID is not set. Usage: make publication-artifact RELEASE_ID=R1_20260312T000000Z)
endif
	$(PYTHON) scripts/build_publication_artifact.py --release-id $(RELEASE_ID) --out-dir reports/publication --horizon 96 --seeds 0 1 2 3 4 5 6 7 8 9

paper-assets:
	bash scripts/export_paper_assets.sh
	PYTHONPATH=src $(PYTHON) scripts/build_paper_table_tex.py
	PYTHONPATH=src $(PYTHON) scripts/update_paper_metrics.py

paper-verify:
	PYTHONPATH=src $(PYTHON) scripts/verify_paper_manifest.py
	PYTHONPATH=src $(PYTHON) scripts/validate_paper_claims.py

# Paper 2: certificate half-life blackout benchmark
paper2-blackout-benchmark:
	$(PYTHON) scripts/run_paper2_blackout_benchmark.py

# Paper 3: four-policy graceful degradation comparison
paper3-four-policy-benchmark:
	$(PYTHON) scripts/run_paper3_four_policy_benchmark.py

paper-compile:
	cd paper && pdflatex -interaction=nonstopmode paper.tex
	cd paper && pdflatex -interaction=nonstopmode paper.tex

paper-refresh: paper-assets paper-verify paper-compile

paper-freeze:
ifndef RELEASE_ID
	$(error RELEASE_ID is not set. Usage: make paper-freeze RELEASE_ID=FINAL_20260312T000000Z)
endif
	$(PYTHON) scripts/post_training_paper_update.py --release-id $(RELEASE_ID) --out-dir reports/publication

dc3s-demo:
	PYTHONPATH=src $(PYTHON) scripts/run_dc3s_demo.py

# ORIUS full pipeline check (imports, config, DC3S demo, CPSBench, locked evidence)
orius-check:
	PYTHONPATH=src $(PYTHON) scripts/run_orius_full_check.py

# Thesis-level pipeline verification: theorems + full ORIUS + AV training
thesis-pipeline-verify: orius-check
	$(PYTHON) scripts/verify_theorem_anchors.py
	@echo "Thesis pipeline verification complete: theorems + ORIUS + training"

# Battery-anchored ORIUS framework proof bundle (all domains + proof gate)
framework-proof:
	PYTHONPATH=src $(PYTHON) scripts/build_orius_framework_proof.py --seeds $(PROOF_SEEDS) --horizon $(PROOF_HORIZON) --out reports/orius_framework_proof
# Thesis writing / production workflow
thesis-train:
ifndef RELEASE_ID
	$(error RELEASE_ID is not set. Usage: make thesis-train RELEASE_ID=FINAL_20260319T000000Z PROFILE=standard)
endif
	$(MAKE) r1-full RELEASE_ID=$(RELEASE_ID) PROFILE=$(PROFILE)

thesis-bench:
ifndef RELEASE_ID
	$(error RELEASE_ID is not set. Usage: make thesis-bench RELEASE_ID=FINAL_20260319T000000Z)
endif
	$(MAKE) r1-cpsbench RELEASE_ID=$(RELEASE_ID)

thesis-artifacts:
ifndef RELEASE_ID
	$(error RELEASE_ID is not set. Usage: make thesis-artifacts RELEASE_ID=FINAL_20260319T000000Z)
endif
	$(MAKE) publication-artifact RELEASE_ID=$(RELEASE_ID)
	$(MAKE) paper-assets

thesis-manuscript:
	$(MAKE) paper-verify
	$(MAKE) paper-compile

thesis-freeze:
ifndef RELEASE_ID
	$(error RELEASE_ID is not set. Usage: make thesis-freeze RELEASE_ID=FINAL_20260319T000000Z)
endif
	$(MAKE) paper-freeze RELEASE_ID=$(RELEASE_ID)

thesis-full:
ifndef RELEASE_ID
	$(error RELEASE_ID is not set. Usage: make thesis-full RELEASE_ID=FINAL_20260319T000000Z PROFILE=standard)
endif
	$(MAKE) thesis-train RELEASE_ID=$(RELEASE_ID) PROFILE=$(PROFILE)
	$(MAKE) thesis-bench RELEASE_ID=$(RELEASE_ID)
	$(MAKE) thesis-artifacts RELEASE_ID=$(RELEASE_ID)
	$(MAKE) thesis-manuscript

# ORIUS quick check (skip CPSBench, ~10s)
orius-check-quick:
	PYTHONPATH=src $(PYTHON) scripts/run_orius_full_check.py --quick

iot-sim:
	PYTHONPATH=src $(PYTHON) iot/simulator/run_closed_loop.py

refresh-data:
	$(PYTHON) scripts/refresh_data_delta.py --dataset ALL --apply

na-audit:
	$(PYTHON) scripts/audit_na_tables.py --config configs/publish_audit.yaml

leakage-audit:
	$(PYTHON) scripts/audit_leakage.py --config configs/publish_audit.yaml

code-health-audit:
	$(PYTHON) scripts/audit_code_health.py --config configs/publish_audit.yaml

backfill-dc3s:
	$(PYTHON) scripts/backfill_dc3s_typed_columns.py

git-delta-audit:
	$(PYTHON) scripts/audit_git_delta.py --baseline-ref origin/main --out-dir reports/publish --scope-config configs/publish_audit.yaml

figure-inventory-audit:
	$(PYTHON) scripts/audit_figure_inventory.py --out-dir reports/publish --manifest paper/metrics_manifest.json

publish-audit:
	$(PYTHON) scripts/final_publish_audit.py --config configs/publish_audit.yaml

publish-audit-isolated:
	bash scripts/run_publish_audit_isolated.sh --config configs/publish_audit.yaml --run-hooks --baseline-ref origin/main --max-runtime-hours 6 --iot-steps 72

# Compare agent artifact zip vs repo thesis (theorem set, structure, evidence)
analyze-artifact:
	$(PYTHON) scripts/analyze_agent_artifact.py

# Download AV datasets for ORIUS vehicles extension (synthetic by default)
av-datasets:
	$(PYTHON) scripts/download_av_datasets.py --source synthetic

# Download Industrial datasets (CCPP or synthetic)
industrial-datasets:
	$(PYTHON) scripts/download_industrial_datasets.py --source synthetic

# Download Healthcare datasets (BIDMC or synthetic)
healthcare-datasets:
	$(PYTHON) scripts/download_healthcare_datasets.py --source synthetic

# Download Aerospace synthetic (no real dataset yet)
aerospace-datasets:
	$(PYTHON) scripts/download_aerospace_datasets.py

# Download all multi-domain datasets (AV + Industrial + Healthcare + Aerospace)
multi-domain-datasets: av-datasets industrial-datasets healthcare-datasets aerospace-datasets

# Build features for all multi-domain datasets (run after multi-domain-datasets)
multi-domain-build:
	PYTHONPATH=src $(PYTHON) scripts/build_features_multi_domain.py

# Train a multi-domain dataset: make train-dataset DATASET=AV
# Supported: AV, INDUSTRIAL, HEALTHCARE, AEROSPACE

# Generate ORIUS universal framework figure
universal-framework-figure:
	$(PYTHON) scripts/build_universal_framework_figure.py

# Full CI check (coverage + lint + integration)
ci: lint lint-release test-cov test-integration
	@echo "✅ CI checks passed"

# Pre-release validation
pre-release: ci release_check
	@echo "✅ Pre-release validation complete"

# ============================================================
# R1 Paper Release Family
# ============================================================

# R1 diagnostic: single-seed quick check across all BAs
r1-diagnostic:
	PYTHONPATH=src $(PYTHON) scripts/run_r1_release.py --stage diagnostic $(if $(RELEASE_ID),--release-id $(RELEASE_ID),)

# R1 full: multi-seed training across DE + US BAs
r1-full:
	PYTHONPATH=src $(PYTHON) scripts/run_r1_release.py --stage full --profile $(PROFILE) $(if $(RELEASE_ID),--release-id $(RELEASE_ID),)

# R1 CPSBench: severity sweeps with MPC baseline
r1-cpsbench:
	PYTHONPATH=src $(PYTHON) scripts/run_r1_release.py --stage cpsbench $(if $(RELEASE_ID),--release-id $(RELEASE_ID),)

# R1 verify: lint + test + paper asset sync
r1-verify:
	PYTHONPATH=src $(PYTHON) scripts/run_r1_release.py --stage verify $(if $(RELEASE_ID),--release-id $(RELEASE_ID),)

# R1 promote: promote candidate artifacts
r1-promote:
	PYTHONPATH=src $(PYTHON) scripts/run_r1_release.py --stage promote $(if $(RELEASE_ID),--release-id $(RELEASE_ID),)

# R1 full pipeline end-to-end
r1-all: r1-diagnostic r1-full r1-cpsbench r1-verify r1-promote
	@echo "✅ R1 release family complete"

# Paper asset sync check
paper-sync:
	PYTHONPATH=src $(PYTHON) scripts/sync_paper_assets.py --check

# ORIUS Paper 1 lock: snapshot battery evidence (run once when freezing)
paper1-lock-snapshot:
	$(PYTHON) scripts/snapshot_paper1_lock.py

# ORIUS Paper 1 lock: verify no drift from locked values
paper1-lock-check:
	$(PYTHON) scripts/check_paper1_lock_drift.py

# Transfer study (cross-region evaluation)
transfer-study:
	PYTHONPATH=src $(PYTHON) scripts/run_transfer_study.py --all-pairs --models baseline_gbm dl_lstm

# Train all BAs with full model suite
train-all-ba:
	PYTHONPATH=src $(PYTHON) scripts/train_dataset.py --dataset DE --profile $(PROFILE)
	PYTHONPATH=src $(PYTHON) scripts/train_dataset.py --dataset US_MISO --profile $(PROFILE)
	PYTHONPATH=src $(PYTHON) scripts/train_dataset.py --dataset US_PJM --profile $(PROFILE)
	PYTHONPATH=src $(PYTHON) scripts/train_dataset.py --dataset US_ERCOT --profile $(PROFILE)
