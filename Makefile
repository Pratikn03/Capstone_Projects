.PHONY: setup lint lint-release test test-cov test-quick api dashboard frontend frontend-build pipeline data train production reports monitor release_check release_check_full extract-data shap-importance stat-tests train-us reports-us verify-training cv-eval ablations stats-tables verify-novelty robustness-analysis train-dataset train-all k6-load locust-load observability down-observability cpsbench dc3s-demo orius-check orius-check-quick framework-proof thesis-pipeline-verify thesis-train thesis-bench thesis-artifacts thesis-manuscript thesis-freeze thesis-full iot-sim refresh-data table-result-integrity-repair table-result-integrity-audit na-audit leakage-audit code-health-audit git-delta-audit figure-inventory-audit backfill-dc3s publish-audit publish-audit-isolated publication-artifact clean-artifact-release analyze-artifact av-datasets nuplan-av-surface healthcare-datasets multi-domain-datasets multi-domain-build universal-framework-figure paper-assets paper-verify paper-compile paper-refresh paper-freeze paper2-blackout-benchmark orius-monograph-assets review-compile orius-book orius-evidence-rerun camera-ready-assets camera-ready-verify camera-ready-freeze orius-review-pack orius-final ieee-assets ieee-main-compile ieee-detailed-compile ieee-appendix-compile ieee-pack ieee-prof-assets ieee-prof-main-compile ieee-prof-appa-compile ieee-prof-appb-compile ieee-prof-pack orius-flagship-manuscripts battery-deep-novelty phase3-proof-book app-c-flagship-proofs app-c-all-theorems external-ssd-setup external-ssd-shell external-ssd-verify external-ssd-preflight full-folder-audit pre-clean clean clean-status fresh-research pre-release appledouble-clean appledouble-check workspace-hygiene-check workspace-hygiene-clean

PRE_RELEASE_TESTS = tests/test_submission_artifacts.py tests/test_three_domain_submission_lane.py tests/test_thesis_package_assets.py

PYTHON ?= $(if $(wildcard .venv/bin/python3),.venv/bin/python3,python3)
PROFILE ?= standard
PROOF_SEEDS ?= 1
PROOF_HORIZON ?= 24
# Reproducibility: fix Python hash seed for deterministic dict/set ordering.
export PYTHONHASHSEED := 0
# Canonical dissertation build is now the senior-review single-flow monograph.
# Historical archive sources remain tracked, but they are indexed rather than
# compiled inline to avoid repeating the same argument in multiple draft voices.
PAPER_MIN_PAGES ?= 90
# The IEEE flagship track is now a reviewable journal-style main paper plus a separate
# appendix package. Keep a real main-paper floor without forcing monograph-scale
# duplication back into the double-column draft.
IEEE_MIN_PAGES ?= 8
IEEE_DETAILED_MIN_PAGES ?= 40
IEEE_PROF_MAIN_MIN_PAGES ?= 20
IEEE_PROF_APP_MIN_PAGES ?= 20
ORIUS_AV_SOURCE ?= nuplan_singapore
NUPLAN_TRAIN_ZIP ?=
NUPLAN_TRAIN_DIR ?= .
NUPLAN_TRAIN_GLOB ?= nuplan-v*.zip
NUPLAN_TRAIN_ARGS = $(if $(NUPLAN_TRAIN_ZIP),--train-zip "$(NUPLAN_TRAIN_ZIP)",--train-dir "$(NUPLAN_TRAIN_DIR)" --train-glob "$(NUPLAN_TRAIN_GLOB)")
NUPLAN_MAPS_ZIP ?= nuplan-maps-v1.0.zip
NUPLAN_AV_OUT_DIR ?= data/orius_av/av/processed_nuplan_singapore
NUPLAN_MAX_DBS ?=
NUPLAN_MAX_SCENARIOS ?=
SSD_VOLUME ?= /Volumes/ORIUS_SSD
SSD_EXTERNAL_ROOT_NAME ?= orius_external_data
STRICT_EXTERNAL_LINK ?= $(if $(ORIUS_STRICT_EXTERNAL_ROOT),$(ORIUS_STRICT_EXTERNAL_ROOT),$(if $(ORIUS_EXTERNAL_DATA_ROOT),$(ORIUS_EXTERNAL_DATA_ROOT),$(HOME)/orius_external_data))
SHELL_PROFILE ?= $(HOME)/.zshrc
EXTERNAL_SSD_ROOT ?= $(if $(ORIUS_EXTERNAL_DATA_ROOT),$(ORIUS_EXTERNAL_DATA_ROOT),$(SSD_VOLUME)/$(SSD_EXTERNAL_ROOT_NAME))

## Canonical paper/review artifact directories are intentionally retained in repository state.

appledouble-clean:
	PYTHONPATH=scripts $(PYTHON) scripts/cleanup_appledouble.py --root . --delete

appledouble-check:
	PYTHONPATH=scripts $(PYTHON) scripts/validate_no_appledouble.py --root . --exclude-active

workspace-hygiene-check:
	PYTHONPATH=scripts $(PYTHON) scripts/validate_workspace_hygiene.py --root . --exclude-active

workspace-hygiene-clean:
	PYTHONPATH=scripts $(PYTHON) scripts/cleanup_appledouble.py --root . --delete
	PYTHONPATH=scripts $(PYTHON) scripts/validate_workspace_hygiene.py --root . --exclude-active --delete-stale-pids

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/*.bbl paper/*.blg paper/*.toc paper/*.lof paper/*.lot
	find paper/monograph -maxdepth 1 -type f ! -name '._*' \( -name "*.aux" -o -name "*.out" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" \) -delete
	find appendices -maxdepth 1 -type f ! -name '._*' \( -name "*.aux" -o -name "*.out" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" \) -delete
	rm -f paper/ieee/*.aux paper/ieee/*.log paper/ieee/*.out paper/ieee/*.fdb_latexmk paper/ieee/*.fls paper/ieee/*.bbl paper/ieee/*.blg paper/ieee/*.toc paper/ieee/*.lof paper/ieee/*.lot paper/ieee/*.pdf
	rm -f paper_r1.aux paper_r1.out paper_r1.log paper_r1.fls paper_r1.fdb_latexmk paper.aux paper.out paper.log paper.fdb_latexmk paper.fls paper.fls.gz
	rm -f .coverage reports/.coverage
	rm -rf .coverage_html reports/coverage .pytest_cache reports/coverage/.pytest_cache reports/.pytest_cache
	rm -rf .ruff_cache .mypy_cache .cache
	rm -rf frontend/.next node_modules .next
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -not -path "./.venv/*" -not -path "./.git/*" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.venv/*" -not -path "./.git/*" -delete
	$(MAKE) appledouble-clean

pre-clean: clean

fresh-research: pre-clean orius-evidence-rerun orius-book orius-review-pack paper-verify

orius-full: fresh-research

clean-status:
	@python3 -c $$'import subprocess, sys\nstatus = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True).stdout.strip().splitlines()\nallowed_prefixes = ("paper/", "reports/publication/")\nunexpected = []\nfor row in status:\n    if len(row) < 3:\n        continue\n    raw_path = row[3:]\n    path = raw_path.split(" -> ")[-1] if " -> " in raw_path else raw_path\n    if not any(path.startswith(prefix) for prefix in allowed_prefixes):\n        unexpected.append(row)\nif not status:\n    print(\"clean-status: repository is clean\")\n    raise SystemExit(0)\nif unexpected:\n    print(\"clean-status: unexpected repo changes detected outside canonical artifacts:\")\n    print(\"\\n\".join(unexpected))\n    raise SystemExit(1)\nprint(\"clean-status: only canonical artifact outputs changed\")'

setup:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && .venv/bin/pip install -r requirements.lock.txt
	cd frontend && npm install

lint:
	PYTHONPYCACHEPREFIX=/tmp/orius_pycache $(PYTHON) -m compileall -x '(^|/)\._' src services scripts

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
	$(PYTHON) -m pytest -x --tb=short --no-cov

# Test with Phase 1 coverage bar (minimum 80% on critical validation/data-glue surfaces)
test-cov:
	$(PYTHON) -m pytest -q --tb=short \
		tests/test_coverage_theorem.py \
		tests/test_universal_validation_gate.py \
		tests/test_domain_closure_matrix.py \
		tests/test_external_real_data_integration.py \
		tests/test_phase1_data_glue.py \
		tests/test_real_data_manifest_refresh.py \
		--cov=scripts \
		--cov=orius \
		--cov-report= \
		--cov-fail-under=0
	$(PYTHON) -m coverage report \
		--include="scripts/_dataset_registry.py,scripts/refresh_real_data_manifests.py,src/orius/orius_bench/real_data_loader.py" \
		--fail-under=80
	$(PYTHON) -m coverage html \
		--include="scripts/_dataset_registry.py,scripts/refresh_real_data_manifests.py,src/orius/orius_bench/real_data_loader.py" \
		-d reports/coverage
	$(PYTHON) -m coverage xml \
		--include="scripts/_dataset_registry.py,scripts/refresh_real_data_manifests.py,src/orius/orius_bench/real_data_loader.py" \
		-o reports/coverage.xml

# Quick tests (exclude slow markers)
test-quick:
	$(PYTHON) -m pytest -q -m "not slow and not integration" --no-cov

# Integration tests only
test-integration:
	$(PYTHON) -m pytest -q -m "integration" --no-cov

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

# Battery-first deep-learning novelty package
battery-deep-novelty:
	PYTHONPATH=src $(PYTHON) scripts/run_battery_deep_novelty.py

# External SSD raw-data setup for MacBook workflows
external-ssd-setup:
	PYTHONPATH=src $(PYTHON) scripts/setup_external_ssd.py --volume-root $(SSD_VOLUME) --strict-link $(STRICT_EXTERNAL_LINK)

external-ssd-shell:
	PYTHONPATH=src $(PYTHON) scripts/setup_external_ssd.py --volume-root $(SSD_VOLUME) --strict-link $(STRICT_EXTERNAL_LINK) --append-shell-profile --shell-profile $(SHELL_PROFILE)

external-ssd-verify:
	PYTHONPATH=src $(PYTHON) scripts/setup_external_ssd.py --volume-root $(SSD_VOLUME) --strict-link $(STRICT_EXTERNAL_LINK) --verify-only

external-ssd-preflight:
	PYTHONPATH=src $(PYTHON) scripts/verify_real_data_preflight.py --external-root $(EXTERNAL_SSD_ROOT)

phase3-proof-book:
	$(PYTHON) scripts/build_phase3_flagship_v1_proof_book.py

app-c-flagship-proofs:
	$(PYTHON) scripts/build_app_c_flagship_proofs.py

app-c-all-theorems:
	$(PYTHON) scripts/build_app_c_all_theorems.py

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

clean-artifact-release:
	$(PYTHON) scripts/build_clean_artifact_release.py --out-root artifacts/releases --mode full-derived --include-manuscripts --verify

paper-assets:
	bash scripts/export_paper_assets.sh
	PYTHONPATH=src $(PYTHON) scripts/build_paper_table_tex.py
	PYTHONPATH=src $(PYTHON) scripts/update_paper_metrics.py

orius-monograph-assets:
	PYTHONPATH=src $(PYTHON) scripts/build_active_theorem_audit.py
	PYTHONPATH=src $(PYTHON) scripts/validate_theorem_surface.py
	PYTHONPATH=src $(PYTHON) scripts/build_orius_monograph_assets.py
	PYTHONPATH=src $(PYTHON) scripts/build_missing_tables.py

ieee-assets: orius-monograph-assets
	PYTHONPATH=src $(PYTHON) scripts/build_orius_ieee_assets.py

camera-ready-assets: ieee-assets
	PYTHONPATH=src $(PYTHON) scripts/build_camera_ready_tables.py
	PYTHONPATH=src $(PYTHON) scripts/build_camera_ready_figures.py
	PYTHONPATH=src $(PYTHON) scripts/build_camera_ready_figure_lineage.py --write

paper-verify: orius-monograph-assets
	PYTHONPATH=src $(PYTHON) scripts/verify_paper_manifest.py --paper orius_book.tex
	PYTHONPATH=src $(PYTHON) scripts/validate_paper_claims.py --tex orius_book.tex
	PYTHONPATH=src $(PYTHON) scripts/repair_table_result_integrity.py
	PYTHONPATH=src $(PYTHON) scripts/audit_table_result_integrity.py

camera-ready-verify: camera-ready-assets
	PYTHONPATH=src $(PYTHON) scripts/build_camera_ready_figure_lineage.py --verify
	PYTHONPATH=src $(PYTHON) scripts/verify_paper_manifest.py --camera-ready
	PYTHONPATH=src $(PYTHON) scripts/validate_paper_claims.py
	PYTHONPATH=src $(PYTHON) scripts/verify_camera_ready_logs.py --waivers paper/camera_ready_warning_waivers.yaml --log paper/ieee/orius_ieee_main.log --log paper/ieee/orius_ieee_detailed_main.log
	PYTHONPATH=src $(PYTHON) scripts/repair_table_result_integrity.py
	PYTHONPATH=src $(PYTHON) scripts/audit_table_result_integrity.py

# Paper 2: certificate half-life blackout benchmark
paper2-blackout-benchmark:
	$(PYTHON) scripts/run_paper2_blackout_benchmark.py

# Paper 3: four-policy graceful degradation comparison
paper3-four-policy-benchmark:
	$(PYTHON) scripts/run_paper3_four_policy_benchmark.py

paper-compile: orius-monograph-assets app-c-flagship-proofs app-c-all-theorems
	PYTHONPATH=src $(PYTHON) scripts/repair_table_result_integrity.py
	PYTHONPATH=src $(PYTHON) scripts/audit_table_result_integrity.py
	rm -f paper/paper.pdf paper/paper.log paper/paper.aux paper/paper.out paper/paper.bbl paper/paper.blg paper/paper.toc paper/paper.lof paper/paper.lot
	rm -f orius_book.pdf orius_book.log orius_book.aux orius_book.out orius_book.bbl orius_book.blg orius_book.toc orius_book.lof orius_book.lot
	find paper/monograph -maxdepth 1 -type f ! -name '._*' \( -name "*.aux" -o -name "*.out" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" \) -delete
	find appendices -maxdepth 1 -type f ! -name '._*' \( -name "*.aux" -o -name "*.out" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" \) -delete
	find chapters chapters_merged -maxdepth 1 -type f ! -name '._*' \( -name "*.aux" -o -name "*.out" -o -name "*.toc" -o -name "*.lof" -o -name "*.lot" \) -delete
	mkdir -p paper/monograph paper/chapters paper/appendices paper/backmatter
	pdflatex -interaction=nonstopmode orius_book.tex
	bibtex orius_book
	pdflatex -interaction=nonstopmode orius_book.tex
	pdflatex -interaction=nonstopmode orius_book.tex
	test -f orius_book.pdf
	test -f orius_book.log
	cp orius_book.pdf paper/paper.pdf
	cp orius_book.log paper/paper.log
	cp orius_book.pdf paper.pdf
	test -f paper.pdf
	cmp -s paper/paper.pdf paper.pdf
	@pages=$$(grep -Eo 'Output written on paper/paper\.pdf \([0-9]+ pages' paper/paper.log | tail -n1 | sed -E 's/.*\(([0-9]+) pages/\1/'); \
	if [ -z "$$pages" ]; then \
		pages=$$(grep -Eo 'Output written on orius_book\.pdf \([0-9]+ pages' paper/paper.log | tail -n1 | sed -E 's/.*\(([0-9]+) pages/\1/'); \
	fi; \
	if [ -z "$$pages" ]; then \
		echo "Could not determine page count from paper/paper.log"; \
		exit 1; \
	fi; \
	if [ "$$pages" -lt "$(PAPER_MIN_PAGES)" ]; then \
		echo "Canonical paper.pdf page count $$pages is below required minimum $(PAPER_MIN_PAGES)"; \
		exit 1; \
	fi; \
	echo "Canonical paper.pdf page count $$pages >= $(PAPER_MIN_PAGES)"
	rm -f orius_book.pdf orius_book.log orius_book.aux orius_book.out orius_book.bbl orius_book.blg orius_book.toc orius_book.lof orius_book.lot
	$(MAKE) appledouble-clean

ieee-main-compile: ieee-assets
	rm -f paper/ieee/orius_ieee_main.pdf paper/ieee/orius_ieee_main.log paper/ieee/orius_ieee_main.aux paper/ieee/orius_ieee_main.out paper/ieee/orius_ieee_main.bbl paper/ieee/orius_ieee_main.blg paper/ieee/orius_ieee_main.toc
	mkdir -p paper/ieee/generated paper/ieee/sections
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_main.tex
	- sh -c 'cd paper/ieee && bibtex orius_ieee_main'
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_main.tex
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_main.tex
	test -f paper/ieee/orius_ieee_main.pdf
	@pages=$$(grep -Eo 'Output written on (paper/ieee/)?orius_ieee_main\.pdf \([0-9]+ pages' paper/ieee/orius_ieee_main.log | tail -n1 | sed -E 's/.*\(([0-9]+) pages/\1/'); \
	if [ -z "$$pages" ]; then \
		echo "Could not determine page count from paper/ieee/orius_ieee_main.log"; \
		exit 1; \
	fi; \
	if [ "$$pages" -lt "$(IEEE_MIN_PAGES)" ]; then \
		echo "IEEE main page count $$pages is below required minimum $(IEEE_MIN_PAGES)"; \
		exit 1; \
	fi; \
	echo "IEEE main page count $$pages >= $(IEEE_MIN_PAGES)"

ieee-appendix-compile: ieee-assets
	rm -f paper/ieee/orius_ieee_appendix.pdf paper/ieee/orius_ieee_appendix.log paper/ieee/orius_ieee_appendix.aux paper/ieee/orius_ieee_appendix.out paper/ieee/orius_ieee_appendix.bbl paper/ieee/orius_ieee_appendix.blg paper/ieee/orius_ieee_appendix.toc
	mkdir -p paper/ieee/generated paper/ieee/sections
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_appendix.tex
	- sh -c 'cd paper/ieee && bibtex orius_ieee_appendix'
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_appendix.tex
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_appendix.tex
	test -f paper/ieee/orius_ieee_appendix.pdf

ieee-detailed-compile: ieee-assets
	rm -f paper/ieee/orius_ieee_detailed_main.pdf paper/ieee/orius_ieee_detailed_main.log paper/ieee/orius_ieee_detailed_main.aux paper/ieee/orius_ieee_detailed_main.out paper/ieee/orius_ieee_detailed_main.bbl paper/ieee/orius_ieee_detailed_main.blg paper/ieee/orius_ieee_detailed_main.toc
	mkdir -p paper/ieee/generated paper/ieee/detailed_sections
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_detailed_main.tex
	- sh -c 'cd paper/ieee && bibtex orius_ieee_detailed_main'
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_detailed_main.tex
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_detailed_main.tex
	test -f paper/ieee/orius_ieee_detailed_main.pdf
	@pages=$$(grep -Eo 'Output written on paper/ieee/orius_ieee_detailed_main\.pdf \([0-9]+ pages' paper/ieee/orius_ieee_detailed_main.log | tail -n1 | sed -E 's/.*\(([0-9]+) pages/\1/'); \
	if [ -z "$$pages" ]; then \
		echo "Could not determine page count from paper/ieee/orius_ieee_detailed_main.log"; \
		exit 1; \
	fi; \
	if [ "$$pages" -lt "$(IEEE_DETAILED_MIN_PAGES)" ]; then \
		echo "IEEE detailed page count $$pages is below required minimum $(IEEE_DETAILED_MIN_PAGES)"; \
		exit 1; \
	fi; \
	echo "IEEE detailed page count $$pages >= $(IEEE_DETAILED_MIN_PAGES)"

ieee-pack: ieee-main-compile ieee-appendix-compile

ieee-prof-assets: ieee-assets

ieee-prof-main-compile: ieee-prof-assets
	rm -f paper/ieee/orius_ieee_professor_main.pdf paper/ieee/orius_ieee_professor_main.log paper/ieee/orius_ieee_professor_main.aux paper/ieee/orius_ieee_professor_main.out paper/ieee/orius_ieee_professor_main.bbl paper/ieee/orius_ieee_professor_main.blg paper/ieee/orius_ieee_professor_main.toc
	mkdir -p paper/ieee/generated paper/ieee/professor_sections
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_professor_main.tex
	- sh -c 'cd paper/ieee && bibtex orius_ieee_professor_main'
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_professor_main.tex
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_professor_main.tex
	test -f paper/ieee/orius_ieee_professor_main.pdf
	@pages=$$(grep -Eo 'Output written on paper/ieee/orius_ieee_professor_main\.pdf \([0-9]+ pages' paper/ieee/orius_ieee_professor_main.log | tail -n1 | sed -E 's/.*\(([0-9]+) pages/\1/'); \
	if [ -z "$$pages" ]; then \
		echo "Could not determine page count from paper/ieee/orius_ieee_professor_main.log"; \
		exit 1; \
	fi; \
	if [ "$$pages" -lt "$(IEEE_PROF_MAIN_MIN_PAGES)" ]; then \
		echo "IEEE professor main page count $$pages is below required minimum $(IEEE_PROF_MAIN_MIN_PAGES)"; \
		exit 1; \
	fi; \
	echo "IEEE professor main page count $$pages >= $(IEEE_PROF_MAIN_MIN_PAGES)"

ieee-prof-appa-compile: ieee-prof-assets
	rm -f paper/ieee/orius_ieee_professor_appendix_a.pdf paper/ieee/orius_ieee_professor_appendix_a.log paper/ieee/orius_ieee_professor_appendix_a.aux paper/ieee/orius_ieee_professor_appendix_a.out paper/ieee/orius_ieee_professor_appendix_a.bbl paper/ieee/orius_ieee_professor_appendix_a.blg paper/ieee/orius_ieee_professor_appendix_a.toc
	mkdir -p paper/ieee/generated paper/ieee/professor_sections
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_professor_appendix_a.tex
	- sh -c 'cd paper/ieee && bibtex orius_ieee_professor_appendix_a'
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_professor_appendix_a.tex
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_professor_appendix_a.tex
	test -f paper/ieee/orius_ieee_professor_appendix_a.pdf
	@pages=$$(grep -Eo 'Output written on paper/ieee/orius_ieee_professor_appendix_a\.pdf \([0-9]+ pages' paper/ieee/orius_ieee_professor_appendix_a.log | tail -n1 | sed -E 's/.*\(([0-9]+) pages/\1/'); \
	if [ -z "$$pages" ]; then \
		echo "Could not determine page count from paper/ieee/orius_ieee_professor_appendix_a.log"; \
		exit 1; \
	fi; \
	if [ "$$pages" -lt "$(IEEE_PROF_APP_MIN_PAGES)" ]; then \
		echo "IEEE professor Appendix A page count $$pages is below required minimum $(IEEE_PROF_APP_MIN_PAGES)"; \
		exit 1; \
	fi; \
	echo "IEEE professor Appendix A page count $$pages >= $(IEEE_PROF_APP_MIN_PAGES)"

ieee-prof-appb-compile: ieee-prof-assets
	rm -f paper/ieee/orius_ieee_professor_appendix_b.pdf paper/ieee/orius_ieee_professor_appendix_b.log paper/ieee/orius_ieee_professor_appendix_b.aux paper/ieee/orius_ieee_professor_appendix_b.out paper/ieee/orius_ieee_professor_appendix_b.bbl paper/ieee/orius_ieee_professor_appendix_b.blg paper/ieee/orius_ieee_professor_appendix_b.toc
	mkdir -p paper/ieee/generated paper/ieee/professor_sections
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_professor_appendix_b.tex
	- sh -c 'cd paper/ieee && bibtex orius_ieee_professor_appendix_b'
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_professor_appendix_b.tex
	- pdflatex -interaction=nonstopmode -output-directory=paper/ieee paper/ieee/orius_ieee_professor_appendix_b.tex
	test -f paper/ieee/orius_ieee_professor_appendix_b.pdf
	@pages=$$(grep -Eo 'Output written on paper/ieee/orius_ieee_professor_appendix_b\.pdf \([0-9]+ pages' paper/ieee/orius_ieee_professor_appendix_b.log | tail -n1 | sed -E 's/.*\(([0-9]+) pages/\1/'); \
	if [ -z "$$pages" ]; then \
		echo "Could not determine page count from paper/ieee/orius_ieee_professor_appendix_b.log"; \
		exit 1; \
	fi; \
	if [ "$$pages" -lt "$(IEEE_PROF_APP_MIN_PAGES)" ]; then \
		echo "IEEE professor Appendix B page count $$pages is below required minimum $(IEEE_PROF_APP_MIN_PAGES)"; \
		exit 1; \
	fi; \
	echo "IEEE professor Appendix B page count $$pages >= $(IEEE_PROF_APP_MIN_PAGES)"

ieee-prof-pack: ieee-prof-main-compile ieee-prof-appa-compile ieee-prof-appb-compile

orius-flagship-manuscripts: orius-book ieee-pack ieee-prof-pack

review-compile: orius-monograph-assets
	rm -f paper/review/orius_review_dossier.pdf paper/review/orius_review_dossier.log paper/review/orius_review_dossier.aux paper/review/orius_review_dossier.out paper/review/orius_review_dossier.toc
	pdflatex -interaction=nonstopmode -output-directory=paper/review paper/review/orius_review_dossier.tex
	pdflatex -interaction=nonstopmode -output-directory=paper/review paper/review/orius_review_dossier.tex
	test -f paper/review/orius_review_dossier.pdf
	cp paper/review/orius_review_dossier.pdf reports/publication/orius_review_dossier.pdf
	test -f reports/publication/orius_review_dossier.pdf
	$(MAKE) appledouble-clean

paper-refresh: paper-assets paper-verify paper-compile

orius-book: paper-verify paper-compile

orius-evidence-rerun: paper-assets orius-monograph-assets
	@echo "Rebuilt monograph evidence surfaces from tracked publication artifacts."

camera-ready-freeze:
	PYTHONPATH=src $(PYTHON) scripts/run_camera_ready_freeze.py --external-root $(STRICT_EXTERNAL_LINK) --compute-lane hybrid --train-missing --repair-invalid-splits --seeds 3 --sil-seeds 3 --sil-rows 96 --horizon 48 --warning-waivers paper/camera_ready_warning_waivers.yaml
	PYTHONPATH=src $(PYTHON) scripts/repair_table_result_integrity.py
	PYTHONPATH=src $(PYTHON) scripts/audit_table_result_integrity.py

full-folder-audit:
	PYTHONPATH=src $(PYTHON) scripts/full_folder_audit.py --out-dir reports/audit

orius-review-pack: orius-monograph-assets review-compile

orius-final: orius-evidence-rerun orius-book orius-review-pack

paper-freeze:
ifndef RELEASE_ID
	$(error RELEASE_ID is not set. Usage: make paper-freeze RELEASE_ID=FINAL_20260312T000000Z)
endif
	$(PYTHON) scripts/post_training_paper_update.py --release-id $(RELEASE_ID) --out-dir reports/publication
	PYTHONPATH=src $(PYTHON) scripts/repair_table_result_integrity.py
	PYTHONPATH=src $(PYTHON) scripts/audit_table_result_integrity.py

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

table-result-integrity-repair:
	PYTHONPATH=src $(PYTHON) scripts/repair_table_result_integrity.py

table-result-integrity-audit:
	PYTHONPATH=src $(PYTHON) scripts/audit_table_result_integrity.py

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
	$(PYTHON) scripts/final_submission_audit.py

publish-audit-isolated:
	bash scripts/run_publish_audit_isolated.sh --config configs/publish_audit.yaml --run-hooks --baseline-ref origin/main --max-runtime-hours 6 --iot-steps 72

# Compare agent artifact zip vs repo thesis (theorem set, structure, evidence)
analyze-artifact:
	$(PYTHON) scripts/analyze_agent_artifact.py

# Build the canonical AV processed dataset from the configured source.
nuplan-av-surface:
	PYTHONPATH=src $(PYTHON) scripts/build_nuplan_av_surface.py $(NUPLAN_TRAIN_ARGS) --maps-zip "$(NUPLAN_MAPS_ZIP)" --out-dir "$(NUPLAN_AV_OUT_DIR)" $(if $(NUPLAN_MAX_DBS),--max-dbs $(NUPLAN_MAX_DBS),) $(if $(NUPLAN_MAX_SCENARIOS),--max-scenarios $(NUPLAN_MAX_SCENARIOS),) --build-features

av-datasets:
	@if [ "$(ORIUS_AV_SOURCE)" = "nuplan_singapore" ]; then \
		PYTHONPATH=src $(PYTHON) scripts/build_nuplan_av_surface.py $(NUPLAN_TRAIN_ARGS) --maps-zip "$(NUPLAN_MAPS_ZIP)" --out-dir "$(NUPLAN_AV_OUT_DIR)" $(if $(NUPLAN_MAX_DBS),--max-dbs $(NUPLAN_MAX_DBS),) $(if $(NUPLAN_MAX_SCENARIOS),--max-scenarios $(NUPLAN_MAX_SCENARIOS),) --build-features; \
	else \
		$(PYTHON) scripts/download_av_datasets.py --source $(ORIUS_AV_SOURCE); \
	fi

# Build Healthcare datasets from the active promoted healthcare sources.
healthcare-datasets:
	$(PYTHON) scripts/download_healthcare_datasets.py --source bidmc

# Download all active multi-domain datasets (AV + Healthcare).
multi-domain-datasets: av-datasets healthcare-datasets

# Verify disk, tooling, and repo-local raw-data readiness for the active 3-domain program.
real-data-preflight:
	$(PYTHON) scripts/verify_real_data_preflight.py

# Refresh lightweight provenance manifests without rebuilding datasets.
refresh-real-data-manifests:
	$(PYTHON) scripts/refresh_real_data_manifests.py

# Build features for the active non-battery datasets (run after multi-domain-datasets)
multi-domain-build:
	PYTHONPATH=src $(PYTHON) scripts/build_features_multi_domain.py

# Train a multi-domain dataset: make train-dataset DATASET=AV
# Supported in the active lane: AV, HEALTHCARE

# Generate ORIUS universal framework figure
universal-framework-figure:
	$(PYTHON) scripts/build_universal_framework_figure.py

# Full CI check (coverage + lint + integration)
ci: lint lint-release test-cov test-integration
	@echo "✅ CI checks passed"

pre-release:
	PYTEST_ADDOPTS=--no-cov PYTHONPATH=src .venv/bin/pytest -q $(PRE_RELEASE_TESTS)
	$(MAKE) paper-verify
	$(MAKE) orius-book
	$(MAKE) orius-review-pack
	$(MAKE) orius-evidence-rerun

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
