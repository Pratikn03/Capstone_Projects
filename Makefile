.PHONY: setup lint test api dashboard frontend frontend-build pipeline data train production reports monitor release_check release_check_full

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install

lint:
	python -m compileall src services scripts

test:
	pytest -q

api:
	PYTHONPATH=. uvicorn services.api.main:app --reload --port 8000

dashboard:
	streamlit run services/dashboard/app.py

frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

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

monitor:
	python scripts/run_monitoring.py

release_check:
	bash scripts/release_check.sh

release_check_full:
	bash scripts/release_check.sh --full

production: pipeline train
