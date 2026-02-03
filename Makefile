.PHONY: setup lint test api dashboard pipeline data train production reports monitor release_check release_check_full

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

lint:
	python -m compileall src

test:
	pytest -q

api:
	uvicorn services.api.main:app --reload --port 8000

dashboard:
	streamlit run services/dashboard/app.py

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
	python scripts/release_check.py

release_check_full:
	python scripts/release_check.py --full

production: pipeline train
