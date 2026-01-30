.PHONY: setup lint test api dashboard pipeline train production

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
	python -m gridpulse.data_pipeline.validate_schema --in data/raw --report reports/data_quality_report.md
	python -m gridpulse.data_pipeline.build_features --in data/raw --out data/processed
	python -m gridpulse.data_pipeline.split_time_series --in data/processed/features.parquet --out data/processed/splits

train:
	python -m gridpulse.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target load_mw
	python -m gridpulse.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target wind_mw
	python -m gridpulse.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target solar_mw
	python -m gridpulse.forecasting.train --config configs/train_forecast.yaml

production: pipeline train
