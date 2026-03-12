# Runbook (End-to-End)

This runbook reproduces the full pipeline on either OPSD or EIA-930.

## 0) Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) OPSD pipeline (country-aware)
```bash
python -m gridpulse.data_pipeline.download_opsd --out data/raw
python -m gridpulse.data_pipeline.validate_schema --in data/raw --report reports/data_quality_report.md
python -m gridpulse.data_pipeline.build_features --in data/raw --out data/processed --country DE
python -m gridpulse.data_pipeline.split_time_series --in data/processed/features.parquet --out data/processed/splits
```

Swap `--country DE` for any supported OPSD country code such as `FR` or `ES`. The normalized downstream schema remains unchanged: `timestamp`, `load_mw`, `wind_mw`, `solar_mw`, and optional `price_eur_mwh`.

## 2) EIA-930 pipeline (US)
```bash
python -m gridpulse.data_pipeline.build_features_eia930 \
  --in data/raw/us_eia930 \
  --out data/processed/us_eia930 \
  --ba MISO

python -m gridpulse.data_pipeline.split_time_series \
  --in data/processed/us_eia930/features.parquet \
  --out data/processed/us_eia930/splits
```

## 3) Train models
```bash
python -m gridpulse.forecasting.train --config configs/train_forecast.yaml
```

## 4) Generate reports
```bash
python scripts/build_reports.py
```

EIA-930 reports:
```bash
python scripts/build_reports.py \
  --features data/processed/us_eia930/features.parquet \
  --splits data/processed/us_eia930/splits \
  --models-dir artifacts/models_eia930 \
  --reports-dir reports/eia930
```

## 5) Update README impact table
```bash
python scripts/update_readme_impact.py
```

## 6) Run API + dashboard
```bash
uvicorn services.api.main:app --reload --port 8000
cd frontend && npm run dev
```

## 7) Run monitoring with DC3S health
```bash
python scripts/run_monitoring.py
cat reports/monitoring_summary.json
```

Expected monitoring payload now includes:
- `dc3s_health.commands_total`
- `dc3s_health.intervention_rate`
- `dc3s_health.low_reliability_rate`
- `dc3s_health.drift_flag_rate`
- `dc3s_health.inflation_p95`
- `dc3s_health.triggered_flags`

If command volume is too low, `dc3s_health.insufficient_data` is `true` and DC3S retraining triggers remain disabled.

## 8) Conditional retraining (includes DC3S triggers)
```bash
python scripts/retrain_if_needed.py --refresh
```

Retraining reasons can include:
- `data_drift`
- `model_drift`
- `scheduled_cadence`
- `dc3s_intervention_spike`
- `dc3s_reliability_degradation`
- `dc3s_drift_persistence`
