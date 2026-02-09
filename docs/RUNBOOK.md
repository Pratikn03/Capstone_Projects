# Runbook (End-to-End)

This runbook reproduces the full pipeline on either OPSD or EIA-930.

## 0) Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) OPSD pipeline (Germany)
```bash
python -m gridpulse.data_pipeline.download_opsd --out data/raw
python -m gridpulse.data_pipeline.validate_schema --in data/raw --report reports/data_quality_report.md
python -m gridpulse.data_pipeline.build_features --in data/raw --out data/processed
python -m gridpulse.data_pipeline.split_time_series --in data/processed/features.parquet --out data/processed/splits
```

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
