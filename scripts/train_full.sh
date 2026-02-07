#!/usr/bin/env bash
set -euo pipefail

# End-to-end local training + report generation (figures in reports/figures).
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

export PYTHONPATH=src
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

if [ ! -f data/processed/features.parquet ]; then
  $PYTHON_BIN -m gridpulse.data_pipeline.build_features --in data/raw --out data/processed
fi

if [ ! -d data/processed/splits ]; then
  $PYTHON_BIN -m gridpulse.data_pipeline.split_time_series --in data/processed/features.parquet --out data/processed/splits
fi

$PYTHON_BIN -m gridpulse.forecasting.train --config configs/train_forecast.yaml
$PYTHON_BIN scripts/build_reports.py
$PYTHON_BIN scripts/build_forecast_interval_report.py
$PYTHON_BIN scripts/build_decision_reports.py

echo "✅ Training + reports complete. See reports/figures/ and reports/*.md"
