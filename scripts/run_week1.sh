#!/usr/bin/env bash
set -euo pipefail

python -m gridpulse.data_pipeline.download_opsd --out data/raw

# auto-detect OPSD CSV if already downloaded in nested folder
if [[ ! -f data/raw/time_series_60min_singleindex.csv ]]; then
  FOUND="$(ls data/raw/opsd-time_series-*/time_series_60min_singleindex.csv 2>/dev/null | head -n 1 || true)"
  if [[ -n "${FOUND}" ]]; then
    cp "${FOUND}" data/raw/time_series_60min_singleindex.csv
  fi
fi

WEATHER_ARG=""
if [[ "${DOWNLOAD_WEATHER:-0}" == "1" ]]; then
  python -m gridpulse.data_pipeline.download_weather --out data/raw --start 2017-01-01 --end 2020-12-31
  WEATHER_ARG="--weather data/raw/weather_berlin_hourly.csv"
fi

python -m gridpulse.data_pipeline.validate_schema --in data/raw --report reports/data_quality_report.md
python -m gridpulse.data_pipeline.build_features --in data/raw --out data/processed ${WEATHER_ARG}
python -m gridpulse.data_pipeline.split_time_series --in data/processed/features.parquet --out data/processed/splits

python -m gridpulse.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target load_mw
python -m gridpulse.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target wind_mw
python -m gridpulse.forecasting.train_baseline --features data/processed/features.parquet --splits data/processed/splits --target solar_mw

echo "Week-1 complete: data + features + baselines saved to artifacts/backtests/"
