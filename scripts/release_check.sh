#!/usr/bin/env bash
set -euo pipefail

FULL_FLAG=""
if [ "${1:-}" = "--full" ]; then
  FULL_FLAG="--full"
fi

echo "=== (1) Core release gate ==="
python scripts/release_check.py $FULL_FLAG

echo "=== (2) Streaming smoke (optional if data exists) ==="
if [ -f data/raw/time_series_60min_singleindex.csv ]; then
  docker compose -f docker/docker-compose.streaming.yml up -d
  python -m gridpulse.streaming.run_consumer --config configs/streaming.yaml --max-messages 500 &
  CON_PID=$!
  python scripts/replay_opsd_to_kafka.py --csv data/raw/time_series_60min_singleindex.csv --rate 200 || true
  sleep 2
  kill $CON_PID || true
  python scripts/build_streaming_report.py || true
else
  echo "Skipping streaming smoke: missing data/raw/time_series_60min_singleindex.csv"
fi

echo "=== (3) Forecast interval report (requires calibration/test npz) ==="
if [ -f artifacts/backtests/calibration.npz ] && [ -f artifacts/backtests/test.npz ]; then
  python scripts/build_forecast_interval_report.py
else
  echo "Skipping intervals: missing artifacts/backtests/calibration.npz and/or test.npz"
fi

echo "=== (4) Decision robustness report ==="
python scripts/build_decision_reports.py || true

echo "Release check complete. See reports/*.md"
