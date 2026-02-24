#!/usr/bin/env bash
set -euo pipefail

FULL_FLAG=""
if [ "${1:-}" = "--full" ]; then
  FULL_FLAG="--full"
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No Python interpreter found in PATH (tried: python, python3)."
    exit 127
  fi
fi

echo "=== (1) Core release gate ==="
"$PYTHON_BIN" scripts/release_check.py $FULL_FLAG

echo "=== (2) Streaming smoke (optional if data exists) ==="
if [ -f data/raw/time_series_60min_singleindex.csv ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Skipping streaming smoke: docker command not found"
  elif ! docker info >/dev/null 2>&1; then
    echo "Skipping streaming smoke: docker daemon unavailable"
  else
    docker compose -f docker/docker-compose.streaming.yml up -d
    "$PYTHON_BIN" -m gridpulse.streaming.run_consumer --config configs/streaming.yaml --max-messages 500 &
    CON_PID=$!
    "$PYTHON_BIN" scripts/replay_opsd_to_kafka.py --csv data/raw/time_series_60min_singleindex.csv --rate 200 || true
    sleep 2
    kill $CON_PID || true
    "$PYTHON_BIN" scripts/build_streaming_report.py || true
  fi
else
  echo "Skipping streaming smoke: missing data/raw/time_series_60min_singleindex.csv"
fi

echo "=== (3) Forecast interval report (requires calibration/test npz) ==="
if [ -f artifacts/backtests/calibration.npz ] && [ -f artifacts/backtests/test.npz ]; then
  "$PYTHON_BIN" scripts/build_forecast_interval_report.py
else
  echo "Skipping intervals: missing artifacts/backtests/calibration.npz and/or test.npz"
fi

echo "=== (4) Decision robustness report ==="
"$PYTHON_BIN" scripts/build_decision_reports.py || true

echo "Release check complete. See reports/*.md"
