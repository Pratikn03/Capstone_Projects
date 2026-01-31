#!/usr/bin/env bash
set -euo pipefail

uvicorn services.api.main:app --reload --port 8000 &
API_PID=$!

streamlit run services/dashboard/app.py &
DASH_PID=$!

cleanup() {
  kill $API_PID $DASH_PID 2>/dev/null || true
}
trap cleanup EXIT

wait
