
#!/usr/bin/env bash
set -euo pipefail

uvicorn services.api.main:app --reload --port 8000 &
API_PID=$!

npm --prefix frontend run dev &
FRONTEND_PID=$!

cleanup() {
  kill $API_PID $FRONTEND_PID 2>/dev/null || true
}
trap cleanup EXIT

wait
