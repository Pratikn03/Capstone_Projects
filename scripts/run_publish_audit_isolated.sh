#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
WORKDIR="${GRIDPULSE_AUDIT_WORKDIR:-/tmp/orius_publish_audit_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COPY_FULL="${GRIDPULSE_AUDIT_COPY_FULL:-0}"
INCLUDE_HEAVY_DATA="${GRIDPULSE_AUDIT_INCLUDE_HEAVY_DATA:-0}"

mkdir -p "${WORKDIR}"

if command -v rsync >/dev/null 2>&1; then
  RSYNC_ARGS=(
    -a
    --exclude '.venv'
    --exclude '.venv_test'
    --exclude '.pytest_cache'
    --exclude '.ruff_cache'
    --exclude 'frontend/node_modules'
    --exclude 'frontend/.next'
    --exclude 'reports/coverage'
    --exclude '__pycache__'
  )
  if [ "${COPY_FULL}" != "1" ]; then
    RSYNC_ARGS+=(--exclude 'reports/publish')
    RSYNC_ARGS+=(--exclude 'artifacts/models')
    RSYNC_ARGS+=(--exclude 'artifacts/models_eia930')
    RSYNC_ARGS+=(--exclude 'artifacts/backtests')
    if [ "${INCLUDE_HEAVY_DATA}" != "1" ]; then
      RSYNC_ARGS+=(--exclude 'data/audit')
      RSYNC_ARGS+=(--exclude 'data/interim')
      RSYNC_ARGS+=(--exclude 'data/raw')
      RSYNC_ARGS+=(--exclude 'logs')
    fi
  fi
  rsync "${RSYNC_ARGS[@]}" "${ROOT_DIR}/" "${WORKDIR}/"
else
  cp -R "${ROOT_DIR}/." "${WORKDIR}/"
fi

cd "${WORKDIR}"

echo "[isolated-audit] workspace: ${WORKDIR}"
echo "[isolated-audit] python: ${PYTHON_BIN}"
echo "[isolated-audit] copy_full=${COPY_FULL} include_heavy_data=${INCLUDE_HEAVY_DATA}"

"${PYTHON_BIN}" scripts/final_publish_audit.py "$@"

echo "[isolated-audit] complete"
