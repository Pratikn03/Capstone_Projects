#!/usr/bin/env bash
# Run evidence gates from orius_evidence_checklist_audit.md.
# Use: bash scripts/verify_evidence_gates.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

echo "=== Evidence gates (python: $PY) ==="
"$PY" scripts/validate_paper_claims.py
"$PY" scripts/verify_theorem_anchors.py
"$PY" scripts/run_universal_orius_validation.py
"$PY" scripts/sync_paper_assets.py --check
"$PY" scripts/run_orius_full_check.py
echo "=== All gates passed ==="
