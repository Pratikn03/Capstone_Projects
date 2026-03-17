#!/usr/bin/env bash
# Verify release integrity: required paths, claim validator, sync assets, negative test.
# Exit 0 if all pass; 1 otherwise.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
FAIL=0

echo "=== Verify release integrity ==="

# 1. Required paths
REQUIRED=(
  paper/metrics_manifest.json
  paper/claim_matrix.csv
  scripts/validate_paper_claims.py
  scripts/sync_paper_assets.py
  reports/impact_summary.csv
  reports/eia930/impact_summary.csv
  reports/research_metrics_de.csv
  reports/research_metrics_us.csv
  configs/dc3s.yaml
  configs/optimization.yaml
)
for p in "${REQUIRED[@]}"; do
  if [[ -f "$p" ]]; then
    echo "  [OK] $p"
  else
    echo "  [MISSING] $p"
    FAIL=1
  fi
done

# 2. Claim validator
echo ""
echo "--- validate_paper_claims ---"
if "$PYTHON" scripts/validate_paper_claims.py; then
  echo "  [OK] validate_paper_claims"
else
  echo "  [FAIL] validate_paper_claims"
  FAIL=1
fi

# 3. Sync paper assets
echo ""
echo "--- sync_paper_assets --check ---"
if "$PYTHON" scripts/sync_paper_assets.py --check; then
  echo "  [OK] sync_paper_assets"
else
  echo "  [FAIL] sync_paper_assets"
  FAIL=1
fi

# 4. Negative test (proves validator fails when claim is changed)
echo ""
echo "--- run_claim_validator_negative_test ---"
if "$PYTHON" scripts/run_claim_validator_negative_test.py; then
  echo "  [OK] claim_validator_negative_test"
else
  echo "  [FAIL] claim_validator_negative_test"
  FAIL=1
fi

# 5. Generate repro_check.json
echo ""
echo "--- generate_repro_check ---"
if "$PYTHON" scripts/generate_repro_check.py; then
  echo "  [OK] repro_check.json generated"
else
  echo "  [FAIL] repro_check.json"
  FAIL=1
fi

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "=== All integrity checks passed ==="
  exit 0
else
  echo "=== One or more checks failed ==="
  exit 1
fi
