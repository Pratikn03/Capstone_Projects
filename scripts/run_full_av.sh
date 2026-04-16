set -euo pipefail
cd /Users/pratik_n/Downloads/gridpulse
source .venv/bin/activate
export PYTHONPATH=src

# --- Config ---
PROCESSED=data/orius_av/av/processed_full_corpus
MODELS=artifacts/models_orius_av_full_corpus
UNCERTAINTY=artifacts/uncertainty/orius_av_full_corpus
REPORTS=reports/orius_av/full_corpus
OVERALL=reports/battery_av/overall

echo "=== 1. Train AV models (full corpus) ==="
python scripts/run_battery_av_pipeline.py \
  --skip-battery \
  --av-full-corpus \
  --av-processed-dir "$PROCESSED" \
  --av-models-dir "$MODELS" \
  --av-uncertainty-dir "$UNCERTAINTY" \
  --av-reports-dir "$REPORTS" \
  --av-skip-runtime \
  --av-skip-validation \
  --overall-dir "$OVERALL"

echo "=== 2. Run DC3S runtime (full corpus) ==="
python -c "
import json
from pathlib import Path
from orius.av_waymo import run_runtime_dry_run

report = run_runtime_dry_run(
    replay_windows_path=Path('${PROCESSED}/replay_windows.parquet'),
    step_features_path=Path('${PROCESSED}/step_features.parquet'),
    models_dir=Path('${MODELS}'),
    out_dir=Path('${REPORTS}'),
)
print(json.dumps({k: v for k, v in report.items() if k != 'traces'}, indent=2, default=str))
"

echo "=== 3. Generate report (tables + figures) ==="
python scripts/build_waymo_av_dry_run_report.py \
  --processed-dir "$PROCESSED" \
  --reports-dir "$REPORTS" \
  --models-dir "$MODELS" \
  --uncertainty-dir "$UNCERTAINTY"

echo "=== 4. Rebuild combined manifest ==="
python scripts/run_battery_av_pipeline.py \
  --skip-battery \
  --av-full-corpus \
  --av-processed-dir "$PROCESSED" \
  --av-models-dir "$MODELS" \
  --av-uncertainty-dir "$UNCERTAINTY" \
  --av-reports-dir "$REPORTS" \
  --av-skip-validation \
  --av-skip-training \
  --av-skip-runtime \
  --overall-dir "$OVERALL"

echo "=== 5. Validate ==="
python -m pytest tests/ --tb=no -q \
  --ignore=tests/test_external_real_data_integration.py -W ignore 2>&1 | tail -3

echo "=== DONE ==="
cat "$OVERALL/domain_summary.csv"
