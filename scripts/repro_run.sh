#!/usr/bin/env bash
set -euo pipefail

# Reproducible end-to-end run (data -> train -> reports)
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

python -m gridpulse.pipeline.run --all
python scripts/update_readme_impact.py
