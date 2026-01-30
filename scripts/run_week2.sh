#!/usr/bin/env bash
set -euo pipefail

python -m gridpulse.forecasting.train --config configs/train_forecast.yaml

echo "Week-2 complete: GBM vs LSTM report saved to reports/ml_vs_dl_comparison.md"
