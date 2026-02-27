#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p paper/assets/{figures,tables,tables/generated,data,configs}

if [[ -x ".venv/bin/python3" ]]; then
  PYTHON_BIN=".venv/bin/python3"
else
  PYTHON_BIN="python3"
fi

copy_or_die() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: missing source file: $src" >&2
    exit 1
  fi
  cp -f "$src" "$dst"
}

copy_first_available() {
  local dst="$1"
  shift
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      cp -f "$candidate" "$dst"
      return 0
    fi
  done
  echo "ERROR: none of the candidate files exist for $dst" >&2
  printf '  - %s\n' "$@" >&2
  exit 1
}

# Figures
copy_first_available "paper/assets/figures/fig01_architecture.png" \
  "reports/figures/architecture.png" \
  "reports/publication/figures/fig01_geographic_scope.png"

copy_first_available "paper/assets/figures/fig02_dc3s_step.png" \
  "reports/publication/figures/fig11_dispatch_comparison.png" \
  "reports/figures/dispatch_compare.png" \
  "paper/assets/figures/fig01_architecture.png"

copy_or_die "reports/publication/fig_true_soc_violation_vs_dropout.png" "paper/assets/figures/fig03_true_soc_violation_vs_dropout.png"
copy_or_die "reports/publication/fig_true_soc_severity_p95_vs_dropout.png" "paper/assets/figures/fig04_true_soc_severity_p95_vs_dropout.png"
copy_or_die "reports/publication/fig_cqr_group_coverage.png" "paper/assets/figures/fig05_cqr_group_coverage.png"
copy_or_die "reports/publication/fig_transfer_coverage.png" "paper/assets/figures/fig06_transfer_coverage.png"
copy_or_die "reports/publication/fig_cost_safety_pareto.png" "paper/assets/figures/fig07_cost_safety_frontier.png"
copy_or_die "reports/publication/fig_rac_sensitivity_vs_width.png" "paper/assets/figures/fig08_rac_sensitivity_vs_width.png"

# Tables
copy_first_available "paper/assets/tables/tbl01_main_results.csv" \
  "reports/publication/table1_main.csv" \
  "reports/publication/dc3s_main_table.csv"
copy_or_die "reports/publication/table2_ablations.csv" "paper/assets/tables/tbl02_ablations.csv"
copy_or_die "reports/publication/cqr_group_coverage.csv" "paper/assets/tables/tbl03_cqr_group_coverage.csv"
copy_or_die "reports/publication/transfer_stress.csv" "paper/assets/tables/tbl04_transfer_stress.csv"
copy_or_die "reports/publication/tables/table1_dataset_summary.csv" "paper/assets/tables/tbl05_dataset_summary.csv"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import csv
import yaml

root = Path('.')
out = root / 'paper/assets/tables/tbl06_hyperparams.csv'
rows = []

def read_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def add_row(group: str, key: str, value, source: str):
    rows.append({
        'group': group,
        'key': key,
        'value': value if value is not None else '',
        'source': source,
    })

cfg_train_de = read_yaml(root / 'configs/train_forecast.yaml')
cfg_train_us = read_yaml(root / 'configs/train_forecast_eia930.yaml')
cfg_dc3s = read_yaml(root / 'configs/dc3s.yaml')
cfg_unc = read_yaml(root / 'configs/uncertainty.yaml')

add_row('train_forecast_de', 'horizon_hours', cfg_train_de.get('horizon_hours'), 'configs/train_forecast.yaml')
add_row('train_forecast_de', 'lookback_hours', cfg_train_de.get('lookback_hours'), 'configs/train_forecast.yaml')
add_row('train_forecast_de', 'cross_validation.n_folds', (cfg_train_de.get('cross_validation') or {}).get('n_folds'), 'configs/train_forecast.yaml')
add_row('train_forecast_de', 'cross_validation.gap', (cfg_train_de.get('cross_validation') or {}).get('gap'), 'configs/train_forecast.yaml')

add_row('train_forecast_us', 'horizon_hours', cfg_train_us.get('horizon_hours'), 'configs/train_forecast_eia930.yaml')
add_row('train_forecast_us', 'lookback_hours', cfg_train_us.get('lookback_hours'), 'configs/train_forecast_eia930.yaml')
add_row('train_forecast_us', 'cross_validation.n_folds', (cfg_train_us.get('cross_validation') or {}).get('n_folds'), 'configs/train_forecast_eia930.yaml')
add_row('train_forecast_us', 'cross_validation.gap', (cfg_train_us.get('cross_validation') or {}).get('gap'), 'configs/train_forecast_eia930.yaml')

rac = ((cfg_dc3s.get('dc3s') or {}).get('rac_cert') or {})
add_row('dc3s_rac', 'sensitivity_probe', rac.get('sensitivity_probe'), 'configs/dc3s.yaml')
add_row('dc3s_rac', 'sens_eps_mw', rac.get('sens_eps_mw'), 'configs/dc3s.yaml')
add_row('dc3s_rac', 'sens_norm_ref', rac.get('sens_norm_ref'), 'configs/dc3s.yaml')
add_row('dc3s_rac', 'beta_reliability', (cfg_dc3s.get('dc3s') or {}).get('beta_reliability'), 'configs/dc3s.yaml')
add_row('dc3s_rac', 'beta_sensitivity', (cfg_dc3s.get('dc3s') or {}).get('beta_sensitivity'), 'configs/dc3s.yaml')
add_row('dc3s_rac', 'k_sensitivity', (cfg_dc3s.get('dc3s') or {}).get('k_sensitivity'), 'configs/dc3s.yaml')

regime = (cfg_unc.get('regime_cqr') or {})
add_row('regime_cqr', 'enabled', regime.get('enabled'), 'configs/uncertainty.yaml')
add_row('regime_cqr', 'n_bins', regime.get('n_bins'), 'configs/uncertainty.yaml')
add_row('regime_cqr', 'vol_window', regime.get('vol_window'), 'configs/uncertainty.yaml')
add_row('regime_cqr', 'quantile_backend_policy', regime.get('quantile_backend_policy'), 'configs/uncertainty.yaml')

out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['group', 'key', 'value', 'source'])
    writer.writeheader()
    writer.writerows(rows)
PY

# Data snapshots
copy_first_available "paper/assets/data/data_manifest.json" \
  "data/dashboard/manifest.json" \
  "paper/metrics_manifest.json"
copy_or_die "reports/publication/stats_summary.json" "paper/assets/data/metrics_snapshot.json"
copy_or_die "paper/claim_matrix.csv" "paper/assets/data/claim_matrix.csv"

# Config snapshots
copy_or_die "configs/train_forecast.yaml" "paper/assets/configs/train_forecast_de.yaml"
copy_or_die "configs/train_forecast_eia930.yaml" "paper/assets/configs/train_forecast_us.yaml"
copy_or_die "configs/dc3s_ablations.yaml" "paper/assets/configs/dc3s_ablations.yaml"
copy_or_die "configs/dc3s.yaml" "paper/assets/configs/dc3s.yaml"
copy_or_die "configs/uncertainty.yaml" "paper/assets/configs/uncertainty.yaml"

echo "Exported curated paper assets under paper/assets/."
