#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import sys
from _battery_wrappers_common import REPO_ROOT, ensure_dir
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.compute_reliability_group_coverage import build_summary

def main() -> None:
    p = argparse.ArgumentParser(description='Run Mondrian subgroup evaluation wrapper')
    p.add_argument('--input-trace', default='reports/publication/48h_trace_final_us.csv')
    p.add_argument('--out-dir', default='reports/calibration')
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    df = pd.read_csv(REPO_ROOT / args.input_trace)
    inp = pd.DataFrame({'y_true': df['soc_true_mwh'], 'y_pred': df['soc_observed_mwh'], 'reliability_w': df['reliability_w']})
    width = (df['interval_width_mw'] if 'interval_width_mw' in df.columns else pd.Series([1.0] * len(df))).astype(float).clip(lower=0.1)
    inp['lower'] = inp['y_pred'] - 0.5 * width
    inp['upper'] = inp['y_pred'] + 0.5 * width
    rows, summary = build_summary(inp, y_true_col='y_true', y_pred_col='y_pred', reliability_col='reliability_w', lower_col='lower', upper_col='upper', alpha=0.10, n_bins=4, min_bin_size=12, binning='uniform')
    rows.to_csv(out/'mondrian_subgroup_eval.csv', index=False)
    (out/'mondrian_subgroup_eval.json').write_text(__import__('json').dumps(summary, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
