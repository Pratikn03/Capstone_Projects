#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from _battery_wrappers_common import REPO_ROOT, run_script, copy_outputs, ensure_dir

def main() -> None:
    p = argparse.ArgumentParser(description='Generate 48h battery fault trace wrapper')
    p.add_argument('--region', default='DE')
    p.add_argument('--fault', default='stale_sensor')
    p.add_argument('--window', type=int, default=48)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default='reports/fault_benchmark')
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script('generate_48h_trace.py', '--region', args.region, '--fault', args.fault, '--window', str(args.window), '--seed', str(args.seed), '--out-dir', 'reports/publication')
    suffix = f"{args.region.lower()}_{args.fault}"
    copy_outputs([
        (REPO_ROOT/'reports/publication/48h_trace.csv', out/f'48h_trace_{suffix}.csv'),
        (REPO_ROOT/'reports/publication/fig_48h_trace.png', out/f'fig_48h_trace_{suffix}.png'),
        (REPO_ROOT/'reports/publication/48h_trace_summary.json', out/f'48h_trace_summary_{suffix}.json'),
    ])

if __name__ == '__main__':
    main()
