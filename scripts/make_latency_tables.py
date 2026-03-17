#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from _battery_wrappers_common import REPO_ROOT, run_script, copy_outputs, ensure_dir

def main() -> None:
    p = argparse.ArgumentParser(description='Generate latency tables wrapper')
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--out-dir', default='reports/latency')
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script('benchmark_dc3s_steps.py', '--iterations', str(args.iterations), '--warmup', str(args.warmup), '--out', 'reports/dc3s_latency_benchmark_wrapper.json')
    copy_outputs([
        (REPO_ROOT/'reports/publication/dc3s_latency_summary.csv', out/'dc3s_latency_summary.csv'),
        (REPO_ROOT/'reports/publication/dc3s_latency_summary.json', out/'dc3s_latency_summary.json'),
    ])

if __name__ == '__main__':
    main()
