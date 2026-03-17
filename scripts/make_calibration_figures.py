#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from _battery_wrappers_common import REPO_ROOT, run_script, copy_outputs, ensure_dir

def main() -> None:
    p = argparse.ArgumentParser(description='Generate calibration figures wrapper')
    p.add_argument('--out-dir', default='reports/calibration')
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script('generate_priority2_artifacts.py')
    copy_outputs([
        (REPO_ROOT/'reports/publication/hyperparameter_surface.png', out/'hyperparameter_surface.png'),
        (REPO_ROOT/'reports/publication/graceful_degradation_trace.png', out/'graceful_degradation_trace.png'),
        (REPO_ROOT/'reports/publication/blackout_half_life.png', out/'blackout_half_life.png'),
    ])

if __name__ == '__main__':
    main()
