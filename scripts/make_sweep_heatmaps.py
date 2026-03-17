#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
from _battery_wrappers_common import REPO_ROOT, run_script, copy_outputs, ensure_dir

def main() -> None:
    p = argparse.ArgumentParser(description='Make sweep heatmaps wrapper')
    p.add_argument('--out-dir', default='reports/calibration')
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script('generate_priority2_artifacts.py')
    copy_outputs([(REPO_ROOT/'reports/publication/hyperparameter_surface.png', out/'hyperparameter_surface.png')])

if __name__ == '__main__':
    main()
