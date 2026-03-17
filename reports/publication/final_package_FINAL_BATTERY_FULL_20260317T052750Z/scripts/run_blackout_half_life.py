#!/usr/bin/env python3
from __future__ import annotations
import argparse
from _battery_wrappers_common import REPO_ROOT, run_script, copy_outputs, ensure_dir

def main() -> None:
    p = argparse.ArgumentParser(description='Run blackout half-life wrapper')
    p.add_argument('--out-dir', default='reports/blackout')
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script('run_blackout_study.py')
    run_script('generate_priority2_artifacts.py')
    copy_outputs([
        (REPO_ROOT/'reports/publication/blackout_study.csv', out/'blackout_study.csv'),
        (REPO_ROOT/'reports/publication/blackout_half_life.csv', out/'blackout_half_life.csv'),
        (REPO_ROOT/'reports/publication/fig_blackout_halflife.png', out/'fig_blackout_halflife.png'),
        (REPO_ROOT/'reports/publication/blackout_half_life.png', out/'blackout_half_life.png'),
    ])

if __name__ == '__main__':
    main()
