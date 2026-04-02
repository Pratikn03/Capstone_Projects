#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd
from _battery_wrappers_common import REPO_ROOT, run_script, copy_outputs, ensure_dir, write_manifest

def main() -> None:
    p = argparse.ArgumentParser(description='Run two battery fleet wrapper')
    p.add_argument('--out-dir', default='reports/fleet')
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script('generate_priority3_artifacts.py')
    fleet = pd.read_csv(REPO_ROOT/'reports/publication/fleet_composition_two_battery.csv')
    metrics = pd.DataFrame([{
        'joint_over_limit_steps': int((fleet['curtailed_mw'] > 0).sum()),
        'max_curtailed_mw': float(fleet['curtailed_mw'].max()),
        'useful_work_preserved_mwh': float(fleet['total_dispatch_mw'].sum()),
    }])
    metrics.to_csv(out/'two_battery_composition_metrics.csv', index=False)
    copy_outputs([(REPO_ROOT/'reports/publication/fleet_composition_two_battery.csv', out/'fleet_composition_two_battery.csv')])
    write_manifest(out, 'run_two_battery_fleet_manifest.json', {'rows': int(len(fleet))})

if __name__ == '__main__':
    main()
