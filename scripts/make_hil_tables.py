#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd
from _battery_wrappers_common import REPO_ROOT, ensure_dir

def main() -> None:
    p = argparse.ArgumentParser(description='Make HIL tables wrapper')
    p.add_argument('--out-dir', default='reports/hil')
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    step = pd.read_csv(out/'hil_step_log.csv')
    pd.DataFrame([{
        'steps': int(len(step)),
        'interventions': int(step['intervened'].sum()),
        'violations': int(step['violated'].sum()),
        'cert_completeness_pct': float(100 * step['cert_complete'].mean()),
    }]).to_csv(out/'hil_timing_table.csv', index=False)
    pd.DataFrame([
        {'component': 'controller_runtime', 'value': 'FastAPI TestClient + EdgeAgent sim'},
        {'component': 'battery_driver', 'value': 'SimBatteryDriver'},
        {'component': 'fault_protocol', 'value': 'nominal/dropout/spike'},
    ]).to_csv(out/'hil_hardware_table.csv', index=False)

if __name__ == '__main__':
    main()
