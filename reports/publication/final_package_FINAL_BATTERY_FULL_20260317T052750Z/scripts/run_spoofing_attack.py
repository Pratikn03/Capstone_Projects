#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd
from _battery_wrappers_common import REPO_ROOT, run_script, ensure_dir, write_manifest

def main() -> None:
    p = argparse.ArgumentParser(description='Run spoofing attack wrapper')
    p.add_argument('--out-dir', default='reports/probing')
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script('generate_priority3_artifacts.py')
    base = pd.read_csv(REPO_ROOT/'reports/publication/active_probing_spoofing_detection.csv')
    base.to_csv(out/'spoofing_attack_results.csv', index=False)
    write_manifest(out, 'run_spoofing_attack_manifest.json', {'source': 'reports/publication/active_probing_spoofing_detection.csv'})

if __name__ == '__main__':
    main()
