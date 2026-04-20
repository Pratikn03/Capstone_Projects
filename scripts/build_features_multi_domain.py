#!/usr/bin/env python3
"""Build features for the active non-battery ORIUS datasets.

Run before training: python scripts/build_features_multi_domain.py

Requires datasets to exist. Run first:
  make av-datasets
  make healthcare-datasets

For repo-local corpus readiness checks, run:
  python scripts/verify_real_data_preflight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    from orius.data_pipeline.build_features_av import build_features as build_av
    from orius.data_pipeline.build_features_healthcare import build_promoted_features as build_healthcare

    results = []

    # AV
    av_csv = REPO_ROOT / "data" / "av" / "processed" / "av_trajectories_orius.csv"
    if av_csv.exists():
        build_av(av_csv, av_csv.parent)
        results.append(("AV", "OK"))
    else:
        results.append(("AV", "SKIP (run: make av-datasets)"))

    # Healthcare
    hc_csv = REPO_ROOT / "data" / "healthcare" / "mimic3" / "processed" / "mimic3_healthcare_orius.csv"
    hc_out = REPO_ROOT / "data" / "healthcare" / "processed"
    if hc_csv.exists():
        build_healthcare(hc_csv, hc_out)
        results.append(("Healthcare", "OK"))
    else:
        results.append(("Healthcare", "SKIP (run: make healthcare-datasets)"))

    print("\nMulti-domain build summary:")
    for domain, status in results:
        print(f"  {domain:12s} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
