#!/usr/bin/env python3
"""Build features for all multi-domain datasets.

Run before training: python scripts/build_features_multi_domain.py

Requires datasets to exist. Run first:
  make av-datasets
  make industrial-datasets
  make healthcare-datasets
  python scripts/download_aerospace_datasets.py
  python scripts/build_navigation_real_dataset.py

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
    from orius.data_pipeline.build_features_industrial import build_features as build_industrial
    from orius.data_pipeline.build_features_healthcare import build_features as build_healthcare
    from orius.data_pipeline.build_features_aerospace import build_features as build_aerospace
    from orius.data_pipeline.build_features_navigation import build_features as build_navigation

    results = []

    # AV
    av_csv = REPO_ROOT / "data" / "av" / "processed" / "av_trajectories_orius.csv"
    if av_csv.exists():
        build_av(av_csv, av_csv.parent)
        results.append(("AV", "OK"))
    else:
        results.append(("AV", "SKIP (run: make av-datasets)"))

    # Industrial
    ind_csv = REPO_ROOT / "data" / "industrial" / "processed" / "industrial_orius.csv"
    if ind_csv.exists():
        build_industrial(ind_csv, ind_csv.parent)
        results.append(("Industrial", "OK"))
    else:
        results.append(("Industrial", "SKIP (run: make industrial-datasets)"))

    # Healthcare
    hc_csv = REPO_ROOT / "data" / "healthcare" / "processed" / "healthcare_orius.csv"
    if hc_csv.exists():
        build_healthcare(hc_csv, hc_csv.parent)
        results.append(("Healthcare", "OK"))
    else:
        results.append(("Healthcare", "SKIP (run: make healthcare-datasets)"))

    # Aerospace
    aero_csv = REPO_ROOT / "data" / "aerospace" / "processed" / "aerospace_orius.csv"
    if aero_csv.exists():
        build_aerospace(aero_csv, aero_csv.parent)
        results.append(("Aerospace", "OK"))
    else:
        results.append(("Aerospace", "SKIP (run: python scripts/download_aerospace_datasets.py)"))

    # Navigation
    nav_csv = REPO_ROOT / "data" / "navigation" / "processed" / "navigation_orius.csv"
    if nav_csv.exists():
        build_navigation(nav_csv, nav_csv.parent)
        results.append(("Navigation", "OK"))
    else:
        results.append(("Navigation", "SKIP (run: python scripts/build_navigation_real_dataset.py)"))

    print("\nMulti-domain build summary:")
    for domain, status in results:
        print(f"  {domain:12s} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
