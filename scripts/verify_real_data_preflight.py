#!/usr/bin/env python3
"""Run repo-local preflight checks for all-domain real-data acquisition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orius.data_pipeline.real_data_contract import (
    DEFAULT_MIN_FREE_GIB,
    module_status,
    summarize_disk_usage,
    tool_status,
    utc_now_iso,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "reports" / "real_data_preflight.json"

DOMAIN_PATHS: dict[str, list[Path]] = {
    "battery": [
        REPO_ROOT / "data" / "raw" / "time_series_60min_singleindex.csv",
        REPO_ROOT / "data" / "raw" / "us_eia930",
    ],
    "industrial": [REPO_ROOT / "data" / "industrial" / "raw" / "ccpp"],
    "healthcare": [REPO_ROOT / "data" / "healthcare" / "raw" / "bidmc_csv"],
    "aerospace": [REPO_ROOT / "data" / "aerospace" / "raw"],
    "navigation": [REPO_ROOT / "data" / "navigation" / "raw" / "kitti_odometry"],
    "av": [REPO_ROOT / "data" / "av" / "raw" / "waymo_open_motion"],
}

REQUIRED_TOOLS = ("git", "hf", "kaggle")
REQUIRED_MODULES = ("pandas", "pyarrow", "openpyxl", "wfdb", "huggingface_hub")


def _domain_status(domain: str) -> dict[str, object]:
    paths = DOMAIN_PATHS[domain]
    items = [
        {
            "path": str(path),
            "exists": path.exists(),
        }
        for path in paths
    ]
    return {
        "domain": domain,
        "checks": items,
        "all_present": all(item["exists"] for item in items),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify real-data preconditions for all-domain training")
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        choices=sorted(DOMAIN_PATHS.keys()),
        help="Limit checks to one or more domains. Defaults to all.",
    )
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    domains = args.domains or sorted(DOMAIN_PATHS.keys())
    disk = summarize_disk_usage(REPO_ROOT, min_free_gib=args.min_free_gib)
    tools = tool_status(REQUIRED_TOOLS)
    modules = module_status(REQUIRED_MODULES)
    domain_rows = [_domain_status(domain) for domain in domains]

    report = {
        "generated_at_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "disk": disk,
        "tools": tools,
        "modules": modules,
        "domains": domain_rows,
        "all_domains_present": all(bool(row["all_present"]) for row in domain_rows),
        "all_tools_present": all(bool(path) for path in tools.values()),
        "all_modules_present": all(bool(value) for value in modules.values()),
    }
    report["passes"] = (
        bool(report["disk"]["passes_threshold"])
        and bool(report["all_domains_present"])
        and bool(report["all_tools_present"])
        and bool(report["all_modules_present"])
    )

    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
