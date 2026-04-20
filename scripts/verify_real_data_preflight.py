#!/usr/bin/env python3
"""Run repo-local preflight checks for the active 3-domain ORIUS program."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.data_pipeline.external_raw import get_external_data_root
from orius.data_pipeline.real_data_contract import (
    DEFAULT_MIN_FREE_GIB,
    module_status,
    resolve_disk_target,
    resolve_repo_or_external_raw_dir,
    resolved_source_has_files,
    tool_status,
    utc_now_iso,
    write_json,
)

DEFAULT_OUT = REPO_ROOT / "reports" / "real_data_preflight.json"

REQUIRED_TOOLS = ("git", "hf", "kaggle")
REQUIRED_MODULES = ("pandas", "pyarrow", "openpyxl", "wfdb", "huggingface_hub")
DOMAIN_NAMES = ["battery", "av", "healthcare"]

BATTERY_PATHS = [
    REPO_ROOT / "data" / "raw" / "time_series_60min_singleindex.csv",
    REPO_ROOT / "data" / "raw" / "us_eia930",
]
HEALTHCARE_PATHS = [REPO_ROOT / "data" / "healthcare" / "raw" / "bidmc_csv"]
AV_DATASET_CHECKS = (
    {
        "repo_dir": REPO_ROOT / "data" / "av" / "raw" / "waymo_open_motion",
        "external_dataset_key": "waymo_open_motion",
        "required_for_pass": True,
        "role": "canonical",
    },
    {
        "repo_dir": REPO_ROOT / "data" / "av" / "raw" / "argoverse2_motion",
        "external_dataset_key": "argoverse2_motion",
        "required_for_pass": False,
        "role": "companion",
    },
    {
        "repo_dir": REPO_ROOT / "data" / "av" / "raw" / "argoverse2_sensor",
        "external_dataset_key": "argoverse2_sensor",
        "required_for_pass": False,
        "role": "companion",
    },
)


def _path_checks(paths: list[Path]) -> list[dict[str, object]]:
    return [{"path": str(path), "exists": path.exists()} for path in paths]


def _external_dataset_check(
    *,
    repo_dir: Path,
    external_dataset_key: str,
    explicit_root: Path | None,
    required_for_pass: bool = True,
    role: str = "canonical",
) -> dict[str, object]:
    raw_source = resolve_repo_or_external_raw_dir(
        repo_dir,
        external_dataset_key=external_dataset_key,
        explicit_root=explicit_root,
        required=False,
    )
    checked_locations = list(raw_source.checked_locations) if raw_source is not None else [str(repo_dir)]
    if raw_source is None:
        external_root = get_external_data_root(explicit_root, required=False)
        if external_root is not None:
            checked_locations.append(str(external_root / external_dataset_key))
        else:
            checked_locations.append(f"$ORIUS_EXTERNAL_DATA_ROOT/{external_dataset_key}")
    has_files = resolved_source_has_files(raw_source)
    resolved_path = str(raw_source.path) if raw_source is not None else checked_locations[-1]
    return {
        "path": resolved_path,
        "exists": raw_source is not None,
        "has_files": has_files,
        "source_kind": None if raw_source is None else raw_source.source_kind,
        "checked_locations": checked_locations,
        "required_for_pass": required_for_pass,
        "role": role,
    }


def _domain_status(domain: str, *, explicit_root: Path | None) -> dict[str, object]:
    if domain == "battery":
        items = _path_checks(BATTERY_PATHS)
        all_present = all(item["exists"] for item in items)
    elif domain == "healthcare":
        items = _path_checks(HEALTHCARE_PATHS)
        all_present = all(item["exists"] for item in items)
    elif domain == "av":
        items = [
            _external_dataset_check(
                repo_dir=row["repo_dir"],
                external_dataset_key=row["external_dataset_key"],
                explicit_root=explicit_root,
                required_for_pass=bool(row["required_for_pass"]),
                role=str(row["role"]),
            )
            for row in AV_DATASET_CHECKS
        ]
        all_present = all(
            bool(item["exists"]) and bool(item["has_files"])
            for item in items
            if bool(item["required_for_pass"])
        )
    else:
        raise KeyError(domain)
    return {
        "domain": domain,
        "checks": items,
        "all_present": all_present,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify real-data preconditions for the active 3-domain ORIUS program")
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        choices=DOMAIN_NAMES,
        help="Limit checks to one or more domains. Defaults to all.",
    )
    parser.add_argument("--external-root", type=Path, default=None, help="Override ORIUS_EXTERNAL_DATA_ROOT for this check")
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    domains = args.domains or DOMAIN_NAMES
    disk = resolve_disk_target(
        REPO_ROOT,
        explicit_external_root=args.external_root,
        min_free_gib=args.min_free_gib,
    )
    tools = tool_status(REQUIRED_TOOLS)
    modules = module_status(REQUIRED_MODULES)
    domain_rows = [_domain_status(domain, explicit_root=args.external_root) for domain in domains]

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
