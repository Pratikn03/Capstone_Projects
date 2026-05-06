#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from _battery_wrappers_common import REPO_ROOT, ensure_dir, write_manifest


def _load_release_manifest() -> dict:
    path = REPO_ROOT / "reports/publication/release_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package canonical regional grounding outputs.")
    parser.add_argument("--out-dir", default="reports/publication")
    args = parser.parse_args()

    out_dir = ensure_dir(REPO_ROOT / args.out_dir)
    release_manifest = _load_release_manifest()
    source_runs = dict(release_manifest.get("source_runs", {}))

    per_ba_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []

    for region, payload in source_runs.items():
        reports_dir = REPO_ROOT / str(payload["reports_dir"])
        impact_path = reports_dir / "impact_summary.csv"
        compare_path = reports_dir / "publication/table4_region_compare.csv"
        impact = _read_csv(impact_path).iloc[0].to_dict()
        compare = _read_csv(compare_path)

        region_rows.append(
            {
                "region": region,
                "release_id": payload["release_id"],
                "run_id": payload["run_id"],
                "accepted": bool(payload.get("accepted", False)),
                "reports_dir": str(payload["reports_dir"]),
                "cost_savings_pct": float(impact.get("cost_savings_pct", 0.0)),
                "carbon_reduction_pct": float(impact.get("carbon_reduction_pct", 0.0)),
                "peak_shaving_pct": float(impact.get("peak_shaving_pct", 0.0)),
                "carbon_source": str(impact.get("carbon_source", "unknown")),
                "targets_present": ",".join(sorted(str(x) for x in compare["target"].unique())),
                "price_target_present": bool(
                    (compare["target"].astype(str) == "price_eur_mwh").any()
                    or (compare["target"].astype(str) == "price_usd_mwh").any()
                ),
                "forecast_table_path": str(compare_path.relative_to(REPO_ROOT)),
                "impact_summary_path": str(impact_path.relative_to(REPO_ROOT)),
            }
        )

        if region.startswith("US_"):
            per_ba_rows.append(
                {
                    "balancing_authority": region.replace("US_", ""),
                    "release_id": payload["release_id"],
                    "run_id": payload["run_id"],
                    "cost_savings_pct": float(impact.get("cost_savings_pct", 0.0)),
                    "carbon_reduction_pct": float(impact.get("carbon_reduction_pct", 0.0)),
                    "peak_shaving_pct": float(impact.get("peak_shaving_pct", 0.0)),
                    "carbon_source": str(impact.get("carbon_source", "unknown")),
                    "impact_summary_path": str(impact_path.relative_to(REPO_ROOT)),
                }
            )

    per_ba_df = pd.DataFrame(per_ba_rows).sort_values("balancing_authority")
    per_ba_path = out_dir / "per_ba_dispatch_impact.csv"
    per_ba_df.to_csv(per_ba_path, index=False, float_format="%.6f")

    region_df = pd.DataFrame(region_rows).sort_values("region")
    region_path = out_dir / "real_price_grounding_runs.csv"
    region_df.to_csv(region_path, index=False, float_format="%.6f")

    write_manifest(
        out_dir,
        "regional_grounding_manifest.json",
        {
            "source_manifest": "reports/publication/release_manifest.json",
            "per_ba_dispatch_impact": str(per_ba_path.relative_to(REPO_ROOT)),
            "real_price_grounding_runs": str(region_path.relative_to(REPO_ROOT)),
            "regions_packaged": sorted(region_df["region"].tolist()),
            "us_balancing_authorities_packaged": sorted(per_ba_df["balancing_authority"].tolist()),
        },
    )


if __name__ == "__main__":
    main()
