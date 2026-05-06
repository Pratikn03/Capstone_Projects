#!/usr/bin/env python3
"""Build the publication dataset summary table from dashboard profile locks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "publication" / "tables" / "table1_dataset_summary.csv"
DATASET_SPECS = (
    {
        "code": "DE",
        "country": "DE",
        "stats_path": REPO_ROOT / "data" / "dashboard" / "de_stats.json",
        "features_path": REPO_ROOT / "data" / "processed" / "features.parquet",
        "label": "Germany (OPSD)",
    },
    {
        "code": "US_MISO",
        "country": "US",
        "stats_path": REPO_ROOT / "data" / "dashboard" / "us_stats.json",
        "features_path": REPO_ROOT / "data" / "processed" / "us_eia930" / "features.parquet",
        "label": "USA (EIA-930 MISO)",
    },
    {
        "code": "US_PJM",
        "country": "US",
        "stats_path": REPO_ROOT / "data" / "dashboard" / "us_pjm_stats.json",
        "features_path": REPO_ROOT / "data" / "processed" / "us_eia930_pjm" / "features.parquet",
        "label": "USA (EIA-930 PJM)",
    },
    {
        "code": "US_ERCOT",
        "country": "US",
        "stats_path": REPO_ROOT / "data" / "dashboard" / "us_ercot_stats.json",
        "features_path": REPO_ROOT / "data" / "processed" / "us_eia930_ercot" / "features.parquet",
        "label": "USA (EIA-930 ERCOT)",
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication dataset summary CSV from dashboard stats")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def _load_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {path}")
    return payload


def _compute_stats_from_features(path: Path, code: str, label: str) -> dict[str, Any]:
    df = pd.read_parquet(path)
    target_summary = {}
    missing_pct = {}
    for signal in ("load_mw", "wind_mw", "solar_mw"):
        if signal in df.columns:
            target_summary[signal] = {"present": True}
            missing_pct[signal] = float(df[signal].isna().mean() * 100.0)
    ts = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if "timestamp" in df.columns
        else pd.Series([], dtype="datetime64[ns, UTC]")
    )
    return {
        "label": label,
        "rows": int(len(df)),
        "date_range": {
            "start": ts.min().isoformat() if not ts.empty else None,
            "end": ts.max().isoformat() if not ts.empty else None,
        },
        "targets_summary": target_summary,
        "missing_pct": missing_pct,
        "country": "DE" if code == "DE" else "US",
        "dataset_key": code,
    }


def _date_only(value: Any) -> str:
    text = str(value or "")
    return text[:10] if "T" in text else text


def build_dataset_summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in DATASET_SPECS:
        stats_path = spec["stats_path"]
        features_path = spec["features_path"]
        if stats_path.exists():
            stats = _load_stats(stats_path)
        elif features_path.exists():
            stats = _compute_stats_from_features(features_path, str(spec["code"]), str(spec["label"]))
        else:
            continue
        dataset_key = str(spec["code"])
        country = str(spec.get("country", stats.get("country", dataset_key)))
        date_range = stats.get("date_range", {}) if isinstance(stats.get("date_range"), dict) else {}
        target_summaries = (
            stats.get("targets_summary", {}) if isinstance(stats.get("targets_summary"), dict) else {}
        )
        total_rows = int(stats.get("rows", 0) or 0)
        missing_pct = stats.get("missing_pct", {}) if isinstance(stats.get("missing_pct"), dict) else {}
        for signal in ("load_mw", "wind_mw", "solar_mw"):
            if signal not in target_summaries:
                continue
            signal_missing_pct = float(missing_pct.get(signal, 0.0) or 0.0)
            non_null = int(round(total_rows * max(0.0, 1.0 - signal_missing_pct / 100.0)))
            coverage = round(100.0 * non_null / total_rows, 6) if total_rows else 0.0
            rows.append(
                {
                    "DatasetKey": dataset_key,
                    "Dataset": str(stats.get("label", spec["label"])),
                    "Country": country,
                    "Start": _date_only(date_range.get("start")),
                    "End": _date_only(date_range.get("end")),
                    "Rows": total_rows,
                    "Signal": signal,
                    "Non-Null": non_null,
                    "Coverage%": coverage,
                }
            )
    return rows


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    rows = build_dataset_summary_rows()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "DatasetKey",
                "Dataset",
                "Country",
                "Start",
                "End",
                "Rows",
                "Signal",
                "Non-Null",
                "Coverage%",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)


if __name__ == "__main__":
    main()
