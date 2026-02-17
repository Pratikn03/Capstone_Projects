#!/usr/bin/env python3
"""Build a frozen documentation metrics snapshot from report artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics source file: {path}")
    return pd.read_csv(path)


def _latest_common_run_id(de_research: pd.DataFrame, us_research: pd.DataFrame) -> str:
    de_runs = set(
        de_research.loc[de_research["row_type"] == "run_summary", "run_id"].astype(str).tolist()
    )
    us_runs = set(
        us_research.loc[us_research["row_type"] == "run_summary", "run_id"].astype(str).tolist()
    )
    common = de_runs.intersection(us_runs)
    if not common:
        raise RuntimeError("No common run_id found between DE and US research summary rows")

    # Preserve source ordering by scanning DE summary rows from bottom.
    de_ordered = de_research.loc[de_research["row_type"] == "run_summary", "run_id"].astype(str).tolist()
    for run_id in reversed(de_ordered):
        if run_id in common:
            return run_id
    raise RuntimeError("Failed to resolve latest common run_id")


def _single_summary_row(df: pd.DataFrame, run_id: str, label: str) -> pd.Series:
    rows = df[(df["row_type"] == "run_summary") & (df["run_id"].astype(str) == run_id)]
    if rows.empty:
        raise RuntimeError(f"Missing run_summary row for {label} run_id={run_id}")
    return rows.iloc[-1]


def _as_float(row: pd.Series, key: str) -> float:
    return float(row[key])


def _source(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": int(stat.st_size),
    }


def build_snapshot(run_id: str | None, output: Path) -> dict[str, Any]:
    imp_de_path = Path("reports/impact_summary.csv")
    imp_us_path = Path("reports/eia930/impact_summary.csv")
    r_de_path = Path("reports/research_metrics_de.csv")
    r_us_path = Path("reports/research_metrics_us.csv")

    impact_de = _read_csv(imp_de_path)
    impact_us = _read_csv(imp_us_path)
    research_de = _read_csv(r_de_path)
    research_us = _read_csv(r_us_path)

    resolved_run = run_id or _latest_common_run_id(research_de, research_us)

    de_impact = impact_de.iloc[0]
    us_impact = impact_us.iloc[0]
    de_summary = _single_summary_row(research_de, resolved_run, "DE")
    us_summary = _single_summary_row(research_us, resolved_run, "US")

    payload = {
        "frozen_run_id": resolved_run,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "impact_de": _source(imp_de_path),
            "impact_us": _source(imp_us_path),
            "research_de": _source(r_de_path),
            "research_us": _source(r_us_path),
        },
        "datasets": {
            "de": {
                "cost_savings_pct": _as_float(de_impact, "cost_savings_pct"),
                "carbon_reduction_pct": _as_float(de_impact, "carbon_reduction_pct"),
                "peak_shaving_pct": _as_float(de_impact, "peak_shaving_pct"),
                "evpi_robust": _as_float(de_summary, "evpi_robust"),
                "evpi_deterministic": _as_float(de_summary, "evpi_deterministic"),
                "vss": _as_float(de_summary, "vss"),
                "research_timestamp_utc": str(de_summary["timestamp_utc"]),
            },
            "us": {
                "cost_savings_pct": _as_float(us_impact, "cost_savings_pct"),
                "carbon_reduction_pct": _as_float(us_impact, "carbon_reduction_pct"),
                "peak_shaving_pct": _as_float(us_impact, "peak_shaving_pct"),
                "evpi_robust": _as_float(us_summary, "evpi_robust"),
                "evpi_deterministic": _as_float(us_summary, "evpi_deterministic"),
                "vss": _as_float(us_summary, "vss"),
                "research_timestamp_utc": str(us_summary["timestamp_utc"]),
            },
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen metrics snapshot for docs/paper sync")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run_id to freeze. If omitted, latest common DE/US run_summary run_id is used.",
    )
    parser.add_argument(
        "--output",
        default="reports/frozen_metrics_snapshot.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    payload = build_snapshot(run_id=args.run_id, output=Path(args.output))
    print(
        f"frozen_run_id={payload['frozen_run_id']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
