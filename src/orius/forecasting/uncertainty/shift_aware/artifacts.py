from __future__ import annotations

from pathlib import Path

import pandas as pd

from .subgroup import SubgroupCoverageTracker


def _write_sorted_csv(rows: list[dict], output_path: Path, sort_cols: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=sort_cols)
    sort_cols_present = [c for c in sort_cols if c in df.columns]
    if sort_cols_present:
        df = df.sort_values(sort_cols_present, kind="mergesort")
    df.to_csv(output_path, index=False)


def write_shift_aware_artifacts(
    *,
    tracker: SubgroupCoverageTracker,
    validity_trace: list[dict],
    adaptive_trace: list[dict],
    publication_dir: str,
) -> None:
    pub = Path(publication_dir)
    rows = tracker.group_rows()

    parsed_rows: list[dict] = []
    for row in rows:
        key = str(row.get("group_key", ""))
        parts = {p.split(":", 1)[0]: p.split(":", 1)[1] for p in key.split("|") if ":" in p}
        parsed_rows.append(
            {
                **row,
                "reliability_bin": parts.get("rel", "unknown"),
                "volatility_bin": parts.get("vol", "unknown"),
                "fault_type": parts.get("fault", "unknown"),
                "hour_bucket": parts.get("hour", "00"),
                "custom_key": parts.get("custom", "none"),
            }
        )

    frame = pd.DataFrame(parsed_rows)
    if frame.empty:
        rel_rows: list[dict] = []
        vol_rows: list[dict] = []
        fault_rows: list[dict] = []
    else:
        target = float(frame.get("target_coverage", pd.Series([0.9])).iloc[0])
        rel_rows = (
            frame.groupby("reliability_bin", as_index=False)
            .agg(
                count=("count", "sum"),
                covered=("covered", "sum"),
                miss_count=("miss_count", "sum"),
                avg_interval_width=("avg_interval_width", "mean"),
                avg_abs_residual=("avg_abs_residual", "mean"),
            )
            .assign(empirical_coverage=lambda d: d["covered"] / d["count"].clip(lower=1))
            .assign(under_coverage_gap=lambda d: (target - d["empirical_coverage"]).clip(lower=0.0))
            .to_dict(orient="records")
        )
        vol_rows = (
            frame.groupby("volatility_bin", as_index=False)
            .agg(
                count=("count", "sum"),
                covered=("covered", "sum"),
                miss_count=("miss_count", "sum"),
                avg_interval_width=("avg_interval_width", "mean"),
                avg_abs_residual=("avg_abs_residual", "mean"),
            )
            .assign(empirical_coverage=lambda d: d["covered"] / d["count"].clip(lower=1))
            .assign(under_coverage_gap=lambda d: (target - d["empirical_coverage"]).clip(lower=0.0))
            .to_dict(orient="records")
        )
        fault_rows = (
            frame.groupby("fault_type", as_index=False)
            .agg(
                count=("count", "sum"),
                covered=("covered", "sum"),
                miss_count=("miss_count", "sum"),
                avg_interval_width=("avg_interval_width", "mean"),
                avg_abs_residual=("avg_abs_residual", "mean"),
            )
            .assign(empirical_coverage=lambda d: d["covered"] / d["count"].clip(lower=1))
            .assign(under_coverage_gap=lambda d: (target - d["empirical_coverage"]).clip(lower=0.0))
            .to_dict(orient="records")
        )

    _write_sorted_csv(rel_rows, pub / "reliability_group_coverage.csv", ["reliability_bin"])
    _write_sorted_csv(vol_rows, pub / "volatility_group_coverage.csv", ["volatility_bin"])
    _write_sorted_csv(fault_rows, pub / "fault_group_coverage.csv", ["fault_type"])
    _write_sorted_csv(validity_trace, pub / "shift_validity_trace.csv", ["t"])
    _write_sorted_csv(adaptive_trace, pub / "adaptive_quantile_trace.csv", ["t"])
