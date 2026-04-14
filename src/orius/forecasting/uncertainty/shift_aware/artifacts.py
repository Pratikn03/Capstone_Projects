from __future__ import annotations

from pathlib import Path
from typing import Any
import csv


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: tuple(str(r.get(c, "")) for c in columns))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({c: row.get(c) for c in columns})


def write_shift_aware_artifacts(
    *,
    reliability_rows: list[dict[str, Any]],
    volatility_rows: list[dict[str, Any]],
    fault_rows: list[dict[str, Any]],
    validity_trace_rows: list[dict[str, Any]],
    quantile_trace_rows: list[dict[str, Any]],
    out_dir: str | Path = "reports/publication",
) -> dict[str, str]:
    out = Path(out_dir)
    rel = out / "reliability_group_coverage.csv"
    vol = out / "volatility_group_coverage.csv"
    fault = out / "fault_group_coverage.csv"
    valid = out / "shift_validity_trace.csv"
    aci = out / "adaptive_quantile_trace.csv"

    group_cols = [
        "group_key",
        "count",
        "covered_count",
        "miss_count",
        "empirical_coverage",
        "target_coverage",
        "under_coverage_gap",
        "avg_interval_width",
        "avg_abs_residual",
    ]
    validity_cols = [
        "step",
        "validity_score",
        "validity_status",
        "normalized_residual",
        "under_coverage_gap",
        "drift_magnitude",
        "adaptation_instability",
        "reliability_score",
    ]
    aci_cols = ["step", "mode", "target_alpha", "effective_alpha", "effective_quantile", "instability"]

    _write_rows(rel, reliability_rows, group_cols)
    _write_rows(vol, volatility_rows, group_cols)
    _write_rows(fault, fault_rows, group_cols)
    _write_rows(valid, validity_trace_rows, validity_cols)
    _write_rows(aci, quantile_trace_rows, aci_cols)
    return {
        "reliability_group_coverage": str(rel),
        "volatility_group_coverage": str(vol),
        "fault_group_coverage": str(fault),
        "shift_validity_trace": str(valid),
        "adaptive_quantile_trace": str(aci),
    }
