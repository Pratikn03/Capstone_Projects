from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ComparisonSummary:
    legacy_rows: int
    shift_rows: int
    legacy_coverage: float
    shift_coverage: float
    legacy_mean_width: float
    shift_mean_width: float
    coverage_delta: float
    width_delta: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "legacy_rows": self.legacy_rows,
            "shift_rows": self.shift_rows,
            "legacy_coverage": self.legacy_coverage,
            "shift_coverage": self.shift_coverage,
            "legacy_mean_width": self.legacy_mean_width,
            "shift_mean_width": self.shift_mean_width,
            "coverage_delta": self.coverage_delta,
            "width_delta": self.width_delta,
        }


def _coverage(df: pd.DataFrame) -> float:
    return float(((df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])).mean()) if len(df) else 0.0


def summarize_legacy_vs_shift(legacy_df: pd.DataFrame, shift_df: pd.DataFrame) -> ComparisonSummary:
    lc = _coverage(legacy_df)
    sc = _coverage(shift_df)
    lw = float((legacy_df["upper"] - legacy_df["lower"]).mean()) if len(legacy_df) else 0.0
    sw = float((shift_df["upper"] - shift_df["lower"]).mean()) if len(shift_df) else 0.0
    return ComparisonSummary(
        legacy_rows=int(len(legacy_df)),
        shift_rows=int(len(shift_df)),
        legacy_coverage=lc,
        shift_coverage=sc,
        legacy_mean_width=lw,
        shift_mean_width=sw,
        coverage_delta=float(sc - lc),
        width_delta=float(sw - lw),
    )


def write_comparison_package(
    *,
    legacy_csv: str,
    shift_csv: str,
    out_dir: str,
    target_coverage: float = 0.9,
    max_width_increase: float = 5.0,
) -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    legacy_df = pd.read_csv(legacy_csv)
    shift_df = pd.read_csv(shift_csv)
    summary = summarize_legacy_vs_shift(legacy_df, shift_df)

    table = pd.DataFrame([summary.to_dict()])
    table.to_csv(out / "legacy_vs_shift_summary.csv", index=False)

    signoff = {
        "target_coverage": float(target_coverage),
        "max_width_increase": float(max_width_increase),
        "coverage_ok": bool(summary.shift_coverage >= target_coverage),
        "width_ok": bool(summary.width_delta <= max_width_increase),
        "summary": summary.to_dict(),
    }
    signoff["all_checks_pass"] = bool(signoff["coverage_ok"] and signoff["width_ok"])
    (out / "acceptance_signoff.json").write_text(json.dumps(signoff, indent=2), encoding="utf-8")
    return signoff
