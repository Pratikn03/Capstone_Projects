#!/usr/bin/env python3
"""Compute reliability-binned coverage summaries for research analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from orius.forecasting.uncertainty.reliability_mondrian import (
    ReliabilityMondrian,
    ReliabilityMondrianConfig,
)


def build_summary(
    df: pd.DataFrame,
    *,
    y_true_col: str,
    y_pred_col: str,
    reliability_col: str,
    lower_col: str | None = None,
    upper_col: str | None = None,
    alpha: float = 0.10,
    n_bins: int = 10,
    min_bin_size: int = 25,
    binning: str = "quantile",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build reliability-binned coverage rows and a compact JSON summary."""
    cfg = ReliabilityMondrianConfig(alpha=alpha, n_bins=n_bins, min_bin_size=min_bin_size, binning=binning)
    model = ReliabilityMondrian(cfg)
    y_true = df[y_true_col].to_numpy(dtype=float)
    y_pred = df[y_pred_col].to_numpy(dtype=float)
    reliability = df[reliability_col].to_numpy(dtype=float)
    model.fit(y_true=y_true, y_pred=y_pred, reliability=reliability)

    if lower_col and upper_col and lower_col in df.columns and upper_col in df.columns:
        lower = df[lower_col].to_numpy(dtype=float)
        upper = df[upper_col].to_numpy(dtype=float)
    else:
        lower, upper = model.predict_interval(y_pred=y_pred, reliability=reliability)

    rows = pd.DataFrame(model.group_coverage(y_true=y_true, lower=lower, upper=upper, reliability=reliability))
    summary = {
        "alpha": float(alpha),
        "n_bins": int(model.n_bins_),
        "binning": binning,
        "n_samples": int(len(df)),
        "overall_picp": float(((y_true >= lower) & (y_true <= upper)).mean()),
        "overall_mean_interval_width": float((upper - lower).mean()),
        "worst_bin_picp": float(rows["picp"].dropna().min()) if not rows["picp"].dropna().empty else None,
        "best_bin_picp": float(rows["picp"].dropna().max()) if not rows["picp"].dropna().empty else None,
    }
    return rows, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute reliability-binned group coverage artifacts")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--y-true-col", default="y_true")
    parser.add_argument("--y-pred-col", default="y_pred")
    parser.add_argument("--reliability-col", default="reliability_w")
    parser.add_argument("--lower-col", default=None)
    parser.add_argument("--upper-col", default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--min-bin-size", type=int, default=25)
    parser.add_argument("--binning", choices=["quantile", "uniform"], default="quantile")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.input_csv)
    rows, summary = build_summary(
        df,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        reliability_col=args.reliability_col,
        lower_col=args.lower_col,
        upper_col=args.upper_col,
        alpha=float(args.alpha),
        n_bins=int(args.n_bins),
        min_bin_size=int(args.min_bin_size),
        binning=str(args.binning),
    )
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
