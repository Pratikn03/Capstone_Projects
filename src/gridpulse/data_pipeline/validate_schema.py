"""Data pipeline: validate schema."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLS = [
    "utc_timestamp",
    "DE_load_actual_entsoe_transparency",
    "DE_wind_generation_actual",
    "DE_solar_generation_actual",
]
OPTIONAL_COLS = [
    "DE_price_day_ahead",
]

def main():
    # Key: normalize inputs and build time-aware features
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True, help="Input directory (data/raw)")
    p.add_argument("--report", default="reports/data_quality_report.md", help="Markdown report output")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = in_dir / "time_series_60min_singleindex.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run download_opsd or place file manually.")

    df = pd.read_csv(csv_path, usecols=lambda c: c in REQUIRED_COLS + OPTIONAL_COLS)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    missing_optional = [c for c in OPTIONAL_COLS if c not in df.columns]

    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")
    df = df.sort_values("utc_timestamp").reset_index(drop=True)

    dup_ts = int(df["utc_timestamp"].duplicated().sum())

    full_idx = pd.date_range(df["utc_timestamp"].min(), df["utc_timestamp"].max(), freq="h", tz="UTC")
    df_idx = df.set_index("utc_timestamp")
    missing_ts = int(len(full_idx.difference(df_idx.index)))

    miss_frac = (df.isna().mean().sort_values(ascending=False)).to_string()

    # basic impossible values
    num_cols = [c for c in df.columns if c != "utc_timestamp"]
    negatives = {c: int((pd.to_numeric(df[c], errors="coerce") < 0).sum()) for c in num_cols}

    md = []
    md.append("# Data Quality Report\n\n")
    md.append(f"Input file: `{csv_path}`\n\n")
    md.append("## Coverage\n")
    md.append(f"- Start: **{df['utc_timestamp'].min()}**\n")
    md.append(f"- End: **{df['utc_timestamp'].max()}**\n")
    md.append(f"- Rows: **{len(df)}**\n")
    md.append(f"- Duplicate timestamps: **{dup_ts}**\n")
    md.append(f"- Missing hourly timestamps (vs full hourly index): **{missing_ts}**\n")
    if missing_optional:
        md.append(f"- Missing optional columns: **{missing_optional}**\n")
    md.append("\n")

    md.append("## Missingness (fraction)\n")
    md.append("```text\n")
    md.append(miss_frac)
    md.append("\n```\n\n")

    md.append("## Negative values check (count)\n")
    md.append("```text\n")
    for k, v in negatives.items():
        md.append(f"{k}: {v}\n")
    md.append("```\n")

    report_path.write_text("".join(md), encoding="utf-8")
    print(f"Wrote report: {report_path}")

if __name__ == "__main__":
    main()
