"""Schema validation and data-quality checks for OPSD raw data and feature parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.data_pipeline.build_features import _candidate_columns, normalize_opsd_country_frame


def _write_report(
    *,
    report_path: Path,
    title: str,
    input_label: str,
    start_ts: object,
    end_ts: object,
    rows: int,
    duplicate_timestamps: int,
    missing_timestamp_count: int | None,
    missingness_text: str,
    negatives: dict[str, int],
    extra_lines: list[str] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    md.append(f"# {title}\n\n")
    md.append(f"Input file: `{input_label}`\n\n")
    md.append("## Coverage\n")
    md.append(f"- Start: **{start_ts}**\n")
    md.append(f"- End: **{end_ts}**\n")
    md.append(f"- Rows: **{rows}**\n")
    md.append(f"- Duplicate timestamps: **{duplicate_timestamps}**\n")
    if missing_timestamp_count is not None:
        md.append(f"- Missing hourly timestamps (vs full hourly index): **{missing_timestamp_count}**\n")
    for line in extra_lines or []:
        md.append(f"- {line}\n")
    md.append("\n")
    md.append("## Missingness (fraction)\n")
    md.append("```text\n")
    md.append(missingness_text)
    md.append("\n```\n\n")
    md.append("## Negative values check (count)\n")
    md.append("```text\n")
    for key, value in negatives.items():
        md.append(f"{key}: {value}\n")
    md.append("```\n")
    report_path.write_text("".join(md), encoding="utf-8")
    print(f"Wrote report: {report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input directory (data/raw) or features parquet file")
    parser.add_argument("--report", default="reports/data_quality_report.md", help="Markdown report output")
    parser.add_argument("--country", default="DE", help="OPSD country code for raw validation (default: DE)")
    parser.add_argument("--required-cols", default="", help="Comma-separated required columns (default: load_mw,wind_mw,solar_mw for energy)")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint to validate raw OPSD inputs or processed feature parquet schema."""
    args = _parse_args()
    in_path = Path(args.in_path)
    report_path = Path(args.report)
    country = str(args.country or "DE").upper()

    if in_path.is_file() and in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
        required_cols_arg = str(args.required_cols or "").strip()
        if required_cols_arg:
            required_feature_cols = ["timestamp"] + [c.strip() for c in required_cols_arg.split(",") if c.strip()]
        else:
            required_feature_cols = ["timestamp", "load_mw", "wind_mw", "solar_mw"]
        missing_cols = [column for column in required_feature_cols if column not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("Feature schema invalid: timestamp contains null/invalid values.")
        df = df.sort_values("timestamp").reset_index(drop=True)
        dup_ts = int(df["timestamp"].duplicated().sum())
        if dup_ts > 0:
            raise ValueError(f"Feature schema invalid: duplicate timestamps detected ({dup_ts}).")

        miss_frac = df.isna().mean().sort_values(ascending=False).to_string()
        numeric_cols = [column for column in df.columns if column != "timestamp"]
        negatives = {column: int((pd.to_numeric(df[column], errors="coerce") < 0).sum()) for column in numeric_cols}
        _write_report(
            report_path=report_path,
            title="Feature Schema Validation Report",
            input_label=str(in_path),
            start_ts=df["timestamp"].min(),
            end_ts=df["timestamp"].max(),
            rows=len(df),
            duplicate_timestamps=dup_ts,
            missing_timestamp_count=None,
            missingness_text=miss_frac,
            negatives=negatives,
        )
        return

    if not in_path.exists() or not in_path.is_dir():
        raise FileNotFoundError(f"Expected input directory or parquet file, got: {in_path}")

    csv_path = in_path / "time_series_60min_singleindex.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run download_opsd or place file manually.")

    required_cols, optional_cols = _candidate_columns(country)
    wanted = set(required_cols + optional_cols)
    raw = pd.read_csv(csv_path, usecols=lambda c: c in wanted)
    normalized = normalize_opsd_country_frame(raw, country=country)
    numeric_cols = [column for column in normalized.columns if column != "timestamp"]
    dup_ts = int(normalized["timestamp"].duplicated().sum())
    full_idx = pd.date_range(normalized["timestamp"].min(), normalized["timestamp"].max(), freq="h", tz="UTC")
    df_idx = normalized.set_index("timestamp")
    missing_ts = int(len(full_idx.difference(df_idx.index)))
    miss_frac = normalized.isna().mean().sort_values(ascending=False).to_string()
    negatives = {column: int((pd.to_numeric(normalized[column], errors="coerce") < 0).sum()) for column in numeric_cols}
    missing_optional = [column for column in optional_cols if column not in raw.columns]

    _write_report(
        report_path=report_path,
        title="Data Quality Report",
        input_label=str(csv_path),
        start_ts=normalized["timestamp"].min(),
        end_ts=normalized["timestamp"].max(),
        rows=len(normalized),
        duplicate_timestamps=dup_ts,
        missing_timestamp_count=missing_ts,
        missingness_text=miss_frac,
        negatives=negatives,
        extra_lines=[
            f"Country: **{country}**",
            f"Missing optional columns: **{missing_optional}**" if missing_optional else "Missing optional columns: **[]**",
        ],
    )


if __name__ == "__main__":
    main()
