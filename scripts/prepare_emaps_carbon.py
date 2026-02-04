"""Prepare Electricity Maps carbon intensity CSV into signals format."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pick_col(df: pd.DataFrame, preferred: str | None, candidates: list[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a column. Tried: {([preferred] if preferred else []) + candidates}")


def _convert_units(values: pd.Series, unit: str) -> pd.Series:
    unit = unit.lower().strip()
    # gCO2/kWh == kg/MWh numerically
    if unit in {"gco2_kwh", "gco2eq_kwh", "gco2/kwh", "gco2eq/kwh"}:
        return pd.to_numeric(values, errors="coerce")
    if unit in {"kgco2_mwh", "kgco2eq_mwh", "kg/mwh", "kgco2eq/mwh"}:
        return pd.to_numeric(values, errors="coerce")
    if unit in {"lbs_co2_per_mwh", "lb_co2_per_mwh", "lbs/mwh"}:
        return pd.to_numeric(values, errors="coerce") * 0.453592
    raise ValueError(f"Unsupported unit: {unit}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV/Parquet from Electricity Maps")
    ap.add_argument("--out", dest="out_path", default="data/raw/price_carbon_signals.csv")
    ap.add_argument("--timestamp-col", default=None, help="Timestamp column name (auto-detect if omitted)")
    ap.add_argument("--carbon-col", default=None, help="Carbon intensity column name (auto-detect if omitted)")
    ap.add_argument("--unit", default="gco2_kwh", help="Units for carbon column (gco2_kwh, kgco2_mwh, lbs_co2_per_mwh)")
    ap.add_argument("--zone-col", default=None, help="Optional zone column to filter")
    ap.add_argument("--zone", default=None, help="Zone to filter (e.g., DE-LU)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    if args.zone_col and args.zone:
        df = df[df[args.zone_col] == args.zone]

    ts_col = _pick_col(
        df,
        args.timestamp_col,
        ["datetime", "timestamp", "time", "date", "utc_datetime", "utc_time", "periodStart"],
    )
    carbon_col = _pick_col(
        df,
        args.carbon_col,
        ["carbonIntensity", "carbon_intensity", "carbonIntensityAvg", "carbon_intensity_avg"],
    )
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    carbon = _convert_units(df[carbon_col], args.unit)

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "carbon_kg_per_mwh": carbon,
        }
    ).dropna()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
