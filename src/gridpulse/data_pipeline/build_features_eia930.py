from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import zipfile

import pandas as pd

from gridpulse.data_pipeline.build_features import add_time_features, add_domain_features, add_lags_rolls


def _iter_balance_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("eia930-*-balance.csv"))


def _iter_zip_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("eia930-*.zip"))


def _read_balance_csv(path: Path, usecols: list[str], chunksize: int = 200_000):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            # find balance file in zip
            balance_name = next((n for n in zf.namelist() if n.endswith("-balance.csv")), None)
            if not balance_name:
                return
            with zf.open(balance_name) as f:
                for chunk in pd.read_csv(f, usecols=lambda c: c in usecols, chunksize=chunksize):
                    yield chunk
    else:
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize):
            yield chunk


def _choose_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # choose demand column (Adjusted preferred)
    demand_col = _choose_col(df, ["Demand (MW) (Adjusted)", "Demand (MW)"])
    if demand_col is None:
        raise ValueError("Demand column not found")

    wind_wo = _choose_col(df, [
        "Net Generation (MW) from Wind without Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Wind without Integrated Battery Storage",
    ])
    wind_w = _choose_col(df, [
        "Net Generation (MW) from Wind with Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Wind with Integrated Battery Storage",
    ])
    solar_wo = _choose_col(df, [
        "Net Generation (MW) from Solar without Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Solar without Integrated Battery Storage",
    ])
    solar_w = _choose_col(df, [
        "Net Generation (MW) from Solar with Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Solar with Integrated Battery Storage",
    ])

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["UTC Time at End of Hour"], utc=True, errors="coerce"),
        "load_mw": pd.to_numeric(df[demand_col], errors="coerce"),
    })

    wind = 0.0
    if wind_wo:
        wind = wind + pd.to_numeric(df[wind_wo], errors="coerce").fillna(0.0)
    if wind_w:
        wind = wind + pd.to_numeric(df[wind_w], errors="coerce").fillna(0.0)
    solar = 0.0
    if solar_wo:
        solar = solar + pd.to_numeric(df[solar_wo], errors="coerce").fillna(0.0)
    if solar_w:
        solar = solar + pd.to_numeric(df[solar_w], errors="coerce").fillna(0.0)

    out["wind_mw"] = wind
    out["solar_mw"] = solar
    return out


def build_eia930_features(raw_dir: Path, out_dir: Path, ba: str, start: str | None, end: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "Balancing Authority",
        "UTC Time at End of Hour",
        "Demand (MW)",
        "Demand (MW) (Adjusted)",
        "Net Generation (MW) from Wind without Integrated Battery Storage",
        "Net Generation (MW) from Wind with Integrated Battery Storage",
        "Net Generation (MW) from Solar without Integrated Battery Storage",
        "Net Generation (MW) from Solar with Integrated Battery Storage",
        "Net Generation (MW) from Wind without Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Wind with Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Solar without Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Solar with Integrated Battery Storage (Adjusted)",
    ]

    frames = []
    # prefer zip files if present
    files = _iter_zip_files(raw_dir)
    if not files:
        files = _iter_balance_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No EIA930 files found in {raw_dir}")

    for f in files:
        for chunk in _read_balance_csv(f, usecols=usecols):
            if ba:
                chunk = chunk[chunk["Balancing Authority"] == ba]
            if chunk.empty:
                continue
            frames.append(_normalize(chunk))

    if not frames:
        raise ValueError(f"No records found for BA '{ba}'. Check the code or spelling.")

    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end, utc=True)]

    # continuous hourly index and interpolate short gaps
    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz="UTC")
    df = df.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()

    for col in ["load_mw", "wind_mw", "solar_mw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(limit=6)

    df = add_time_features(df)
    df = add_domain_features(df)
    df = add_lags_rolls(df, cols=["load_mw", "wind_mw", "solar_mw"])
    df = df.dropna().reset_index(drop=True)

    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Rows:", len(df), "| Columns:", df.shape[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True, help="Input directory (data/raw/us_eia930)")
    p.add_argument("--out", dest="out_dir", default="data/processed/us_eia930", help="Output directory")
    p.add_argument("--ba", default="MISO", help="Balancing Authority code (e.g., MISO, PJM, CAISO)")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = p.parse_args()

    build_eia930_features(Path(args.in_dir), Path(args.out_dir), args.ba, args.start, args.end)


if __name__ == "__main__":
    main()
