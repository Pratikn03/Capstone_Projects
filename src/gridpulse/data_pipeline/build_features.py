"""Data pipeline: build features."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gridpulse.data_pipeline.storage import write_sql

REQUIRED_COLS = [
    "utc_timestamp",
    "DE_load_actual_entsoe_transparency",
    "DE_wind_generation_actual",
    "DE_solar_generation_actual",
]
OPTIONAL_COLS = [
    "DE_price_day_ahead",
]


def add_price_carbon_features(
    # Key: normalize inputs and build time-aware features
    df: pd.DataFrame,
    price_col: str = "price_eur_mwh",
    base_price: float = 50.0,
) -> pd.DataFrame:
    """Create or sanitize price + carbon-intensity features.

    Price is time-of-day + seasonal proxy if not present. Carbon intensity
    varies with renewable share and peak periods to enable carbon-aware dispatch.
    """
    out = df.copy()

    if price_col not in out.columns:
        price = (
            base_price
            + 10.0 * out["is_morning_peak"]
            + 20.0 * out["is_evening_peak"]
            + 5.0 * (out["season"] == 3).astype(int)
            - 5.0 * out["is_weekend"]
        )
        out[price_col] = price.clip(lower=10.0)
    else:
        out[price_col] = pd.to_numeric(out[price_col], errors="coerce").interpolate(limit=6)

    load = out["load_mw"].clip(lower=1.0)
    ren = out["wind_mw"] + out["solar_mw"]
    ren_share = (ren / load).clip(0.0, 1.0)
    carbon = 450.0 - 200.0 * ren_share + 30.0 * out["is_evening_peak"] - 10.0 * out["is_weekend"]
    out["carbon_kg_per_mwh"] = carbon.clip(lower=50.0)
    return out

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    out = df.copy()
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    # simple season bucket (1=winter, 2=spring, 3=summer, 4=fall)
    out["season"] = ((out["month"] % 12 + 3) // 3).astype(int)
    return out

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # ramps
    for c in ["load_mw", "wind_mw", "solar_mw"]:
        out[f"{c}_delta_1h"] = out[c].diff(1)
        out[f"{c}_delta_24h"] = out[c].diff(24)
    # peak-ish flags (Germany typical evening peak)
    out["is_morning_peak"] = out["hour"].between(7, 10).astype(int)
    out["is_evening_peak"] = out["hour"].between(17, 21).astype(int)
    out["is_daylight"] = (out["solar_mw"] > 0).astype(int)
    return out

def add_lags_rolls(df: pd.DataFrame, cols: list[str], lags=(1, 24, 168), rolls=(24, 168)) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for l in lags:
            out[f"{c}_lag_{l}"] = out[c].shift(l)
        for w in rolls:
            out[f"{c}_roll_mean_{w}"] = out[c].shift(1).rolling(w).mean()
            out[f"{c}_roll_std_{w}"] = out[c].shift(1).rolling(w).std()
    return out

def load_weather(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        else:
            raise ValueError(f"Weather file missing timestamp column: {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True, help="Input directory (data/raw)")
    p.add_argument("--out", dest="out_dir", default="data/processed", help="Output directory")
    p.add_argument("--weather", default=None, help="Optional weather file (csv/parquet) to merge on timestamp")
    p.add_argument("--sql-out", default=None, help="Optional SQL DB path to write features table")
    p.add_argument("--sql-engine", choices=["duckdb", "sqlite"], default="duckdb")
    p.add_argument("--sql-table", default="features")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = in_dir / "time_series_60min_singleindex.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run download_opsd or place file manually.")

    df = pd.read_csv(csv_path, usecols=lambda c: c in REQUIRED_COLS + OPTIONAL_COLS)

    rename = {
        "utc_timestamp": "timestamp",
        "DE_load_actual_entsoe_transparency": "load_mw",
        "DE_wind_generation_actual": "wind_mw",
        "DE_solar_generation_actual": "solar_mw",
    }
    if "DE_price_day_ahead" in df.columns:
        rename["DE_price_day_ahead"] = "price_eur_mwh"
    df = df.rename(columns=rename)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = ["load_mw", "wind_mw", "solar_mw"]
    if "price_eur_mwh" in df.columns:
        numeric_cols.append("price_eur_mwh")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # continuous hourly index
    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz="UTC")
    df = df.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()

    # interpolate short gaps
    for col in numeric_cols:
        df[col] = df[col].interpolate(limit=6)

    df = add_time_features(df)
    df = add_domain_features(df)
    df = add_price_carbon_features(df, price_col="price_eur_mwh", base_price=50.0)

    if args.weather:
        wx = load_weather(Path(args.weather))
        # merge weather on timestamp, keep all energy rows
        df = df.merge(wx, on="timestamp", how="left")
        wx_cols = [c for c in df.columns if c.startswith("wx_")]
        for col in wx_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(limit=6)

    lag_cols = list(numeric_cols)
    if "price_eur_mwh" in df.columns and "price_eur_mwh" not in lag_cols:
        lag_cols.append("price_eur_mwh")
    if "carbon_kg_per_mwh" in df.columns:
        lag_cols.append("carbon_kg_per_mwh")
    df = add_lags_rolls(df, cols=lag_cols)

    # drop NaNs caused by lags/rolls
    df = df.dropna().reset_index(drop=True)

    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print("Rows:", len(df), "| Columns:", df.shape[1])

    if args.sql_out:
        sql_path = Path(args.sql_out)
        write_sql(df, sql_path, table=args.sql_table, engine=args.sql_engine)
        print(f"Saved SQL table '{args.sql_table}' to: {sql_path}")

if __name__ == "__main__":
    main()
