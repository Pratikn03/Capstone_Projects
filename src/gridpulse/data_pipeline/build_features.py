"""Data pipeline: build model-ready features from raw energy data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gridpulse.data_pipeline.storage import write_sql

# Minimal OPSD columns required for core targets.
REQUIRED_COLS = [
    "utc_timestamp",
    "DE_load_actual_entsoe_transparency",
    "DE_wind_generation_actual",
    "DE_solar_generation_actual",
]
# Optional columns that improve optimization realism.
OPTIONAL_COLS = [
    "DE_price_day_ahead",
    "DE_LU_price_day_ahead",
]


def add_price_carbon_features(
    df: pd.DataFrame,
    price_col: str = "price_eur_mwh",
    base_price: float = 50.0,
) -> pd.DataFrame:
    """Create or sanitize price + carbon-intensity features.

    Price is time-of-day + seasonal proxy if not present. Carbon intensity
    varies with renewable share and peak periods to enable carbon-aware dispatch.
    """
    # Work on a copy to avoid mutating caller data.
    out = df.copy()

    if price_col not in out.columns:
        # Proxy price when market prices are missing (time-of-day + seasonality).
        price = (
            base_price
            + 10.0 * out["is_morning_peak"]
            + 20.0 * out["is_evening_peak"]
            + 5.0 * (out["season"] == 3).astype(int)
            - 5.0 * out["is_weekend"]
        )
        out[price_col] = price.clip(lower=10.0)
    else:
        # Clean and interpolate provided prices.
        out[price_col] = pd.to_numeric(out[price_col], errors="coerce").interpolate(limit=6)

    # Carbon intensity proxy uses renewable share + peak heuristics.
    load = out["load_mw"].clip(lower=1.0)
    ren = out["wind_mw"] + out["solar_mw"]
    ren_share = (ren / load).clip(0.0, 1.0)
    carbon = 450.0 - 200.0 * ren_share + 30.0 * out["is_evening_peak"] - 10.0 * out["is_weekend"]
    if "carbon_kg_per_mwh" not in out.columns:
        out["carbon_kg_per_mwh"] = carbon.clip(lower=50.0)
    else:
        out["carbon_kg_per_mwh"] = (
            pd.to_numeric(out["carbon_kg_per_mwh"], errors="coerce")
            .interpolate(limit=6)
            .fillna(carbon)
            .clip(lower=50.0)
        )
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features derived from timestamps."""
    ts = df["timestamp"]
    out = df.copy()
    # Calendar features capture daily/weekly/seasonal patterns.
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    # simple season bucket (1=winter, 2=spring, 3=summer, 4=fall)
    out["season"] = ((out["month"] % 12 + 3) // 3).astype(int)
    return out


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add energy-domain features like ramps and peak flags."""
    out = df.copy()
    # ramps
    for c in ["load_mw", "wind_mw", "solar_mw"]:
        # Short/long ramps highlight operational changes.
        out[f"{c}_delta_1h"] = out[c].diff(1)
        out[f"{c}_delta_24h"] = out[c].diff(24)
    # peak-ish flags (Germany typical evening peak)
    out["is_morning_peak"] = out["hour"].between(7, 10).astype(int)
    out["is_evening_peak"] = out["hour"].between(17, 21).astype(int)
    out["is_daylight"] = (out["solar_mw"] > 0).astype(int)
    return out


def add_holiday_features(
    df: pd.DataFrame,
    country: str = "DE",
    holiday_file: Path | None = None,
) -> pd.DataFrame:
    """Add holiday, pre-holiday, and post-holiday flags."""
    out = df.copy()
    dates = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.date
    holiday_dates: set = set()

    if holiday_file and holiday_file.exists():
        # Prefer a user-provided holiday calendar for deterministic runs.
        hdf = pd.read_csv(holiday_file)
        col = "date"
        if col not in hdf.columns:
            # fallback to timestamp column
            if "timestamp" in hdf.columns:
                col = "timestamp"
            elif "time" in hdf.columns:
                col = "time"
            else:
                col = None
        if col is not None:
            holiday_dates = set(pd.to_datetime(hdf[col], errors="coerce").dt.date.dropna().unique())
    else:
        # Fallback to python-holidays if available.
        try:
            import holidays  # type: ignore

            years = range(out["timestamp"].dt.year.min(), out["timestamp"].dt.year.max() + 1)
            holiday_dates = set(holidays.country_holidays(country, years=years).keys())
        except Exception:
            holiday_dates = set()

    out["is_holiday"] = dates.isin(holiday_dates).astype(int) if holiday_dates else 0
    # Adjacent days are often behaviorally similar.
    if holiday_dates:
        dates_ts = pd.to_datetime(dates)
        next_dates = (dates_ts + pd.Timedelta(days=1)).dt.date
        prev_dates = (dates_ts - pd.Timedelta(days=1)).dt.date
        out["is_pre_holiday"] = next_dates.isin(holiday_dates).astype(int)
        out["is_post_holiday"] = prev_dates.isin(holiday_dates).astype(int)
    else:
        out["is_pre_holiday"] = 0
        out["is_post_holiday"] = 0
    return out


def add_lags_rolls(df: pd.DataFrame, cols: list[str], lags=(1, 24, 168), rolls=(24, 168)) -> pd.DataFrame:
    """Add lagged values and rolling summary stats for each column."""
    out = df.copy()
    for c in cols:
        for l in lags:
            # Lag features preserve autocorrelation.
            out[f"{c}_lag_{l}"] = out[c].shift(l)
        for w in rolls:
            # Rolling stats summarize trend and volatility.
            out[f"{c}_roll_mean_{w}"] = out[c].shift(1).rolling(w).mean()
            out[f"{c}_roll_std_{w}"] = out[c].shift(1).rolling(w).std()
    return out


def load_weather(path: Path) -> pd.DataFrame:
    """Load a weather file (CSV/Parquet) with a timestamp column."""
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


def load_signals(path: Path) -> pd.DataFrame:
    """Load price/carbon signals, normalize columns, and keep known fields."""
    if not path.exists():
        raise FileNotFoundError(f"Signals file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        else:
            raise ValueError(f"Signals file missing timestamp column: {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    rename = {}
    if "price" in df.columns and "price_eur_mwh" not in df.columns:
        rename["price"] = "price_eur_mwh"
    if "carbon_intensity" in df.columns and "carbon_kg_per_mwh" not in df.columns:
        rename["carbon_intensity"] = "carbon_kg_per_mwh"
    if "co2_moer" in df.columns and "moer_kg_per_mwh" not in df.columns:
        rename["co2_moer"] = "moer_kg_per_mwh"
    if rename:
        df = df.rename(columns=rename)
    if "carbon_gco2_kwh" in df.columns and "carbon_kg_per_mwh" not in df.columns:
        # gCO2/kWh is numerically equivalent to kg/MWh.
        df["carbon_kg_per_mwh"] = pd.to_numeric(df["carbon_gco2_kwh"], errors="coerce")
    if "moer_lbs_per_mwh" in df.columns and "moer_kg_per_mwh" not in df.columns:
        df["moer_kg_per_mwh"] = pd.to_numeric(df["moer_lbs_per_mwh"], errors="coerce") * 0.453592

    keep_cols = [
        c
        for c in df.columns
        if c
        in {
            "timestamp",
            "price_eur_mwh",
            "price_usd_mwh",
            "carbon_kg_per_mwh",
            "moer_kg_per_mwh",
        }
    ]
    df = df[keep_cols]
    return df


def main():
    """CLI entrypoint to build model features from raw inputs."""
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True, help="Input directory (data/raw)")
    p.add_argument("--out", dest="out_dir", default="data/processed", help="Output directory")
    p.add_argument("--weather", default=None, help="Optional weather file (csv/parquet) to merge on timestamp")
    p.add_argument("--signals", default=None, help="Optional price/carbon signals file (csv/parquet) to merge on timestamp")
    p.add_argument("--holidays", default=None, help="Optional holidays file (csv) with date column")
    p.add_argument("--holiday-country", default="DE", help="Holiday country code (default: DE)")
    p.add_argument("--sql-out", default=None, help="Optional SQL DB path to write features table")
    p.add_argument("--sql-engine", choices=["duckdb", "sqlite"], default="duckdb")
    p.add_argument("--sql-table", default="features")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load raw OPSD time series.
    csv_path = in_dir / "time_series_60min_singleindex.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run download_opsd or place file manually.")

    df = pd.read_csv(csv_path, usecols=lambda c: c in REQUIRED_COLS + OPTIONAL_COLS)

    # 2) Normalize column names to model-friendly labels.
    rename = {
        "utc_timestamp": "timestamp",
        "DE_load_actual_entsoe_transparency": "load_mw",
        "DE_wind_generation_actual": "wind_mw",
        "DE_solar_generation_actual": "solar_mw",
    }
    # Prefer DE_LU day-ahead price when available (common in OPSD exports).
    if "DE_LU_price_day_ahead" in df.columns:
        rename["DE_LU_price_day_ahead"] = "price_eur_mwh"
    elif "DE_price_day_ahead" in df.columns:
        rename["DE_price_day_ahead"] = "price_eur_mwh"
    df = df.rename(columns=rename)

    # 3) Parse timestamps and ensure stable ordering.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 4) Coerce numeric columns and keep price if present.
    numeric_cols = ["load_mw", "wind_mw", "solar_mw"]
    if "price_eur_mwh" in df.columns:
        numeric_cols.append("price_eur_mwh")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Enforce continuous hourly index for stable lag/roll features.
    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz="UTC")
    df = df.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()

    # 6) Interpolate short gaps to avoid feature discontinuities.
    for col in numeric_cols:
        df[col] = df[col].interpolate(limit=6)

    # 7) Merge optional signals (price/carbon).
    if args.signals:
        signals = load_signals(Path(args.signals))
        df = df.merge(signals, on="timestamp", how="left", suffixes=("", "_sig"))
        for col in ("price_eur_mwh", "price_usd_mwh", "carbon_kg_per_mwh"):
            sig_col = f"{col}_sig"
            if sig_col in df.columns:
                if col in df.columns:
                    df[col] = df[sig_col].combine_first(df[col])
                else:
                    df[col] = df[sig_col]
                df.drop(columns=[sig_col], inplace=True)

    # 8) Ensure any merged signals are treated as numeric.
    for col in ("price_eur_mwh", "price_usd_mwh", "carbon_kg_per_mwh", "moer_kg_per_mwh"):
        if col in df.columns and col not in numeric_cols:
            numeric_cols.append(col)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 9) Base time and domain features, plus optional holiday flags.
    df = add_time_features(df)
    df = add_domain_features(df)
    df = add_holiday_features(df, country=args.holiday_country, holiday_file=Path(args.holidays) if args.holidays else None)
    price_col = "price_eur_mwh"
    if price_col not in df.columns and "price_usd_mwh" in df.columns:
        price_col = "price_usd_mwh"
    df = add_price_carbon_features(df, price_col=price_col, base_price=50.0)

    # 10) Merge optional weather signals.
    if args.weather:
        wx = load_weather(Path(args.weather))
        # merge weather on timestamp, keep all energy rows
        df = df.merge(wx, on="timestamp", how="left")
        wx_cols = [c for c in df.columns if c.startswith("wx_")]
        for col in wx_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(limit=6)

    # 11) Build lag and rolling statistics for numeric inputs.
    lag_cols = list(numeric_cols)
    for price_col in ("price_eur_mwh", "price_usd_mwh"):
        if price_col in df.columns and price_col not in lag_cols:
            lag_cols.append(price_col)
    if "carbon_kg_per_mwh" in df.columns:
        lag_cols.append("carbon_kg_per_mwh")
    # Add weather lags if present.
    wx_cols = [c for c in df.columns if c.startswith("wx_")]
    for col in wx_cols:
        if col not in lag_cols:
            lag_cols.append(col)
    df = add_lags_rolls(df, cols=lag_cols)

    # 12) Drop rows where lag/roll windows are incomplete.
    df = df.dropna().reset_index(drop=True)

    # 13) Persist features to disk (Parquet is fast + compact).
    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print("Rows:", len(df), "| Columns:", df.shape[1])

    # 14) Optional SQL export for downstream tools.
    if args.sql_out:
        sql_path = Path(args.sql_out)
        write_sql(df, sql_path, table=args.sql_table, engine=args.sql_engine)
        print(f"Saved SQL table '{args.sql_table}' to: {sql_path}")

if __name__ == "__main__":
    main()
