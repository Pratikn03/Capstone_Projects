"""Data pipeline: build model-ready features from raw OPSD energy data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from orius.data_pipeline.storage import write_sql

REQUIRED_NORMALIZED_COLS = ["timestamp", "load_mw", "wind_mw", "solar_mw"]

# Explicit OPSD lookup contract: direct columns are tried in order, and
# component sets are summed only when no direct total column is available.
OPSD_COLUMN_LOOKUP: dict[str, dict[str, dict[str, list[Any]]]] = {
    "default": {
        "load_mw": {
            "direct": [
                "{country}_load_actual_entsoe_transparency",
                "{country}_load_actual_entsoe_power_statistics",
                "{country}_load_actual_tso",
            ],
            "component_sets": [],
        },
        "wind_mw": {
            "direct": [
                "{country}_wind_generation_actual",
            ],
            "component_sets": [
                [
                    "{country}_wind_onshore_generation_actual",
                    "{country}_wind_offshore_generation_actual",
                ]
            ],
        },
        "solar_mw": {
            "direct": [
                "{country}_solar_generation_actual",
            ],
            "component_sets": [],
        },
        "price_eur_mwh": {
            "direct": [
                "{country}_LU_price_day_ahead",
                "{country}_price_day_ahead",
                "{country}_price_day_ahead_epex",
            ],
            "component_sets": [],
        },
    },
    "DE": {
        "price_eur_mwh": {
            "direct": [
                "DE_LU_price_day_ahead",
                "DE_price_day_ahead",
            ],
            "component_sets": [],
        },
    },
    "FR": {
        "price_eur_mwh": {
            "direct": [
                "FR_price_day_ahead",
                "FR_price_day_ahead_epex",
            ],
            "component_sets": [],
        },
    },
    "ES": {
        "price_eur_mwh": {
            "direct": [
                "ES_price_day_ahead",
            ],
            "component_sets": [],
        },
    },
}


def _country_lookup(country: str) -> dict[str, dict[str, list[Any]]]:
    code = country.upper()
    base = {
        key: {
            "direct": list(spec.get("direct", [])),
            "component_sets": [list(group) for group in spec.get("component_sets", [])],
        }
        for key, spec in OPSD_COLUMN_LOOKUP["default"].items()
    }
    override = OPSD_COLUMN_LOOKUP.get(code, {})
    for key, spec in override.items():
        base[key] = {
            "direct": list(spec.get("direct", [])),
            "component_sets": [list(group) for group in spec.get("component_sets", [])],
        }
    return base


def _format_candidate(candidate: str, *, country: str) -> str:
    return candidate.format(country=country.upper())


def _candidate_columns(country: str) -> tuple[list[str], list[str]]:
    lookup = _country_lookup(country)
    required = ["utc_timestamp"]
    optional: list[str] = []
    for normalized in ("load_mw", "wind_mw", "solar_mw"):
        spec = lookup[normalized]
        required.extend(_format_candidate(col, country=country) for col in spec["direct"])
        for group in spec["component_sets"]:
            required.extend(_format_candidate(col, country=country) for col in group)
    price_spec = lookup["price_eur_mwh"]
    optional.extend(_format_candidate(col, country=country) for col in price_spec["direct"])
    for group in price_spec["component_sets"]:
        optional.extend(_format_candidate(col, country=country) for col in group)
    return sorted(set(required)), sorted(set(optional))


def _pick_first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _sum_components(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = []
    for column in columns:
        if column in df.columns:
            values.append(pd.to_numeric(df[column], errors="coerce"))
    if not values:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return pd.concat(values, axis=1).sum(axis=1, min_count=1)


def normalize_opsd_country_frame(df: pd.DataFrame, *, country: str) -> pd.DataFrame:
    """Normalize raw OPSD rows for one country into the stable feature schema."""
    code = country.upper()
    if "utc_timestamp" not in df.columns:
        raise ValueError("Raw OPSD frame must include 'utc_timestamp'")

    lookup = _country_lookup(code)
    out = pd.DataFrame({"timestamp": pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")})

    for normalized in ("load_mw", "wind_mw", "solar_mw", "price_eur_mwh"):
        spec = lookup[normalized]
        direct_cols = [_format_candidate(col, country=code) for col in spec["direct"]]
        direct = _pick_first_present(df, direct_cols)
        if direct is not None:
            out[normalized] = pd.to_numeric(df[direct], errors="coerce")
            continue

        component_sets = [[_format_candidate(col, country=code) for col in group] for group in spec["component_sets"]]
        resolved = pd.Series(float("nan"), index=df.index, dtype=float)
        for group in component_sets:
            series = _sum_components(df, group)
            if not series.isna().all():
                resolved = series
                break
        if normalized != "price_eur_mwh" and resolved.isna().all():
            raise ValueError(f"Missing required OPSD columns for {code} {normalized}")
        if not resolved.isna().all():
            out[normalized] = resolved

    missing = [col for col in REQUIRED_NORMALIZED_COLS if col not in out.columns]
    if missing:
        raise ValueError(f"Normalized OPSD frame missing required columns: {missing}")
    return out


def add_price_carbon_features(
    df: pd.DataFrame,
    price_col: str = "price_eur_mwh",
    base_price: float = 50.0,
) -> pd.DataFrame:
    """Create or sanitize price + carbon-intensity features."""
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
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["season"] = ((out["month"] % 12 + 3) // 3).astype(int)
    return out


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add energy-domain features like ramps and peak flags."""
    out = df.copy()
    for column in ["load_mw", "wind_mw", "solar_mw"]:
        out[f"{column}_delta_1h"] = out[column].diff(1)
        out[f"{column}_delta_24h"] = out[column].diff(24)
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
        hdf = pd.read_csv(holiday_file)
        col = "date"
        if col not in hdf.columns:
            if "timestamp" in hdf.columns:
                col = "timestamp"
            elif "time" in hdf.columns:
                col = "time"
            else:
                col = None
        if col is not None:
            holiday_dates = set(pd.to_datetime(hdf[col], errors="coerce").dt.date.dropna().unique())
    else:
        try:
            import holidays  # type: ignore

            years = range(out["timestamp"].dt.year.min(), out["timestamp"].dt.year.max() + 1)
            holiday_dates = set(holidays.country_holidays(country, years=years).keys())
        except Exception:
            holiday_dates = set()

    out["is_holiday"] = dates.isin(holiday_dates).astype(int) if holiday_dates else 0
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
    for column in cols:
        for lag in lags:
            out[f"{column}_lag_{lag}"] = out[column].shift(lag)
        for window in rolls:
            out[f"{column}_roll_mean_{window}"] = out[column].shift(1).rolling(window).mean()
            out[f"{column}_roll_std_{window}"] = out[column].shift(1).rolling(window).std()
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
        df["carbon_kg_per_mwh"] = pd.to_numeric(df["carbon_gco2_kwh"], errors="coerce")
    if "moer_lbs_per_mwh" in df.columns and "moer_kg_per_mwh" not in df.columns:
        df["moer_kg_per_mwh"] = pd.to_numeric(df["moer_lbs_per_mwh"], errors="coerce") * 0.453592

    keep_cols = [
        column
        for column in df.columns
        if column in {"timestamp", "price_eur_mwh", "price_usd_mwh", "carbon_kg_per_mwh", "moer_kg_per_mwh"}
    ]
    return df[keep_cols]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True, help="Input directory (data/raw)")
    parser.add_argument("--out", dest="out_dir", default="data/processed", help="Output directory")
    parser.add_argument("--country", default="DE", help="OPSD country code to extract (default: DE)")
    parser.add_argument("--weather", default=None, help="Optional weather file (csv/parquet) to merge on timestamp")
    parser.add_argument("--signals", default=None, help="Optional price/carbon signals file (csv/parquet) to merge on timestamp")
    parser.add_argument("--holidays", default=None, help="Optional holidays file (csv) with date column")
    parser.add_argument("--holiday-country", default=None, help="Holiday country code (default: same as --country)")
    parser.add_argument("--sql-out", default=None, help="Optional SQL DB path to write features table")
    parser.add_argument("--sql-engine", choices=["duckdb", "sqlite"], default="duckdb")
    parser.add_argument("--sql-table", default="features")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint to build model features from raw OPSD inputs."""
    args = _parse_args()
    country = str(args.country or "DE").upper()
    holiday_country = str(args.holiday_country or country).upper()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = in_dir / "time_series_60min_singleindex.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run download_opsd or place file manually.")

    required_cols, optional_cols = _candidate_columns(country)
    wanted = set(required_cols + optional_cols)
    df_raw = pd.read_csv(csv_path, usecols=lambda c: c in wanted)
    df = normalize_opsd_country_frame(df_raw, country=country)

    df = df.sort_values("timestamp").reset_index(drop=True)
    numeric_cols = ["load_mw", "wind_mw", "solar_mw"]
    if "price_eur_mwh" in df.columns:
        numeric_cols.append("price_eur_mwh")
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz="UTC")
    df = df.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()
    for column in numeric_cols:
        df[column] = df[column].interpolate(limit=6)

    if args.signals:
        signals = load_signals(Path(args.signals))
        df = df.merge(signals, on="timestamp", how="left", suffixes=("", "_sig"))
        for column in ("price_eur_mwh", "price_usd_mwh", "carbon_kg_per_mwh"):
            sig_col = f"{column}_sig"
            if sig_col in df.columns:
                if column in df.columns:
                    df[column] = df[sig_col].combine_first(df[column])
                else:
                    df[column] = df[sig_col]
                df.drop(columns=[sig_col], inplace=True)

    for column in ("price_eur_mwh", "price_usd_mwh", "carbon_kg_per_mwh", "moer_kg_per_mwh"):
        if column in df.columns and column not in numeric_cols:
            numeric_cols.append(column)
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = add_time_features(df)
    df = add_domain_features(df)
    df = add_holiday_features(
        df,
        country=holiday_country,
        holiday_file=Path(args.holidays) if args.holidays else None,
    )
    price_col = "price_eur_mwh"
    if price_col not in df.columns and "price_usd_mwh" in df.columns:
        price_col = "price_usd_mwh"
    df = add_price_carbon_features(df, price_col=price_col, base_price=50.0)

    if args.weather:
        weather = load_weather(Path(args.weather))
        df = df.merge(weather, on="timestamp", how="left")
        weather_cols = [column for column in df.columns if column.startswith("wx_")]
        for column in weather_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce").interpolate(limit=6)

    lag_cols = list(numeric_cols)
    for price_column in ("price_eur_mwh", "price_usd_mwh"):
        if price_column in df.columns and price_column not in lag_cols:
            lag_cols.append(price_column)
    if "carbon_kg_per_mwh" in df.columns and "carbon_kg_per_mwh" not in lag_cols:
        lag_cols.append("carbon_kg_per_mwh")
    for column in [column for column in df.columns if column.startswith("wx_")]:
        if column not in lag_cols:
            lag_cols.append(column)
    df = add_lags_rolls(df, cols=lag_cols)

    df = df.dropna().reset_index(drop=True)

    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print("Country:", country, "| Rows:", len(df), "| Columns:", df.shape[1])

    if args.sql_out:
        sql_path = Path(args.sql_out)
        write_sql(df, sql_path, table=args.sql_table, engine=args.sql_engine)
        print(f"Saved SQL table '{args.sql_table}' to: {sql_path}")


if __name__ == "__main__":
    main()
