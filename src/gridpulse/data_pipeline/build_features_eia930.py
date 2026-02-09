"""Data pipeline: build features for EIA‑930 (US balancing authorities).

Handles three CSV schema eras:
  - 2015H2–2018H1: 19 columns, no fuel‑type breakdown (wind/solar unavailable)
  - 2018H2–2024H1: aggregate "from Solar" / "from Wind" columns
  - 2024H2+:       battery‑storage split columns
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import zipfile

import pandas as pd

from gridpulse.data_pipeline.build_features import (
    add_time_features,
    add_domain_features,
    add_holiday_features,
    add_lags_rolls,
    add_price_carbon_features,
)

# ---------------------------------------------------------------------------
# All column names we might encounter across schema versions.  The
# ``usecols=lambda c: c in _ALL_USECOLS`` pattern lets pandas skip
# columns that don't exist in a particular file.
# ---------------------------------------------------------------------------
_ALL_USECOLS: set[str] = {
    # common
    "Balancing Authority",
    "UTC Time at End of Hour",
    "Demand (MW)",
    "Demand (MW) (Adjusted)",
    "Net Generation (MW)",
    "Net Generation (MW) (Adjusted)",
    # --- aggregate (2018H2–2024H1) ---
    "Net Generation (MW) from Solar",
    "Net Generation (MW) from Wind",
    "Net Generation (MW) from Solar (Adjusted)",
    "Net Generation (MW) from Wind (Adjusted)",
    # --- battery‑storage split (2024H2+) ---
    "Net Generation (MW) from Solar without Integrated Battery Storage",
    "Net Generation (MW) from Solar with Integrated Battery Storage",
    "Net Generation (MW) from Wind without Integrated Battery Storage",
    "Net Generation (MW) from Wind with Integrated Battery Storage",
    "Net Generation (MW) from Solar without Integrated Battery Storage (Adjusted)",
    "Net Generation (MW) from Solar witho Integrated Battery Storage (Adjusted)",  # sic – EIA typo
    "Net Generation (MW) from Solar with Integrated Battery Storage (Adjusted)",
    "Net Generation (MW) from Wind without Integrated Battery Storage (Adjusted)",
    "Net Generation (MW) from Wind with Integrated Battery Storage (Adjusted)",
    # --- fuel mix (for carbon accuracy) ---
    "Net Generation (MW) from Coal",
    "Net Generation (MW) from Natural Gas",
    "Net Generation (MW) from Nuclear",
    "Net Generation (MW) from Hydropower Excluding Pumped Storage",
    "Net Generation (MW) from Coal (Adjusted)",
    "Net Generation (MW) from Natural Gas (Adjusted)",
    "Net Generation (MW) from Nuclear (Adjusted)",
    "Net Generation (MW) from Hydropower Excluding Pumped Storage (Adjusted)",
}


def _iter_balance_files(raw_dir: Path) -> list[Path]:
    """Return unpacked balance CSVs if present."""
    return sorted(raw_dir.glob("eia930-*-balance.csv"))


def _iter_zip_files(raw_dir: Path) -> list[Path]:
    """Return ZIP archives when raw data is downloaded in bulk."""
    return sorted(raw_dir.glob("eia930-*.zip"))


def _read_balance_csv(path: Path, chunksize: int = 200_000):
    """Yield chunks from a balance CSV (handles zipped or plain files)."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            balance_name = next(
                (n for n in zf.namelist() if n.endswith("-balance.csv")), None
            )
            if not balance_name:
                return
            with zf.open(balance_name) as f:
                for chunk in pd.read_csv(
                    f, usecols=lambda c: c in _ALL_USECOLS, chunksize=chunksize
                ):
                    yield chunk
    else:
        for chunk in pd.read_csv(
            path, usecols=lambda c: c in _ALL_USECOLS, chunksize=chunksize
        ):
            yield chunk


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column present in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    """Return a numeric Series for *col*, or 0.0 if col is ``None``."""
    if col is None:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw EIA‑930 rows into the standard schema.

    Handles all three CSV schema eras automatically by trying columns
    in priority order: battery‑storage‑split → aggregate → zero.
    """
    demand_col = _pick(df, ["Demand (MW) (Adjusted)", "Demand (MW)"])
    if demand_col is None:
        raise ValueError("Demand column not found")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                df["UTC Time at End of Hour"], utc=True, errors="coerce"
            ),
            "load_mw": pd.to_numeric(df[demand_col], errors="coerce"),
        }
    )

    # ── Wind ──────────────────────────────────────────────────────────
    # Priority: split (wo+w) → aggregate (Adjusted, then raw)
    wind_wo = _pick(df, [
        "Net Generation (MW) from Wind without Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Wind without Integrated Battery Storage",
    ])
    wind_w = _pick(df, [
        "Net Generation (MW) from Wind with Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Wind with Integrated Battery Storage",
    ])
    wind_agg = _pick(df, [
        "Net Generation (MW) from Wind (Adjusted)",
        "Net Generation (MW) from Wind",
    ])

    if wind_wo or wind_w:
        out["wind_mw"] = _safe_numeric(df, wind_wo) + _safe_numeric(df, wind_w)
    elif wind_agg:
        out["wind_mw"] = _safe_numeric(df, wind_agg)
    else:
        out["wind_mw"] = 0.0

    # ── Solar ─────────────────────────────────────────────────────────
    solar_wo = _pick(df, [
        "Net Generation (MW) from Solar without Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Solar without Integrated Battery Storage",
    ])
    solar_w = _pick(df, [
        "Net Generation (MW) from Solar witho Integrated Battery Storage (Adjusted)",  # EIA typo
        "Net Generation (MW) from Solar with Integrated Battery Storage (Adjusted)",
        "Net Generation (MW) from Solar with Integrated Battery Storage",
    ])
    solar_agg = _pick(df, [
        "Net Generation (MW) from Solar (Adjusted)",
        "Net Generation (MW) from Solar",
    ])

    if solar_wo or solar_w:
        out["solar_mw"] = _safe_numeric(df, solar_wo) + _safe_numeric(df, solar_w)
    elif solar_agg:
        out["solar_mw"] = _safe_numeric(df, solar_agg)
    else:
        out["solar_mw"] = 0.0

    # ── Fuel mix (optional — for carbon accuracy) ─────────────────────
    for short, adj, raw in [
        ("coal_mw", "Net Generation (MW) from Coal (Adjusted)", "Net Generation (MW) from Coal"),
        ("gas_mw", "Net Generation (MW) from Natural Gas (Adjusted)", "Net Generation (MW) from Natural Gas"),
        ("nuclear_mw", "Net Generation (MW) from Nuclear (Adjusted)", "Net Generation (MW) from Nuclear"),
        ("hydro_mw", "Net Generation (MW) from Hydropower Excluding Pumped Storage (Adjusted)",
         "Net Generation (MW) from Hydropower Excluding Pumped Storage"),
    ]:
        col = _pick(df, [adj, raw])
        if col:
            out[short] = _safe_numeric(df, col)

    return out


# ---------------------------------------------------------------------------
# Weather: Open-Meteo historical archive for MISO region (Chicago)
# ---------------------------------------------------------------------------
def fetch_weather_openmeteo(
    start: str, end: str,
    lat: float = 41.88, lon: float = -87.63,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Download hourly weather from Open-Meteo archive API for MISO/Chicago."""
    import urllib.request, json

    variables = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "cloud_cover", "wind_speed_10m", "surface_pressure",
        "shortwave_radiation",
    ]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&hourly={','.join(variables)}&timezone=UTC"
    )
    print(f"  Fetching weather: {url[:120]}…")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read().decode())

    hourly = data["hourly"]
    df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"], utc=True)})
    for v in variables:
        df[f"wx_{v}"] = pd.to_numeric(pd.Series(hourly.get(v, [])), errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  Saved weather → {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_eia930_features(
    raw_dir: Path,
    out_dir: Path,
    ba: str,
    start: str | None,
    end: str | None,
    *,
    weather: bool = True,
    holidays_country: str = "US",
):
    """Build model-ready features from EIA‑930 data."""
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    # Prefer extracted CSVs (faster), fall back to ZIPs.
    files = _iter_balance_files(raw_dir / "extracted")
    if not files:
        files = _iter_zip_files(raw_dir)
    if not files:
        files = _iter_balance_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No EIA‑930 files found in {raw_dir}")

    print(f"Reading {len(files)} files for BA={ba} …")
    for f in files:
        file_rows = 0
        for chunk in _read_balance_csv(f):
            if ba:
                chunk = chunk[chunk["Balancing Authority"] == ba]
            if chunk.empty:
                continue
            normed = _normalize(chunk)
            file_rows += len(normed)
            frames.append(normed)
        print(f"  {f.name}: {file_rows:,} rows")

    if not frames:
        raise ValueError(f"No records for BA='{ba}'. Check spelling.")

    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="last")

    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end, utc=True)]

    # Enforce continuous hourly index and interpolate short gaps.
    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz="UTC")
    df = df.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()

    core_cols = ["load_mw", "wind_mw", "solar_mw"]
    fuel_cols = [c for c in ("coal_mw", "gas_mw", "nuclear_mw", "hydro_mw") if c in df.columns]
    for col in core_cols + fuel_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(limit=6)

    # ── Feature engineering ───────────────────────────────────────────
    df = add_time_features(df)
    df = add_domain_features(df)

    # Holidays (US federal)
    df = add_holiday_features(df, country=holidays_country)

    # Price + carbon proxies
    df = add_price_carbon_features(df, price_col="price_usd_mwh", base_price=45.0)

    # ── Weather merge (Open-Meteo archive) ────────────────────────────
    if weather:
        wx_cache = raw_dir / "weather_chicago_hourly.csv"
        date_min = df["timestamp"].min().strftime("%Y-%m-%d")
        date_max = df["timestamp"].max().strftime("%Y-%m-%d")
        try:
            if wx_cache.exists():
                print("  Loading cached weather …")
                wx = pd.read_csv(wx_cache)
                wx["timestamp"] = pd.to_datetime(wx["timestamp"], utc=True)
            else:
                wx = fetch_weather_openmeteo(date_min, date_max, out_path=wx_cache)
            df = df.merge(wx, on="timestamp", how="left")
            wx_cols = [c for c in df.columns if c.startswith("wx_")]
            for col in wx_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(limit=6)
            print(f"  Merged {len(wx_cols)} weather features")
        except Exception as e:
            print(f"  ⚠ Weather fetch failed ({e}), continuing without weather")

    # ── Lags / rolling stats ──────────────────────────────────────────
    lag_cols = list(core_cols)
    for extra in ["price_usd_mwh", "carbon_kg_per_mwh"]:
        if extra in df.columns:
            lag_cols.append(extra)
    # Weather lags
    wx_lag_cols = [c for c in df.columns if c.startswith("wx_")]
    lag_cols.extend(wx_lag_cols)

    df = add_lags_rolls(df, cols=lag_cols)
    df = df.dropna().reset_index(drop=True)

    # ── Persist ───────────────────────────────────────────────────────
    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")
    print(f"   Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"   Wind  non-zero: {(df['wind_mw'] > 0).mean():.1%}")
    print(f"   Solar non-zero: {(df['solar_mw'] > 0).mean():.1%}")
    return df


def main():
    """CLI entrypoint for EIA‑930 feature generation."""
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True, help="Input directory (data/raw/us_eia930)")
    p.add_argument("--out", dest="out_dir", default="data/processed/us_eia930", help="Output directory")
    p.add_argument("--ba", default="MISO", help="Balancing Authority code (e.g., MISO, PJM, CAISO)")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--no-weather", action="store_true", help="Skip weather fetch")
    p.add_argument("--holiday-country", default="US", help="Holiday country code")
    args = p.parse_args()

    build_eia930_features(
        Path(args.in_dir), Path(args.out_dir), args.ba, args.start, args.end,
        weather=not args.no_weather, holidays_country=args.holiday_country,
    )


if __name__ == "__main__":
    main()
