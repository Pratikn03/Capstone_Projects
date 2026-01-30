"""Download hourly weather data for a location (default: Berlin) via Open-Meteo archive API.

Example:
  python -m gridpulse.data_pipeline.download_weather --out data/raw --start 2017-01-01 --end 2020-12-31
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
]

def _parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()

def _chunks(start: date, end: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)

def _fetch_chunk(base_url: str, lat: float, lon: float, start: date, end: date, hourly: List[str], tz: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(hourly),
        "timezone": tz,
    }
    r = requests.get(base_url, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()
    hourly_payload = payload.get("hourly", {})
    times = hourly_payload.get("time", [])
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame(hourly_payload)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.drop(columns=["time"])
    # Prefix weather columns to avoid collisions.
    rename = {c: f"wx_{c}" for c in df.columns if c != "timestamp"}
    df = df.rename(columns=rename)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw", help="Output directory")
    p.add_argument("--lat", type=float, default=52.52, help="Latitude (default: Berlin)")
    p.add_argument("--lon", type=float, default=13.405, help="Longitude (default: Berlin)")
    p.add_argument("--start", default="2017-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2020-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument("--hourly", default=",".join(DEFAULT_HOURLY), help="Comma-separated hourly variables")
    p.add_argument("--timezone", default="UTC", help="Timezone for API response")
    p.add_argument("--chunk-days", type=int, default=365, help="Chunk size to avoid huge requests")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Archive API base URL")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    p.add_argument("--filename", default="weather_berlin_hourly", help="Base filename (no extension)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    hourly = [v.strip() for v in args.hourly.split(",") if v.strip()]

    frames = []
    for s, e in _chunks(start, end, args.chunk_days):
        print(f"Fetching {s} to {e} ...")
        df = _fetch_chunk(args.base_url, args.lat, args.lon, s, e, hourly, args.timezone)
        if not df.empty:
            frames.append(df)

    if frames:
        out_df = pd.concat(frames, ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        out_df = pd.DataFrame(columns=["timestamp"])

    suffix = "csv" if args.format == "csv" else "parquet"
    out_path = out_dir / f"{args.filename}.{suffix}"

    if args.format == "csv":
        out_df.to_csv(out_path, index=False)
    else:
        out_df.to_parquet(out_path, index=False)

    (out_dir / "README_WEATHER.md").write_text(
        f"""Downloaded weather data:
- Location: lat={args.lat}, lon={args.lon}
- Date range: {args.start} to {args.end}
- Hourly vars: {", ".join(hourly)}
- File: {out_path.name}

Columns are prefixed with `wx_` and timestamps are UTC.
""",
        encoding="utf-8",
    )

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
