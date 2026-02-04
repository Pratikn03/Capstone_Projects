"""Download Electricity Maps carbon intensity via API into signals format."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
import sys

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.utils.logging import get_logger
from gridpulse.utils.net import get_session


def _parse_time(value: str) -> datetime:
    return pd.to_datetime(value, utc=True).to_pydatetime()


def _iter_ranges(start: datetime, end: datetime, chunk_hours: int) -> Iterable[tuple[datetime, datetime]]:
    cur = start
    delta = timedelta(hours=chunk_hours)
    while cur < end:
        nxt = min(end, cur + delta)
        yield cur, nxt
        cur = nxt


def _extract_rows(payload: dict | list) -> list[dict]:
    if isinstance(payload, list):
        return payload
    for key in ("history", "data", "results"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    raise ValueError("Unsupported response shape. Expected list or payload['history'|'data'|'results'].")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, help="Zone code (e.g., DE-LU)")
    ap.add_argument("--start", required=True, help="Start datetime (ISO 8601, UTC)")
    ap.add_argument("--end", required=True, help="End datetime (ISO 8601, UTC)")
    ap.add_argument("--out", default="data/raw/carbon_signals.csv")
    ap.add_argument("--token", default=os.getenv("ELECTRICITYMAPS_TOKEN"))
    ap.add_argument("--base-url", default="https://api.electricitymaps.com")
    ap.add_argument("--endpoint", default="/v3/carbon-intensity/past-range")
    ap.add_argument("--chunk-hours", type=int, default=168, help="Chunk size to avoid API limits")
    ap.add_argument("--timestamp-key", default="datetime", help="Timestamp key in response rows")
    ap.add_argument("--value-key", default="carbonIntensity", help="Carbon intensity key in response rows")
    ap.add_argument("--retries", type=int, default=3, help="HTTP retry attempts")
    ap.add_argument("--backoff", type=float, default=0.5, help="Retry backoff factor (seconds)")
    args = ap.parse_args()

    if not args.token:
        raise RuntimeError("Set ELECTRICITYMAPS_TOKEN or pass --token")

    log = get_logger("gridpulse.download_emaps")
    session = get_session(retries=args.retries, backoff=args.backoff)

    start = _parse_time(args.start)
    end = _parse_time(args.end)

    headers = {
        "Authorization": f"Bearer {args.token}",
        "auth-token": args.token,  # some deployments expect this header
    }

    rows: list[dict] = []
    for s, e in _iter_ranges(start, end, args.chunk_hours):
        params = {
            "zone": args.zone,
            "start": s.isoformat().replace("+00:00", "Z"),
            "end": e.isoformat().replace("+00:00", "Z"),
        }
        url = args.base_url.rstrip("/") + args.endpoint
        try:
            resp = session.get(url, headers=headers, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            log.error("Failed to fetch Electricity Maps data for %s to %s", s, e, exc_info=exc)
            raise
        rows.extend(_extract_rows(payload))

    if not rows:
        raise RuntimeError("No data returned for the requested window.")

    df = pd.DataFrame(rows)
    ts_key = args.timestamp_key if args.timestamp_key in df.columns else None
    if ts_key is None:
        for k in ("datetime", "timestamp", "time", "periodStart"):
            if k in df.columns:
                ts_key = k
                break
    if ts_key is None:
        raise ValueError(f"Could not find timestamp column in response. Columns: {list(df.columns)[:20]}")

    val_key = args.value_key if args.value_key in df.columns else None
    if val_key is None:
        for k in ("carbonIntensity", "carbon_intensity", "carbonIntensityAvg"):
            if k in df.columns:
                val_key = k
                break
    if val_key is None:
        raise ValueError(f"Could not find carbon intensity column in response. Columns: {list(df.columns)[:20]}")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_key], utc=True, errors="coerce"),
            "carbon_kg_per_mwh": pd.to_numeric(df[val_key], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info("Wrote %s (%s rows)", out_path, len(out))


if __name__ == "__main__":
    main()
