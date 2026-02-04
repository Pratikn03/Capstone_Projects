"""Download WattTime marginal emissions (MOER) history into signals format."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
from requests.auth import HTTPBasicAuth

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.utils.logging import get_logger
from gridpulse.utils.net import get_session


def _login(session, username: str, password: str) -> str:
    resp = session.get("https://api.watttime.org/login", auth=HTTPBasicAuth(username, password), timeout=30)
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise RuntimeError("WattTime login failed: no token")
    return token


def _fetch_historical(session, token: str, region: str, start: str, end: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "region": region,
        "start": start,
        "end": end,
        "signal_type": "co2_moer",
    }
    resp = session.get("https://api.watttime.org/v3/historical", headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["point_time"], utc=True, errors="coerce")
    df["moer_kg_per_mwh"] = pd.to_numeric(df["value"], errors="coerce") * 0.453592
    return df[["timestamp", "moer_kg_per_mwh"]].dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help="WattTime region (e.g., CAISO_NORTH, PJM, MISO)")
    ap.add_argument("--start", required=True, help="Start datetime (ISO 8601, UTC)")
    ap.add_argument("--end", required=True, help="End datetime (ISO 8601, UTC)")
    ap.add_argument("--out", default="data/raw/moer_signals.csv")
    ap.add_argument("--username", default=os.getenv("WATTTIME_USERNAME"))
    ap.add_argument("--password", default=os.getenv("WATTTIME_PASSWORD"))
    ap.add_argument("--retries", type=int, default=3, help="HTTP retry attempts")
    ap.add_argument("--backoff", type=float, default=0.5, help="Retry backoff factor (seconds)")
    args = ap.parse_args()

    if not args.username or not args.password:
        raise RuntimeError("Set WATTTIME_USERNAME and WATTTIME_PASSWORD or pass --username/--password")

    log = get_logger("gridpulse.download_watttime")
    session = get_session(retries=args.retries, backoff=args.backoff)

    try:
        token = _login(session, args.username, args.password)
    except Exception as exc:
        log.error("WattTime login failed", exc_info=exc)
        raise

    try:
        df = _fetch_historical(session, token, args.region, args.start, args.end)
    except Exception as exc:
        log.error("WattTime historical fetch failed", exc_info=exc)
        raise
    if df.empty:
        raise RuntimeError("No data returned from WattTime for the requested window.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote %s (%s rows)", out_path, len(df))


if __name__ == "__main__":
    main()
