"""Download SMARD generation mix and compute hourly carbon intensity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import sys

import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.utils.logging import get_logger
from gridpulse.utils.net import get_session


SMARD_BASE = "https://www.smard.de/app/chart_data"

# SMARD filter IDs (generation by type, hourly resolution).
SMARD_FILTERS = {
    "1223": "lignite",
    "4069": "hard_coal",
    "4071": "gas",
    "1224": "nuclear",
    "1225": "wind_offshore",
    "4067": "wind_onshore",
    "4068": "solar",
    "1226": "hydro",
    "4066": "biomass",
    "1227": "other_conventional",
    "1228": "other_renewables",
    "4070": "pumped_storage",
}


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _parse_time(value: str) -> datetime:
    return pd.to_datetime(value, utc=True).to_pydatetime()


def _load_factors(path: Path | None) -> dict[str, float]:
    defaults = {
        "coal": 820.0,
        "gas": 490.0,
        "biomass": 230.0,
        "solar": 48.0,
        "wind_onshore": 11.0,
        "wind_offshore": 12.0,
        "hydro": 24.0,
        "nuclear": 12.0,
        "other_conventional": 820.0,
        "other_renewables": 24.0,
        "pumped_storage": 0.0,
    }
    if not path:
        return defaults
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    factors = payload.get("factors_kg_per_mwh", {})
    if not factors:
        return defaults
    out = defaults.copy()
    out.update({k: float(v) for k, v in factors.items()})
    return out


def _fetch_index(session, filter_id: str, region: str, resolution: str = "hour") -> list[int]:
    url = f"{SMARD_BASE}/{filter_id}/{region}/index_{resolution}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict):
        stamps = payload.get("timestamps") or payload.get("data") or payload.get("series")
    else:
        stamps = payload
    if not stamps:
        return []
    return [int(x) for x in stamps]


def _fetch_series(session, filter_id: str, region: str, resolution: str, timestamp: int) -> list[tuple[int, float]]:
    url = f"{SMARD_BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    series = None
    if isinstance(payload, dict):
        series = payload.get("series") or payload.get("data") or payload.get("values")
    if series is None:
        raise ValueError(f"Unexpected payload for {filter_id}: {payload}")
    out = []
    for ts, val in series:
        if val is None:
            continue
        out.append((int(ts), float(val)))
    return out


def _iter_chunks(index: Iterable[int], start_ms: int, end_ms: int) -> list[int]:
    chunks = []
    for ts in sorted(index):
        if ts > end_ms:
            break
        if ts <= end_ms and ts >= start_ms:
            chunks.append(ts)
        # include one chunk before start to ensure coverage
        if ts < start_ms:
            last = ts
    if "last" in locals():
        chunks.insert(0, last)
    return sorted(set(chunks))


def _load_filter_series(session, filter_id: str, region: str, start: datetime, end: datetime) -> pd.DataFrame:
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    index = _fetch_index(session, filter_id, region)
    if not index:
        return pd.DataFrame(columns=["timestamp", "value"])
    chunks = _iter_chunks(index, start_ms, end_ms)
    rows = []
    for ts in chunks:
        for point_ts, val in _fetch_series(session, filter_id, region, "hour", ts):
            if point_ts < start_ms or point_ts > end_ms:
                continue
            rows.append((point_ts, val))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "value"])
    df = pd.DataFrame(rows, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="DE", help="SMARD region code (default: DE)")
    ap.add_argument("--start", required=True, help="Start datetime (ISO 8601, UTC)")
    ap.add_argument("--end", required=True, help="End datetime (ISO 8601, UTC)")
    ap.add_argument("--out", default="data/raw/carbon_signals.csv")
    ap.add_argument("--factors", default="configs/carbon_factors.yaml", help="YAML file with emission factors")
    ap.add_argument("--include-pumped", action="store_true", help="Include pumped storage in total generation")
    ap.add_argument("--cache-dir", default=None, help="Optional cache directory for raw series")
    ap.add_argument("--retries", type=int, default=3, help="HTTP retry attempts")
    ap.add_argument("--backoff", type=float, default=0.5, help="Retry backoff factor (seconds)")
    args = ap.parse_args()

    log = get_logger("gridpulse.download_smard")
    session = get_session(retries=args.retries, backoff=args.backoff)

    start = _parse_time(args.start)
    end = _parse_time(args.end)
    factors = _load_factors(Path(args.factors)) if args.factors else _load_factors(None)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for filter_id, label in SMARD_FILTERS.items():
        try:
            df = _load_filter_series(session, filter_id, args.region, start, end)
        except Exception as exc:
            log.error("Failed to fetch SMARD series %s", label, exc_info=exc)
            raise
        if cache_dir:
            df.to_parquet(cache_dir / f"{filter_id}_{label}.parquet", index=False)
        df = df.rename(columns={"value": label})
        frames.append(df)

    if not frames:
        raise RuntimeError("No SMARD series loaded.")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="timestamp", how="outer")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    for col in merged.columns:
        if col != "timestamp":
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # Map fuel categories to emission factors.
    def _factor_for(label: str) -> float:
        if label in ("lignite", "hard_coal"):
            return factors["coal"]
        if label in factors:
            return factors[label]
        return 0.0

    gen_cols = [c for c in merged.columns if c != "timestamp"]
    if not args.include_pumped and "pumped_storage" in gen_cols:
        gen_cols.remove("pumped_storage")

    total_gen = merged[gen_cols].clip(lower=0.0).sum(axis=1)
    total_gen = total_gen.replace(0.0, pd.NA)

    weighted = 0.0
    for col in gen_cols:
        factor = _factor_for(col)
        weighted = weighted + merged[col].clip(lower=0.0) * factor

    carbon = (weighted / total_gen).astype(float)
    out = pd.DataFrame({"timestamp": merged["timestamp"], "carbon_kg_per_mwh": carbon}).dropna()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info("Wrote %s (%s rows)", out_path, len(out))


if __name__ == "__main__":
    main()
