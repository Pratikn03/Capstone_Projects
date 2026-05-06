"""Hybrid delta data refresh for publish audits.

This script is intentionally conservative:
- Detects latest local timestamps for DE/US datasets.
- Optionally runs dataset refresh hooks (currently best-effort placeholders).
- Re-validates and de-duplicates feature parquet files by timestamp.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


DATASETS = {
    "DE": {
        "features": Path("data/processed/features.parquet"),
        "raw_candidates": [
            Path("data/raw/time_series_60min_singleindex.csv"),
            Path("data/raw/opsd_de.csv"),
        ],
        "required_columns": {"timestamp", "load_mw", "wind_mw", "solar_mw"},
        "refresh_cmd": [sys.executable, "-m", "orius.data_pipeline.download_opsd", "--out", "data/raw"],
    },
    "US": {
        "features": Path("data/processed/us_eia930/features.parquet"),
        "raw_candidates": [
            Path("data/raw/us_eia930"),
        ],
        "required_columns": {"timestamp", "load_mw", "wind_mw", "solar_mw"},
        "refresh_cmd": [
            sys.executable,
            "-m",
            "orius.data_pipeline.build_features_eia930",
            "--in",
            "data/raw/us_eia930",
            "--out",
            "data/processed/us_eia930",
            "--ba",
            "MISO",
        ],
    },
}

TS_COL_CANDIDATES = (
    "timestamp",
    "ts_utc",
    "utc_timestamp",
    "datetime_utc",
    "time",
    "date",
)


def _read_timestamp_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    if path.is_dir():
        values: list[pd.Timestamp] = []
        for child in sorted(path.glob("*.csv"))[:40]:
            series = _read_timestamp_series(child)
            if series is not None and not series.empty:
                values.append(series.max())
        if not values:
            return None
        return pd.Series(values)
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None
    elif path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, nrows=200000)
        except Exception:
            return None
    else:
        return None
    for col in TS_COL_CANDIDATES:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=True).dropna()
            if not ts.empty:
                return ts
    return None


def _latest_local_timestamp(paths: list[Path]) -> str | None:
    maxima: list[pd.Timestamp] = []
    for p in paths:
        ts = _read_timestamp_series(p)
        if ts is not None and not ts.empty:
            maxima.append(ts.max())
    if not maxima:
        return None
    max_ts = max(maxima)
    if max_ts.tzinfo is None:
        max_ts = max_ts.tz_localize("UTC")
    return max_ts.isoformat()


def _validate_and_dedup_features(
    features_path: Path, required_columns: set[str], apply_changes: bool
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(features_path),
        "exists": features_path.exists(),
        "schema_ok": False,
        "rows_before": 0,
        "rows_after": 0,
        "duplicates_removed": 0,
        "timestamp_max": None,
    }
    if not features_path.exists():
        return out

    df = pd.read_parquet(features_path)
    out["rows_before"] = int(len(df))
    missing = sorted(required_columns - set(df.columns))
    out["missing_columns"] = missing
    out["schema_ok"] = len(missing) == 0

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        out["timestamp_max"] = ts.max().isoformat() if ts.notna().any() else None
        df = df.assign(timestamp=ts).sort_values("timestamp")
        deduped = df.drop_duplicates(subset=["timestamp"], keep="last")
    else:
        deduped = df

    out["rows_after"] = int(len(deduped))
    out["duplicates_removed"] = max(0, out["rows_before"] - out["rows_after"])

    if apply_changes and out["duplicates_removed"] > 0:
        features_path.parent.mkdir(parents=True, exist_ok=True)
        deduped.to_parquet(features_path, index=False)
        out["applied"] = True
    else:
        out["applied"] = False
    return out


def _run_refresh_hook(cmd: list[str], enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"attempted": False, "status": "skipped"}
    try:
        subprocess.run(cmd, check=True)
        return {"attempted": True, "status": "ok", "cmd": cmd}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"attempted": True, "status": "failed", "cmd": cmd, "error": str(exc)}


def run_refresh(*, dataset_filter: str, apply_changes: bool, run_hooks: bool) -> dict[str, Any]:
    selected = sorted(DATASETS.keys()) if dataset_filter == "ALL" else [dataset_filter]
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_filter": dataset_filter,
        "apply_changes": apply_changes,
        "run_hooks": run_hooks,
        "datasets": {},
    }
    for key in selected:
        cfg = DATASETS[key]
        latest_local = _latest_local_timestamp([*list(cfg["raw_candidates"]), cfg["features"]])
        refresh_status = _run_refresh_hook(cfg["refresh_cmd"], enabled=run_hooks)
        features_status = _validate_and_dedup_features(
            features_path=cfg["features"],
            required_columns=set(cfg["required_columns"]),
            apply_changes=apply_changes,
        )
        summary["datasets"][key] = {
            "latest_local_timestamp": latest_local,
            "refresh": refresh_status,
            "features": features_status,
        }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid delta refresh for DE/US datasets")
    parser.add_argument("--dataset", choices=["DE", "US", "ALL"], default="ALL")
    parser.add_argument("--apply", action="store_true", help="Write de-dup changes to feature parquet files")
    parser.add_argument("--run-hooks", action="store_true", help="Run dataset refresh commands")
    parser.add_argument("--out", default="reports/publish/data_refresh_summary.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_refresh(
        dataset_filter=args.dataset,
        apply_changes=bool(args.apply),
        run_hooks=bool(args.run_hooks),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
