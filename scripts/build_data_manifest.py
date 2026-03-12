#!/usr/bin/env python3
"""Build deterministic data identity manifest for raw/processed/split artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


DATASET_PATHS: dict[str, dict[str, Any]] = {
    "DE": {
        "features": Path("data/processed/features.parquet"),
        "splits_dir": Path("data/processed/splits"),
    },
    "US": {
        "features": Path("data/processed/us_eia930/features.parquet"),
        "splits_dir": Path("data/processed/us_eia930/splits"),
    },
    "US_MISO": {
        "features": Path("data/processed/us_eia930/features.parquet"),
        "splits_dir": Path("data/processed/us_eia930/splits"),
    },
    "US_PJM": {
        "features": Path("data/processed/us_eia930_pjm/features.parquet"),
        "splits_dir": Path("data/processed/us_eia930_pjm/splits"),
    },
    "US_ERCOT": {
        "features": Path("data/processed/us_eia930_ercot/features.parquet"),
        "splits_dir": Path("data/processed/us_eia930_ercot/splits"),
    },
}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _schema_hash_frame(df: pd.DataFrame) -> str:
    schema_rows = sorted((str(col), str(dtype)) for col, dtype in df.dtypes.items())
    payload = json.dumps(schema_rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _schema_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        return _schema_hash_frame(frame)
    if path.suffix == ".csv":
        frame = pd.read_csv(path, nrows=200)
        return _schema_hash_frame(frame)
    return None


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256(path),
        "schema_hash": _schema_hash(path),
    }


def _split_boundaries(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
        start = ts.min().isoformat() if not ts.empty else None
        end = ts.max().isoformat() if not ts.empty else None
    else:
        start = None
        end = None
    return {
        "exists": True,
        "rows": int(len(df)),
        "start_utc": start,
        "end_utc": end,
    }


def _discover_raw_files() -> list[Path]:
    raw_root = REPO_ROOT / "data" / "raw"
    if not raw_root.exists():
        return []
    patterns = ("**/*.csv", "**/*.json", "**/*.zip", "**/*.xlsx", "**/*.parquet")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(p for p in raw_root.glob(pattern) if p.is_file()))
    return sorted(set(files))


def build_manifest(datasets: list[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in _discover_raw_files():
        records.append(_file_record(path))

    datasets_payload: dict[str, Any] = {}
    for name in datasets:
        cfg = DATASET_PATHS[name]
        features_path = REPO_ROOT / cfg["features"]
        splits_dir = REPO_ROOT / cfg["splits_dir"]
        split_files = {
            "train": splits_dir / "train.parquet",
            "calibration": splits_dir / "calibration.parquet",
            "val": splits_dir / "val.parquet",
            "test": splits_dir / "test.parquet",
        }

        if features_path.exists():
            records.append(_file_record(features_path))
        for p in split_files.values():
            if p.exists():
                records.append(_file_record(p))

        datasets_payload[name] = {
            "features_path": str(features_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "features_schema_hash": _schema_hash(features_path),
            "split_paths": {
                key: str(path.relative_to(REPO_ROOT)).replace("\\", "/")
                for key, path in split_files.items()
            },
            "split_boundaries": {
                key: _split_boundaries(path) for key, path in split_files.items()
            },
            "split_schema_hashes": {
                key: _schema_hash(path) for key, path in split_files.items() if path.exists()
            },
        }

    records = sorted(records, key=lambda row: row["path"])
    aggregated_split_boundaries = {
        name: payload.get("split_boundaries", {}) for name, payload in datasets_payload.items()
    }
    schema_components = sorted(
        f"{row['path']}::{row.get('schema_hash')}"
        for row in records
        if row.get("schema_hash")
    )
    aggregate_schema_hash = hashlib.sha256(
        json.dumps(schema_components, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    manifest_core = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets_payload,
        "files": records,
        "split_boundaries": aggregated_split_boundaries,
        "schema_hash": aggregate_schema_hash,
    }
    manifest_bytes = json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    manifest_core["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest_core


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic data identity manifest")
    parser.add_argument("--dataset", choices=["DE", "US", "US_MISO", "US_PJM", "US_ERCOT", "ALL"], default="ALL")
    parser.add_argument("--output", default="data/dashboard/data_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = ["DE", "US_MISO", "US_PJM", "US_ERCOT"] if args.dataset == "ALL" else [args.dataset]
    datasets = ["US_MISO" if dataset == "US" else dataset for dataset in datasets]
    payload = build_manifest(datasets=datasets)
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "datasets": datasets, "files": len(payload["files"])}, indent=2))


if __name__ == "__main__":
    main()
