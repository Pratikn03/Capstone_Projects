"""Deterministic, integrity-checked split carving for release runs.

Replicates the exact ``train_end / calibration_start / val_start / test_start``
indexing the legacy trainer computes in ``orius.forecasting.train.main`` so the
parquet slices written here are byte-identical to what the GBM/DL trainer
loads in-memory from ``features.parquet``. Every model in the release sees the
same rows; the manifest records a single ``splits_sha256`` that proves it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from orius.forecasting.train_config import (
    resolve_gap_hours,
    resolve_split_cfg,
    resolve_task_horizon,
    resolve_task_lookback,
)


@dataclass(frozen=True)
class SplitsConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    calibration_ratio: float
    gap_hours: int
    horizon: int
    lookback: int
    timestamp_col: str
    sort_cols: tuple[str, ...]


@dataclass(frozen=True)
class CarvedSplits:
    train_path: Path
    calibration_path: Path
    val_path: Path
    test_path: Path
    boundaries: dict[str, int]
    n_rows: int
    splits_sha256: str
    features_sha256: str


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_sort_cols(data_cfg: dict) -> tuple[str, ...]:
    raw = data_cfg.get("order_cols")
    if isinstance(raw, list) and raw:
        return tuple(str(c).strip() for c in raw if str(c).strip())
    timestamp_col = data_cfg.get("timestamp_col", "timestamp")
    return (str(timestamp_col),)


def splits_config_from_yaml(cfg: dict) -> SplitsConfig:
    train_ratio, val_ratio, test_ratio, calibration_ratio = resolve_split_cfg(cfg)
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    task_cfg = cfg.get("task", {}) if isinstance(cfg.get("task"), dict) else {}
    return SplitsConfig(
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        calibration_ratio=float(calibration_ratio),
        gap_hours=int(resolve_gap_hours(cfg)),
        horizon=int(resolve_task_horizon(task_cfg)),
        lookback=int(resolve_task_lookback(task_cfg)),
        timestamp_col=str(data_cfg.get("timestamp_col", "timestamp")),
        sort_cols=_resolve_sort_cols(data_cfg),
    )


def _sort_frame(df: pd.DataFrame, sort_cols: tuple[str, ...]) -> pd.DataFrame:
    available = [c for c in sort_cols if c in df.columns]
    if not available:
        return df.reset_index(drop=True)
    return df.sort_values(by=list(available), kind="mergesort").reset_index(drop=True)


def _slice_indices(n: int, cfg: SplitsConfig) -> dict[str, int]:
    train_end = int(n * cfg.train_ratio)
    calibration_start = min(n, train_end + cfg.gap_hours)
    calibration_end = min(n, calibration_start + int(n * cfg.calibration_ratio))
    val_start = min(n, calibration_end + cfg.gap_hours)
    val_end = min(n, val_start + int(n * cfg.val_ratio))
    test_start = min(n, val_end + cfg.gap_hours)
    return {
        "n_rows": int(n),
        "train_end": int(train_end),
        "calibration_start": int(calibration_start),
        "calibration_end": int(calibration_end),
        "val_start": int(val_start),
        "val_end": int(val_end),
        "test_start": int(test_start),
        "gap_hours": int(cfg.gap_hours),
    }


def sha256_splits(*paths: Path) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        digest = _sha256_file(path)
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"=")
        hasher.update(digest.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def carve_splits(
    *,
    features_path: Path,
    out_dir: Path,
    cfg: SplitsConfig,
) -> CarvedSplits:
    """Carve the canonical train/cal/val/test slices and persist them as parquet."""
    if not features_path.exists():
        raise FileNotFoundError(f"features parquet not found: {features_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _sort_frame(pd.read_parquet(features_path), cfg.sort_cols)
    n = len(df)
    if n == 0:
        raise ValueError(f"features parquet is empty: {features_path}")
    boundaries = _slice_indices(n, cfg)

    train_df = df.iloc[: boundaries["train_end"]]
    cal_df = df.iloc[boundaries["calibration_start"] : boundaries["calibration_end"]]
    val_df = df.iloc[boundaries["val_start"] : boundaries["val_end"]]
    test_df = df.iloc[boundaries["test_start"] :]
    if len(test_df) == 0:
        test_df = val_df.copy()

    paths = {
        "train": out_dir / "train.parquet",
        "calibration": out_dir / "calibration.parquet",
        "val": out_dir / "val.parquet",
        "test": out_dir / "test.parquet",
    }
    for key, frame in (
        ("train", train_df),
        ("calibration", cal_df),
        ("val", val_df),
        ("test", test_df),
    ):
        frame.reset_index(drop=True).to_parquet(paths[key])

    splits_sha = sha256_splits(paths["train"], paths["calibration"], paths["val"], paths["test"])
    feat_sha = _sha256_file(features_path)

    metadata_path = out_dir / "splits_manifest.json"
    metadata_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "boundaries": boundaries,
                "splits_sha256": splits_sha,
                "features_sha256": feat_sha,
                "row_counts": {
                    "train": int(len(train_df)),
                    "calibration": int(len(cal_df)),
                    "val": int(len(val_df)),
                    "test": int(len(test_df)),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return CarvedSplits(
        train_path=paths["train"],
        calibration_path=paths["calibration"],
        val_path=paths["val"],
        test_path=paths["test"],
        boundaries=boundaries,
        n_rows=int(n),
        splits_sha256=splits_sha,
        features_sha256=feat_sha,
    )
