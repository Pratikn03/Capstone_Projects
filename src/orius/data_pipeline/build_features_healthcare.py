"""Build features for Healthcare domain from ORIUS vital signs CSV.

The legacy builder reads ``data/healthcare/processed/healthcare_orius.csv``
and reproduces the original single-surface feature contract. The max-input
builder is opt-in and merges the richer BIDMC bridge plus the staged MIMIC-III
bridge into a per-patient feature surface suitable for larger healthcare
training runs without disturbing the canonical submission-facing path.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from orius.data_pipeline.split_time_series import time_split_with_calibration

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = REPO_ROOT / "data" / "healthcare" / "processed" / "healthcare_orius.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "healthcare" / "processed"
DEFAULT_BIDMC_BRIDGE = REPO_ROOT / "data" / "healthcare" / "processed" / "healthcare_bidmc_orius.csv"
DEFAULT_MIMIC3_BRIDGE = (
    REPO_ROOT / "data" / "healthcare" / "mimic3" / "processed" / "mimic3_healthcare_orius.csv"
)
DEFAULT_MAX_INPUT_OUT = REPO_ROOT / "data" / "healthcare" / "max_input"
TARGET = "hr_bpm"
LAG_STEPS = [1, 2, 4, 8, 12, 24]
ROLL_WINDOWS = [6, 12, 24]
MAX_INPUT_COLUMNS = [
    "timestamp",
    "ts_utc",
    "patient_id",
    "source_dataset",
    "spo2_pct",
    "hr_bpm",
    "respiratory_rate",
    "pulse_bpm",
    "reliability",
    "forecast_spo2_pct",
    "shock_index",
]
SPLIT_LABELS = ("train", "calibration", "val", "test")


def _format_timestamp(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return pd.Timestamp(value).isoformat()


def _frame_timestamps(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()


def _frame_patient_ids(df: pd.DataFrame) -> pd.Series:
    if "patient_id" not in df.columns:
        return pd.Series(dtype="object")
    return df["patient_id"].astype(str).dropna()


def _pair_overlap(a: pd.Series, b: pd.Series) -> int:
    if a.empty or b.empty:
        return 0
    return int(len(set(a.astype(str)) & set(b.astype(str))))


def _split_summary_markdown(
    *,
    source_name: str,
    split_strategy: str,
    total_rows: int,
    total_patients: int,
    train_ratio: float,
    calibration_ratio: float,
    val_ratio: float,
    splits: dict[str, pd.DataFrame],
) -> str:
    target_ratios = {
        "train": train_ratio,
        "calibration": calibration_ratio,
        "val": val_ratio,
        "test": 1.0 - train_ratio - calibration_ratio - val_ratio,
    }
    split_rows: list[str] = []
    timestamp_series = {name: _frame_timestamps(frame) for name, frame in splits.items()}
    patient_series = {name: _frame_patient_ids(frame) for name, frame in splits.items()}
    for name in SPLIT_LABELS:
        frame = splits[name]
        ts = timestamp_series[name]
        patients = patient_series[name]
        actual_ratio = (len(frame) / total_rows) if total_rows else 0.0
        split_rows.append(
            "| "
            f"{name} | {len(frame)} | {target_ratios[name]:.4f} | {actual_ratio:.4f} | "
            f"{patients.nunique()} | {_format_timestamp(ts.min() if not ts.empty else None)} | "
            f"{_format_timestamp(ts.max() if not ts.empty else None)} |"
        )

    overlap_rows: list[str] = []
    boundary_ok = True
    pairs = (
        ("train", "calibration"),
        ("calibration", "val"),
        ("val", "test"),
        ("train", "val"),
        ("train", "test"),
        ("calibration", "test"),
    )
    for left, right in pairs:
        left_ts = timestamp_series[left]
        right_ts = timestamp_series[right]
        left_patients = patient_series[left]
        right_patients = patient_series[right]
        timestamp_overlap = _pair_overlap(left_ts, right_ts)
        patient_overlap = _pair_overlap(left_patients, right_patients)
        if not left_ts.empty and not right_ts.empty and not (left_ts.max() < right_ts.min()):
            boundary_ok = False
        overlap_rows.append(f"| {left} / {right} | {timestamp_overlap} | {patient_overlap} |")

    return "\n".join(
        [
            "# Healthcare Split Summary",
            "",
            f"Source: `{source_name}`",
            f"Strategy: `{split_strategy}`",
            f"Total rows: {total_rows}",
            f"Total patients: {total_patients}",
            "",
            "## Split Metrics",
            "",
            "| Split | Rows | Target Ratio | Actual Ratio | Patients | Start | End |",
            "|---|---:|---:|---:|---:|---|---|",
            *split_rows,
            "",
            "## Integrity",
            "",
            f"- Timestamp boundary ordering: {'PASS' if boundary_ok else 'FAIL'}",
            "",
            "| Pair | Timestamp Overlap | Patient Overlap |",
            "|---|---:|---:|",
            *overlap_rows,
            "",
        ]
    )


def _select_patient_block_splits(
    df: pd.DataFrame,
    *,
    train_ratio: float,
    calibration_ratio: float,
    val_ratio: float,
) -> dict[str, pd.DataFrame] | None:
    if "patient_id" not in df.columns:
        return None

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = (
        frame.dropna(subset=["timestamp", "patient_id"])
        .sort_values(["timestamp", "patient_id"])
        .reset_index(drop=True)
    )
    patient_meta = (
        frame.groupby("patient_id", dropna=False)
        .agg(rows=("patient_id", "size"), start=("timestamp", "min"))
        .reset_index()
        .sort_values(["start", "patient_id"], kind="stable")
        .reset_index(drop=True)
    )
    if len(patient_meta) < len(SPLIT_LABELS):
        return None

    targets = (
        train_ratio,
        calibration_ratio,
        val_ratio,
        1.0 - train_ratio - calibration_ratio - val_ratio,
    )
    prefix = patient_meta["rows"].cumsum().tolist()
    total_rows = int(prefix[-1])

    def _rows(start: int, end: int) -> int:
        if start >= end:
            return 0
        return int(prefix[end - 1] - (prefix[start - 1] if start else 0))

    best_key: tuple[float, float, int, int, int] | None = None
    best_bounds: tuple[int, int, int] | None = None
    patient_count = len(patient_meta)

    for first in range(1, patient_count - 2):
        for second in range(first + 1, patient_count - 1):
            for third in range(second + 1, patient_count):
                counts = (
                    _rows(0, first),
                    _rows(first, second),
                    _rows(second, third),
                    _rows(third, patient_count),
                )
                ratios = tuple(count / total_rows for count in counts)
                errors = tuple(abs(actual - target) for actual, target in zip(ratios, targets, strict=False))
                key = (
                    round(sum(errors), 12),
                    round(max(errors), 12),
                    abs(counts[1] - counts[2]),
                    abs(counts[2] - counts[3]),
                    abs(counts[0] - counts[3]),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_bounds = (first, second, third)

    if best_bounds is None:
        return None

    first, second, third = best_bounds
    patient_lists = {
        "train": patient_meta.iloc[:first]["patient_id"].astype(str).tolist(),
        "calibration": patient_meta.iloc[first:second]["patient_id"].astype(str).tolist(),
        "val": patient_meta.iloc[second:third]["patient_id"].astype(str).tolist(),
        "test": patient_meta.iloc[third:]["patient_id"].astype(str).tolist(),
    }
    split_frames: dict[str, pd.DataFrame] = {}
    for name in SPLIT_LABELS:
        members = set(patient_lists[name])
        split_frames[name] = (
            frame.loc[frame["patient_id"].astype(str).isin(members)]
            .sort_values(["timestamp", "patient_id"], kind="stable")
            .reset_index(drop=True)
        )
    return split_frames


def _write_splits(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    source_name: str,
    train_ratio: float,
    calibration_ratio: float,
    val_ratio: float,
    gap_hours: float,
) -> None:
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_frames = _select_patient_block_splits(
        df,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
    )
    split_strategy = "contiguous_patient_blocks_by_earliest_timestamp"
    if split_frames is None:
        train, cal, val, test = time_split_with_calibration(
            df,
            ts_col="timestamp",
            train_ratio=train_ratio,
            calibration_ratio=calibration_ratio,
            val_ratio=val_ratio,
            gap_hours=gap_hours,
        )
        split_frames = {
            "train": train,
            "calibration": cal,
            "val": val,
            "test": test,
        }
        split_strategy = "row_time_fallback_insufficient_patients"

    for name, frame in split_frames.items():
        frame.to_parquet(splits_dir / f"{name}.parquet", index=False)

    summary = _split_summary_markdown(
        source_name=source_name,
        split_strategy=split_strategy,
        total_rows=int(len(df)),
        total_patients=int(df["patient_id"].astype(str).nunique()) if "patient_id" in df.columns else 0,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        splits=split_frames,
    )
    (splits_dir / "SPLIT_SUMMARY.md").write_text(summary, encoding="utf-8")


def _finalize_legacy_features(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    source_name: str,
    train_ratio: float,
    calibration_ratio: float,
    val_ratio: float,
    gap_hours: float,
) -> Path:
    features_path = out_dir / "features.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path, index=False)
    _write_splits(
        df,
        out_dir,
        source_name=source_name,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=gap_hours,
    )
    return features_path


def _legacy_build_features(
    csv_path: Path,
    out_dir: Path,
    *,
    train_ratio: float,
    calibration_ratio: float,
    val_ratio: float,
    gap_hours: float,
) -> Path:
    """Preserve the original healthcare feature contract for legacy callers."""
    df = pd.read_csv(csv_path)
    for column in ("hr_bpm", "spo2_pct", "respiratory_rate"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "patient_id" not in df.columns:
        df["patient_id"] = "legacy_patient"
    df["patient_id"] = df["patient_id"].astype(str)
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
    else:
        df["step"] = df.groupby("patient_id").cumcount()
    if "ts_utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    elif "Time [s]" in df.columns:
        df["timestamp"] = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(
            pd.to_numeric(df["Time [s]"], errors="coerce"), unit="s"
        )
    else:
        df["timestamp"] = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(df["step"], unit="s")
    df = df.dropna(subset=["patient_id", "timestamp", "hr_bpm", "spo2_pct", "respiratory_rate"]).copy()
    df["ts_utc"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.sort_values(["patient_id", "timestamp"], kind="stable").reset_index(drop=True)
    df = _engineer_max_input_features(
        df[["timestamp", "ts_utc", "patient_id", "step", "spo2_pct", "hr_bpm", "respiratory_rate"]]
    )
    return _finalize_legacy_features(
        df,
        out_dir,
        source_name=csv_path.name,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=gap_hours,
    )


def _extract_step_from_token(value: object) -> int:
    text = str(value)
    match = re.search(r"(?:_t|_s)(\d+)$", text)
    if match:
        return int(match.group(1))
    if text.isdigit():
        return int(text)
    return 0


def _parse_bool_like_series(values: pd.Series) -> pd.Series:
    normalized = values.fillna(False).map(
        lambda value: str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
    )
    return normalized.astype(bool)


def _normalize_bridge_source(path: Path, source_dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"patient_id", "target", "hr", "resp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required bridge columns: {sorted(missing)}")

    reliability = (
        pd.to_numeric(df["reliability"], errors="coerce")
        if "reliability" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    forecast = (
        pd.to_numeric(df["forecast"], errors="coerce")
        if "forecast" in df.columns
        else pd.to_numeric(df["target"], errors="coerce")
    )
    pulse = (
        pd.to_numeric(df["pulse"], errors="coerce")
        if "pulse" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="object")
    )
    is_critical = (
        _parse_bool_like_series(df["is_critical"])
        if "is_critical" in df.columns
        else pd.Series(False, index=df.index)
    )

    normalized = pd.DataFrame(
        {
            "patient_id": [f"{source_dataset}:{value}" for value in df["patient_id"].astype(str)],
            "source_dataset": source_dataset,
            "step": df["timestamp"].map(_extract_step_from_token)
            if "timestamp" in df.columns
            else range(len(df)),
            "spo2_pct": pd.to_numeric(df["target"], errors="coerce"),
            "hr_bpm": pd.to_numeric(df["hr"], errors="coerce"),
            "respiratory_rate": pd.to_numeric(df["resp"], errors="coerce"),
            "pulse_bpm": pulse,
            "reliability": reliability.fillna(1.0),
            "forecast_spo2_pct": forecast,
            "is_critical": is_critical,
        }
    )
    return normalized


def _normalize_legacy_source(path: Path, source_dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"patient_id", "hr_bpm", "spo2_pct", "respiratory_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required legacy columns: {sorted(missing)}")

    if "step" in df.columns:
        step = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(int)
    else:
        step = df.groupby("patient_id").cumcount()

    normalized = pd.DataFrame(
        {
            "patient_id": [f"{source_dataset}:{value}" for value in df["patient_id"].astype(str)],
            "source_dataset": source_dataset,
            "step": step,
            "spo2_pct": pd.to_numeric(df["spo2_pct"], errors="coerce"),
            "hr_bpm": pd.to_numeric(df["hr_bpm"], errors="coerce"),
            "respiratory_rate": pd.to_numeric(df["respiratory_rate"], errors="coerce"),
            "pulse_bpm": pd.NA,
            "reliability": 1.0,
            "forecast_spo2_pct": pd.to_numeric(df["spo2_pct"], errors="coerce"),
            "is_critical": False,
        }
    )
    return normalized


def _infer_source_name(path: Path) -> str:
    lower = path.name.lower()
    if "mimic" in lower:
        return "mimic3"
    if "bidmc" in lower:
        return "bidmc"
    return path.stem.lower().replace("-", "_")


def _normalize_input_source(path: Path) -> pd.DataFrame:
    sample = pd.read_csv(path, nrows=5)
    source_dataset = _infer_source_name(path)
    if {"target", "hr", "resp"}.issubset(sample.columns):
        return _normalize_bridge_source(path, source_dataset)
    if {"hr_bpm", "spo2_pct", "respiratory_rate"}.issubset(sample.columns):
        return _normalize_legacy_source(path, source_dataset)
    raise ValueError(f"Unsupported healthcare source schema in {path}")


def _assign_monotonic_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    ordered_ids = sorted(df["patient_id"].astype(str).unique())
    patient_offsets = {patient_id: idx for idx, patient_id in enumerate(ordered_ids)}

    normalized = df.copy()
    normalized["patient_order"] = normalized["patient_id"].map(patient_offsets).astype(int)
    normalized["timestamp"] = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(
        normalized["patient_order"] * 86400 + normalized["step"].astype(int),
        unit="s",
    )
    normalized["ts_utc"] = normalized["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    normalized = normalized.drop(columns=["patient_order"])
    return normalized


def _engineer_max_input_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["shock_index"] = work["hr_bpm"] / work["spo2_pct"].clip(lower=1.0)
    work = work.sort_values(["patient_id", "step"]).reset_index(drop=True)

    grouped = work.groupby("patient_id", group_keys=False)
    for feature in ("hr_bpm", "spo2_pct", "respiratory_rate", "shock_index"):
        for lag in LAG_STEPS:
            work[f"{feature}_lag{lag}"] = grouped[feature].shift(lag)

    for feature in ("hr_bpm", "spo2_pct", "respiratory_rate"):
        for window in ROLL_WINDOWS:
            rolling = grouped[feature].rolling(window=window, min_periods=window)
            work[f"{feature}_roll{window}_mean"] = rolling.mean().reset_index(level=0, drop=True)
            work[f"{feature}_roll{window}_std"] = rolling.std(ddof=0).reset_index(level=0, drop=True)

    work["hour"] = work["timestamp"].dt.hour
    work["minute"] = work["timestamp"].dt.minute
    work = work.dropna().reset_index(drop=True)
    ordered_cols = [column for column in MAX_INPUT_COLUMNS if column in work.columns]
    remaining_cols = [column for column in work.columns if column not in ordered_cols]
    return work[ordered_cols + remaining_cols]


def _write_max_input_manifest(
    normalized: pd.DataFrame,
    input_paths: Sequence[Path],
    out_dir: Path,
) -> None:
    manifest = {
        "input_paths": [str(path) for path in input_paths],
        "sources": sorted(normalized["source_dataset"].astype(str).unique()),
        "n_rows": int(len(normalized)),
        "n_patients": int(normalized["patient_id"].nunique()),
        "source_patient_counts": {
            source: int(group["patient_id"].nunique())
            for source, group in normalized.groupby("source_dataset")
        },
        "source_row_counts": {
            source: int(len(group)) for source, group in normalized.groupby("source_dataset")
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_max_input_features(
    input_paths: Sequence[Path] | None = None,
    out_dir: Path = DEFAULT_MAX_INPUT_OUT,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.05,
    val_ratio: float = 0.10,
    gap_hours: float = 0.0,
) -> Path:
    """Build a richer per-patient healthcare feature surface from multiple sources."""
    input_paths = list(input_paths or [DEFAULT_BIDMC_BRIDGE, DEFAULT_MIMIC3_BRIDGE])
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing healthcare max-input source(s): {missing}")

    normalized_frames = [_normalize_input_source(path) for path in input_paths]
    normalized = pd.concat(normalized_frames, ignore_index=True)
    normalized = normalized.dropna(subset=["spo2_pct", "hr_bpm", "respiratory_rate"]).reset_index(drop=True)
    normalized["reliability"] = (
        pd.to_numeric(normalized["reliability"], errors="coerce").fillna(1.0).clip(0.05, 1.0)
    )
    normalized = _assign_monotonic_timestamps(normalized)
    normalized["shock_index"] = normalized["hr_bpm"] / normalized["spo2_pct"].clip(lower=1.0)

    features = _engineer_max_input_features(normalized)

    out_dir.mkdir(parents=True, exist_ok=True)
    normalized[
        [column for column in [*MAX_INPUT_COLUMNS, "is_critical"] if column in normalized.columns]
    ].to_csv(
        out_dir / "healthcare_max_input_orius.csv",
        index=False,
    )
    features_path = out_dir / "features.parquet"
    features.to_parquet(features_path, index=False)
    _write_splits(
        features,
        out_dir,
        source_name=" + ".join(path.name for path in input_paths),
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=gap_hours,
    )
    _write_max_input_manifest(normalized, input_paths, out_dir)
    return features_path


def build_promoted_features(
    csv_path: Path = DEFAULT_MIMIC3_BRIDGE,
    out_dir: Path = DEFAULT_OUT,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.05,
    val_ratio: float = 0.10,
    gap_hours: float = 0.0,
) -> Path:
    """Build the promoted MIMIC-backed healthcare feature surface."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing promoted healthcare source: {csv_path}")

    normalized = _normalize_input_source(csv_path)
    normalized = normalized.dropna(subset=["spo2_pct", "hr_bpm", "respiratory_rate"]).reset_index(drop=True)
    normalized["reliability"] = (
        pd.to_numeric(normalized["reliability"], errors="coerce").fillna(1.0).clip(0.05, 1.0)
    )
    normalized = _assign_monotonic_timestamps(normalized)

    features = _engineer_max_input_features(normalized)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.parquet"
    features.to_parquet(features_path, index=False)
    _write_splits(
        features,
        out_dir,
        source_name=csv_path.name,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=gap_hours,
    )
    return features_path


def build_features(
    csv_path: Path,
    out_dir: Path,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.05,
    val_ratio: float = 0.10,
    gap_hours: float = 0.001,
) -> Path:
    """Build the legacy healthcare feature surface."""
    return _legacy_build_features(
        csv_path,
        out_dir,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=gap_hours,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Healthcare features for training")
    parser.add_argument("--in", dest="inputs", type=Path, action="append", default=None, help="Input CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument(
        "--max-input",
        action="store_true",
        help="Merge the richer BIDMC and MIMIC3 healthcare sources into a max-input feature surface",
    )
    args = parser.parse_args()
    if args.max_input:
        inputs = args.inputs or [DEFAULT_BIDMC_BRIDGE, DEFAULT_MIMIC3_BRIDGE]
        missing = [path for path in inputs if not path.exists()]
        if missing:
            return 1
        out_dir = args.out if args.out != DEFAULT_OUT else DEFAULT_MAX_INPUT_OUT
        build_max_input_features(inputs, out_dir)
        return 0

    input_path = args.inputs[0] if args.inputs else DEFAULT_RAW
    if not input_path.exists():
        return 1
    build_features(input_path, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
