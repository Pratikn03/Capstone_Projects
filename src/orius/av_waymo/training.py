"""Training and calibration helpers for the Waymo AV dry run."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from orius.forecasting.ml_gbm import predict_gbm, train_gbm
from orius.forecasting.uncertainty.conformal import ConformalConfig, ConformalInterval, save_conformal
from orius.forecasting.uncertainty.shift_aware import ShiftAwareConfig
from orius.release.artifact_loader import load_pickle_artifact

HORIZON_LABELS: dict[str, int] = {"1s": 10, "2s": 20, "4s": 40}
TARGETS = ("ego_speed_mps", "relative_gap_m")
IN_MEMORY_FEATURE_ROW_LIMIT = 500_000
FEATURE_BUILD_BATCH_ROWS = 200_000
FEATURE_WRITE_BATCH_ROWS = 50_000
GROUPED_ARCHIVE_DB_CITY_SPLIT = "grouped_archive_db_city"
SUPPORTED_SPLIT_STRATEGIES = ("hash", "balanced", "all_test", GROUPED_ARCHIVE_DB_CITY_SPLIT)
SPLIT_WEIGHTS: dict[str, float] = {
    "train": 0.70,
    "calibration": 0.10,
    "val": 0.10,
    "test": 0.10,
}
META_COLUMNS = {
    "scenario_id",
    "shard_id",
    "record_index",
    "step_index",
    "split",
    "speed_bin",
    "neighbor_count_bin",
    "object_mix_bin",
    "ego_track_id",
}


def _emit_progress(event: Mapping[str, Any]) -> None:
    pass


@dataclass(slots=True)
class ModelBundle:
    target: str
    horizon_label: str
    feature_columns: list[str]
    median_model: Any
    lower_model: Any
    upper_model: Any
    qhat: float
    feature_mean: dict[str, float]
    feature_std: dict[str, float]


def default_shift_aware_config(*, publication_dir: str | None = None) -> ShiftAwareConfig:
    cfg = ShiftAwareConfig(
        enabled=True,
        policy_mode="shift_aware_piecewise",
        aci_mode="aci_clipped",
        adaptation_step=0.02,
        alpha=0.10,
        alpha_min=0.05,
        alpha_max=0.20,
        coverage_target=0.90,
        coverage_window_size=128,
        max_inflation_multiplier=3.0,
    )
    if publication_dir is not None:
        cfg.publication_dir = str(publication_dir)
    return cfg


def _hash_percent(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 100


def assign_split(scenario_id: str) -> str:
    bucket = _hash_percent(str(scenario_id))
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "calibration"
    if bucket < 90:
        return "val"
    return "test"


def assign_balanced_splits(scenario_ids: Iterable[str]) -> dict[str, str]:
    """Assign deterministic non-empty train/calibration/val/test splits for bounded surfaces."""
    labels = ("train", "calibration", "val", "test")
    unique_ids = sorted(
        {str(scenario_id) for scenario_id in scenario_ids},
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(unique_ids) < len(labels):
        return {scenario_id: assign_split(scenario_id) for scenario_id in unique_ids}
    return {scenario_id: labels[index % len(labels)] for index, scenario_id in enumerate(unique_ids)}


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return str(value) != ""
    return str(value) != ""


def _split_strategy_error(split_strategy: str) -> ValueError:
    return ValueError(
        f"Unsupported split_strategy={split_strategy!r}; expected one of {SUPPORTED_SPLIT_STRATEGIES}."
    )


def _read_split_metadata(replay_windows_path: Path) -> tuple[pd.DataFrame, list[str]]:
    scenario_index_path = replay_windows_path.parent / "scenario_index.parquet"
    source_path = scenario_index_path if scenario_index_path.exists() else replay_windows_path
    schema_names = set(pq.ParquetFile(source_path).schema_arrow.names)
    desired = ["scenario_id", "source_archive_id", "db_entry", "location", "source_dataset", "shard_id"]
    columns = [column for column in desired if column in schema_names]
    if "scenario_id" not in columns:
        raise ValueError(f"{source_path} must include scenario_id for AV feature splitting.")
    return pd.read_parquet(source_path, columns=columns), columns


def _group_key(row: pd.Series) -> tuple[str, list[str]]:
    if _has_value(row.get("source_archive_id")) and _has_value(row.get("db_entry")):
        return f"archive={row['source_archive_id']}|db={row['db_entry']}", ["source_archive_id", "db_entry"]
    if _has_value(row.get("source_archive_id")) and _has_value(row.get("shard_id")):
        return f"archive={row['source_archive_id']}|shard={row['shard_id']}", [
            "source_archive_id",
            "shard_id",
        ]
    if _has_value(row.get("shard_id")):
        return f"shard={row['shard_id']}", ["shard_id"]
    return f"scenario={row['scenario_id']}", ["scenario_id"]


def _city_key(row: pd.Series) -> str:
    for column in ("location", "source_dataset", "shard_id"):
        value = row.get(column)
        if _has_value(value):
            return str(value)
    return "unknown"


def _target_split_counts(n_items: int) -> dict[str, int]:
    labels = tuple(SPLIT_WEIGHTS)
    raw_counts = {label: float(n_items) * SPLIT_WEIGHTS[label] for label in labels}
    counts = {label: int(np.floor(raw_counts[label])) for label in labels}
    remaining = n_items - sum(counts.values())
    remainders = sorted(
        labels,
        key=lambda label: (raw_counts[label] - counts[label], SPLIT_WEIGHTS[label], label),
        reverse=True,
    )
    for label in remainders[:remaining]:
        counts[label] += 1
    if n_items >= len(labels):
        for label in labels:
            if counts[label] > 0:
                continue
            donor = max(labels, key=lambda candidate: (counts[candidate], SPLIT_WEIGHTS[candidate]))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[label] = 1
    return counts


def assign_grouped_archive_db_city_splits(
    scenario_frame: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign 70/10/10/10 splits by nuPlan archive+DB group, stratified by city."""
    if "scenario_id" not in scenario_frame.columns:
        raise ValueError("scenario_frame must include scenario_id for grouped AV splitting.")
    frame = scenario_frame.copy()
    frame["scenario_id"] = frame["scenario_id"].astype(str)

    group_columns_seen: set[str] = set()
    group_keys: list[str] = []
    for _, row in frame.iterrows():
        key, columns = _group_key(row)
        group_keys.append(key)
        group_columns_seen.update(columns)
    frame["_split_group_key"] = group_keys
    frame["_split_city"] = frame.apply(_city_key, axis=1).astype(str)

    group_frame = (
        frame.groupby("_split_group_key", sort=False)
        .agg(_split_city=("_split_city", lambda values: "|".join(sorted({str(value) for value in values}))))
        .reset_index()
    )
    group_to_split: dict[str, str] = {}
    for city, city_groups in group_frame.groupby("_split_city", sort=True):
        ordered_group_keys = sorted(
            city_groups["_split_group_key"].astype(str).tolist(),
            key=lambda value: hashlib.sha256(f"{city}|{value}".encode()).hexdigest(),
        )
        target_counts = _target_split_counts(len(ordered_group_keys))
        offset = 0
        for split_name, count in target_counts.items():
            for group_key in ordered_group_keys[offset : offset + count]:
                group_to_split[group_key] = split_name
            offset += count

    frame["split"] = frame["_split_group_key"].map(group_to_split)
    if frame["split"].isna().any():
        missing = sorted(frame.loc[frame["split"].isna(), "_split_group_key"].astype(str).unique())
        raise ValueError(f"Grouped AV split failed for group keys: {missing[:5]}")

    group_frame["split"] = group_frame["_split_group_key"].map(group_to_split)
    metadata = {
        "split_group_columns": sorted(group_columns_seen),
        "split_group_count": int(group_frame["_split_group_key"].nunique()),
        "split_group_counts": {
            str(key): int(value) for key, value in group_frame["split"].value_counts().sort_index().items()
        },
        "split_city_group_counts": {
            str(city): {
                str(key): int(value) for key, value in city_frame["split"].value_counts().sort_index().items()
            }
            for city, city_frame in group_frame.groupby("_split_city", sort=True)
        },
        "split_weights": {key: float(value) for key, value in SPLIT_WEIGHTS.items()},
    }
    split_map = {
        str(row["scenario_id"]): str(row["split"])
        for _, row in frame.drop_duplicates("scenario_id").iterrows()
    }
    return split_map, metadata


def _split_map_for_strategy(
    *,
    scenario_frame: pd.DataFrame,
    split_strategy: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    scenario_ids = scenario_frame["scenario_id"].astype(str).unique()
    if split_strategy == "all_test":
        return {str(scenario_id): "test" for scenario_id in scenario_ids}, {}
    if split_strategy == "balanced":
        return assign_balanced_splits(scenario_ids), {}
    if split_strategy == "hash":
        return {str(scenario_id): assign_split(str(scenario_id)) for scenario_id in scenario_ids}, {}
    if split_strategy == GROUPED_ARCHIVE_DB_CITY_SPLIT:
        return assign_grouped_archive_db_city_splits(scenario_frame)
    raise _split_strategy_error(split_strategy)


def _speed_bin(speed: float) -> str:
    if speed < 5.0:
        return "0_5"
    if speed < 10.0:
        return "5_10"
    if speed < 15.0:
        return "10_15"
    if speed < 20.0:
        return "15_20"
    return "20_plus"


def _neighbor_count_bin(count: int) -> str:
    if count <= 1:
        return "0_1"
    if count <= 3:
        return "2_3"
    if count <= 5:
        return "4_5"
    return "6_plus"


def _build_feature_rows_for_scenario(group: pd.DataFrame, *, history_steps: int = 10) -> list[dict[str, Any]]:
    group = group.sort_values("step_index").reset_index(drop=True)
    max_horizon = max(HORIZON_LABELS.values())
    rows: list[dict[str, Any]] = []
    for current_idx in range(history_steps, len(group) - max_horizon):
        history = group.iloc[current_idx - history_steps : current_idx + 1].reset_index(drop=True)
        current = history.iloc[-1]
        feature_row: dict[str, Any] = {
            "scenario_id": str(current["scenario_id"]),
            "shard_id": str(current["shard_id"]),
            "record_index": int(current["record_index"]),
            "step_index": int(current["step_index"]),
            "ego_track_id": int(current["ego_track_id"]),
            "neighbor_count": int(current["neighbor_count"]),
            "object_mix_bin": str(current["object_mix_bin"]),
            "speed_bin": _speed_bin(float(current["ego_speed_mps"])),
            "neighbor_count_bin": _neighbor_count_bin(int(current["neighbor_count"])),
        }
        valid_values: list[float] = []
        for lag, (_, hist_row) in enumerate(history.iloc[::-1].iterrows()):
            feature_row[f"ego_speed_mps_lag{lag}"] = float(hist_row["ego_speed_mps"])
            feature_row[f"ego_heading_rad_lag{lag}"] = float(hist_row["ego_heading_rad"])
            feature_row[f"lead_gap_m_lag{lag}"] = float(hist_row["min_gap_m"])
            feature_row[f"lead_rel_speed_mps_lag{lag}"] = float(hist_row["lead_rel_speed_mps"])
            feature_row[f"neighbor_count_lag{lag}"] = int(hist_row["neighbor_count"])
            for slot in range(8):
                prefix = f"neighbor_slot_{slot}"
                gap = hist_row.get(f"{prefix}_rel_longitudinal_gap_m")
                lat = hist_row.get(f"{prefix}_rel_lateral_offset_m")
                speed = hist_row.get(f"{prefix}_speed_mps")
                valid = int(bool(hist_row.get(f"{prefix}_valid", False)))
                feature_row[f"{prefix}_gap_m_lag{lag}"] = np.nan if gap is None else float(gap)
                feature_row[f"{prefix}_lat_m_lag{lag}"] = np.nan if lat is None else float(lat)
                feature_row[f"{prefix}_speed_mps_lag{lag}"] = np.nan if speed is None else float(speed)
                feature_row[f"{prefix}_valid_lag{lag}"] = valid
                valid_values.append(float(valid))
        feature_row["reliability_proxy"] = float(np.mean(valid_values)) if valid_values else 1.0

        for horizon_label, horizon_steps in HORIZON_LABELS.items():
            future = group.iloc[current_idx + horizon_steps]
            feature_row[f"target_ego_speed_mps__{horizon_label}"] = float(future["ego_speed_mps"])
            feature_row[f"target_relative_gap_m__{horizon_label}"] = float(future["min_gap_m"])
        rows.append(feature_row)
    return rows


def build_feature_tables(
    *,
    replay_windows_path: str | Path,
    out_dir: str | Path,
    split_strategy: str = "hash",
) -> dict[str, Any]:
    replay_windows_path = Path(replay_windows_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    replay_metadata = pq.ParquetFile(replay_windows_path).metadata

    if replay_metadata.num_rows <= IN_MEMORY_FEATURE_ROW_LIMIT:
        replay_df = pd.read_parquet(replay_windows_path)

        feature_rows: list[dict[str, Any]] = []
        for _, group in replay_df.groupby("scenario_id", sort=True):
            feature_rows.extend(_build_feature_rows_for_scenario(group))

        step_features = pd.DataFrame(feature_rows)
        if step_features.empty:
            raise ValueError("No step-wise feature rows were generated from replay windows.")
        scenario_frame, _ = _read_split_metadata(replay_windows_path)
        split_map, split_metadata = _split_map_for_strategy(
            scenario_frame=scenario_frame,
            split_strategy=split_strategy,
        )
        step_features["split"] = step_features["scenario_id"].astype(str).map(split_map)
        if step_features["split"].isna().any():
            missing = sorted(
                step_features.loc[step_features["split"].isna(), "scenario_id"].astype(str).unique()
            )
            raise ValueError(f"Feature split map is missing scenario IDs: {missing[:5]}")
        step_features_path = out_path / "step_features.parquet"
        step_features.to_parquet(step_features_path, index=False)

        anchor_features = step_features[
            step_features["step_index"] == int(step_features["step_index"].min())
        ].copy()
        anchor_features_path = out_path / "anchor_features.parquet"
        anchor_features.to_parquet(anchor_features_path, index=False)

        splits_dir = out_path / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_df in anchor_features.groupby("split", sort=False):
            split_df.to_parquet(splits_dir / f"{split_name}.parquet", index=False)

        report = {
            "row_count": int(len(step_features)),
            "anchor_row_count": int(len(anchor_features)),
            "scenario_count": int(anchor_features["scenario_id"].nunique()),
            "split_strategy": split_strategy,
            "split_counts": {
                str(key): int(value)
                for key, value in anchor_features["split"].value_counts().sort_index().items()
            },
            **split_metadata,
            "artifacts": {
                "step_features": str(step_features_path),
                "anchor_features": str(anchor_features_path),
                "splits_dir": str(splits_dir),
            },
        }
        (out_path / "feature_table_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    _emit_progress(
        {
            "event": "av_feature_stream_start",
            "replay_rows": int(replay_metadata.num_rows),
            "split_strategy": split_strategy,
        }
    )
    replay_windows_path.parent / "scenario_index.parquet"
    scenario_frame, _ = _read_split_metadata(replay_windows_path)
    split_map, split_metadata = _split_map_for_strategy(
        scenario_frame=scenario_frame,
        split_strategy=split_strategy,
    )

    step_features_path = out_path / "step_features.parquet"
    anchor_features_path = out_path / "anchor_features.parquet"
    tmp_step_path = step_features_path.with_name(f"{step_features_path.name}.tmp")
    tmp_step_path.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    feature_schema: pa.Schema | None = None
    feature_buffer: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    row_count = 0
    next_progress_row_count = 1_000_000

    def flush_feature_buffer() -> None:
        nonlocal feature_buffer, writer, feature_schema, row_count, next_progress_row_count
        if not feature_buffer:
            return
        frame = pd.DataFrame(feature_buffer)
        if feature_schema is None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            feature_schema = table.schema
            writer = pq.ParquetWriter(str(tmp_step_path), feature_schema)
        else:
            for column in feature_schema.names:
                if column not in frame.columns:
                    frame[column] = pd.NA
            frame = frame[feature_schema.names]
            table = pa.Table.from_pandas(frame, schema=feature_schema, preserve_index=False)
        assert writer is not None
        writer.write_table(table)
        row_count += int(len(frame))
        feature_buffer = []
        if row_count >= next_progress_row_count:
            _emit_progress(
                {
                    "event": "av_feature_stream_progress",
                    "feature_rows": int(row_count),
                    "anchor_rows": int(len(anchor_rows)),
                }
            )
            while row_count >= next_progress_row_count:
                next_progress_row_count += 1_000_000

    def process_scenario_group(group: pd.DataFrame) -> None:
        rows = _build_feature_rows_for_scenario(group)
        if not rows:
            return
        for row in rows:
            row["split"] = split_map[str(row["scenario_id"])]
        anchor_rows.append(rows[0])
        feature_buffer.extend(rows)
        if len(feature_buffer) >= FEATURE_WRITE_BATCH_ROWS:
            flush_feature_buffer()

    pending = pd.DataFrame()
    try:
        for batch in pq.ParquetFile(replay_windows_path).iter_batches(batch_size=FEATURE_BUILD_BATCH_ROWS):
            batch_frame = batch.to_pandas()
            if not pending.empty:
                batch_frame = pd.concat([pending, batch_frame], ignore_index=True)
            if batch_frame.empty:
                pending = batch_frame
                continue
            scenario_ids = batch_frame["scenario_id"].astype(str)
            last_scenario_id = str(scenario_ids.iloc[-1])
            complete_frame = batch_frame[scenario_ids != last_scenario_id]
            pending = batch_frame[scenario_ids == last_scenario_id].copy()
            for _, group in complete_frame.groupby("scenario_id", sort=False):
                process_scenario_group(group)

        if not pending.empty:
            for _, group in pending.groupby("scenario_id", sort=False):
                process_scenario_group(group)
        flush_feature_buffer()
    finally:
        if writer is not None:
            writer.close()

    if row_count <= 0 or not anchor_rows:
        tmp_step_path.unlink(missing_ok=True)
        raise ValueError("No step-wise feature rows were generated from replay windows.")
    tmp_step_path.replace(step_features_path)

    anchor_features = pd.DataFrame(anchor_rows)
    anchor_features.to_parquet(anchor_features_path, index=False)
    splits_dir = out_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in anchor_features.groupby("split", sort=False):
        split_df.to_parquet(splits_dir / f"{split_name}.parquet", index=False)

    report = {
        "row_count": int(row_count),
        "anchor_row_count": int(len(anchor_features)),
        "scenario_count": int(anchor_features["scenario_id"].nunique()),
        "split_strategy": split_strategy,
        "streaming": True,
        "split_counts": {
            str(key): int(value)
            for key, value in anchor_features["split"].value_counts().sort_index().items()
        },
        **split_metadata,
        "artifacts": {
            "step_features": str(step_features_path),
            "anchor_features": str(anchor_features_path),
            "splits_dir": str(splits_dir),
        },
    }
    (out_path / "feature_table_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _emit_progress(
        {
            "event": "av_feature_stream_done",
            "feature_rows": int(row_count),
            "anchor_rows": int(len(anchor_features)),
            "scenario_count": int(anchor_features["scenario_id"].nunique()),
        }
    )
    return report


def _raw_feature_columns(df: pd.DataFrame) -> list[str]:
    target_cols = [col for col in df.columns if col.startswith("target_")]
    return [col for col in df.columns if col not in META_COLUMNS and col not in target_cols]


def prepare_feature_matrix(
    df: pd.DataFrame, *, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    bool_cols = [col for col in work.columns if work[col].dtype == bool]
    for col in bool_cols:
        work[col] = work[col].astype(int)
    matrix = pd.get_dummies(work, dummy_na=True)
    if feature_columns is None:
        feature_columns = sorted(matrix.columns.tolist())
    matrix = matrix.reindex(columns=feature_columns, fill_value=0)
    return matrix, feature_columns


def _feature_stats(matrix: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    mean = {str(col): float(matrix[col].mean()) for col in matrix.columns}
    std = {str(col): float(max(matrix[col].std(ddof=0), 1e-6)) for col in matrix.columns}
    return mean, std


def _estimate_shift_scores(
    matrix: pd.DataFrame, feature_mean: Mapping[str, float], feature_std: Mapping[str, float]
) -> np.ndarray:
    z_values = []
    for col in matrix.columns:
        mean = float(feature_mean[col])
        std = float(feature_std[col])
        z_values.append(np.abs((matrix[col].astype(float).to_numpy() - mean) / std))
    if not z_values:
        return np.zeros(len(matrix), dtype=float)
    aggregate = np.mean(np.vstack(z_values), axis=0)
    return np.clip(aggregate / 3.0, 0.0, 1.0)


def widening_factor(reliability: np.ndarray, shift_score: np.ndarray) -> np.ndarray:
    reliability = np.nan_to_num(np.asarray(reliability, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    shift_score = np.nan_to_num(np.asarray(shift_score, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    factor = 1.0 + 0.7 * (1.0 - np.clip(reliability, 0.0, 1.0)) + 0.5 * np.clip(shift_score, 0.0, 1.0)
    return np.clip(factor, 1.0, 3.0)


def widen_bounds(
    lower: np.ndarray, upper: np.ndarray, reliability: np.ndarray, shift_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    factor = widening_factor(reliability, shift_score)
    widened_half = half_width * factor
    return center - widened_half, center + widened_half, factor


def _target_tasks() -> list[tuple[str, str]]:
    return [(target, horizon_label) for target in TARGETS for horizon_label in HORIZON_LABELS]


def _coverage_by_groups(
    frame: pd.DataFrame,
    *,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    group_col: str,
    target: str,
    horizon_label: str,
    interval_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_value, group_df in frame.groupby(group_col, dropna=False):
        idx = group_df.index.to_numpy()
        if idx.size == 0:
            continue
        rows.append(
            {
                "group_col": group_col,
                "group_value": str(group_value),
                "target": target,
                "horizon_label": horizon_label,
                "interval_kind": interval_kind,
                "n": int(idx.size),
                "coverage": float(np.mean((y_true[idx] >= lower[idx]) & (y_true[idx] <= upper[idx]))),
                "mean_width": float(np.mean(upper[idx] - lower[idx])),
            }
        )
    return rows


def _default_model_params() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    median_params = {
        "backend": "lightgbm",
        "n_estimators": 240,
        "learning_rate": 0.04,
        "max_depth": 10,
        "num_leaves": 63,
        "min_child_samples": 5,
        "verbosity": -1,
    }
    lower_params = dict(median_params, objective="quantile", alpha=0.1)
    upper_params = dict(median_params, objective="quantile", alpha=0.9)
    return median_params, lower_params, upper_params


def train_dry_run_models(
    *,
    anchor_features_path: str | Path,
    step_features_path: str | Path,
    models_dir: str | Path,
    uncertainty_dir: str | Path,
    reports_dir: str | Path,
    artifact_prefix: str = "waymo_av",
) -> dict[str, Any]:
    anchor_df = pd.read_parquet(anchor_features_path).reset_index(drop=True)
    step_row_count = int(pq.ParquetFile(step_features_path).metadata.num_rows)
    if anchor_df.empty:
        raise ValueError("Anchor feature table is empty.")

    raw_feature_cols = _raw_feature_columns(anchor_df)
    train_df = anchor_df[anchor_df["split"] == "train"].reset_index(drop=True)
    cal_df = anchor_df[anchor_df["split"] == "calibration"].reset_index(drop=True)
    val_df = anchor_df[anchor_df["split"] == "val"].reset_index(drop=True)
    test_df = anchor_df[anchor_df["split"] == "test"].reset_index(drop=True)
    if min(len(train_df), len(cal_df), len(val_df), len(test_df)) == 0:
        raise ValueError("At least one dry-run split is empty; cannot train/calibrate/evaluate.")

    X_train, feature_columns = prepare_feature_matrix(train_df[raw_feature_cols])
    X_cal, _ = prepare_feature_matrix(cal_df[raw_feature_cols], feature_columns=feature_columns)
    X_val, _ = prepare_feature_matrix(val_df[raw_feature_cols], feature_columns=feature_columns)
    X_test, _ = prepare_feature_matrix(test_df[raw_feature_cols], feature_columns=feature_columns)
    feature_mean, feature_std = _feature_stats(X_train)
    models_path = Path(models_dir)
    uncertainty_path = Path(uncertainty_dir)
    reports_path = Path(reports_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    uncertainty_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    shift_cfg = default_shift_aware_config(publication_dir=str(reports_path / "shift_aware"))
    shift_cfg_path = reports_path / "shift_aware_config.json"
    shift_cfg_path.write_text(json.dumps(shift_cfg.to_dict(), indent=2), encoding="utf-8")

    median_params, lower_params, upper_params = _default_model_params()
    summary_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    task_registry: dict[str, str] = {}

    for target, horizon_label in _target_tasks():
        _emit_progress({"event": "av_training_task_start", "target": target, "horizon_label": horizon_label})
        target_col = f"target_{target}__{horizon_label}"
        y_train = train_df[target_col].to_numpy(dtype=float)
        y_cal = cal_df[target_col].to_numpy(dtype=float)
        y_val = val_df[target_col].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)

        _, median_model = train_gbm(X_train, y_train, params=median_params)
        _, lower_model = train_gbm(X_train, y_train, params=lower_params)
        _, upper_model = train_gbm(X_train, y_train, params=upper_params)

        q_lo_cal = predict_gbm(lower_model, X_cal)
        q_hi_cal = predict_gbm(upper_model, X_cal)
        ci = ConformalInterval(ConformalConfig(alpha=0.10, method="cqr", horizon_wise=True, rolling=False))
        ci.fit_calibration_cqr(y_cal.reshape(-1, 1), q_lo_cal.reshape(-1, 1), q_hi_cal.reshape(-1, 1))
        qhat = float(ci.q_h[0] if ci.q_h is not None else ci.q_global or 0.0)

        for split_name, split_df, X_split, y_split in (
            ("val", val_df, X_val, y_val),
            ("test", test_df, X_test, y_test),
        ):
            q_lo = predict_gbm(lower_model, X_split)
            q_hi = predict_gbm(upper_model, X_split)
            lower, upper = ci.predict_interval_cqr(q_lo.reshape(-1, 1), q_hi.reshape(-1, 1))
            lower = lower.reshape(-1)
            upper = upper.reshape(-1)
            reliability = split_df["reliability_proxy"].to_numpy(dtype=float)
            shift_score = _estimate_shift_scores(X_split, feature_mean, feature_std)
            widened_lower, widened_upper, factor = widen_bounds(lower, upper, reliability, shift_score)

            summary_rows.append(
                {
                    "target": target,
                    "horizon_label": horizon_label,
                    "split": split_name,
                    "base_coverage": float(np.mean((y_split >= lower) & (y_split <= upper))),
                    "base_mean_width": float(np.mean(upper - lower)),
                    "widened_coverage": float(
                        np.mean((y_split >= widened_lower) & (y_split <= widened_upper))
                    ),
                    "widened_mean_width": float(np.mean(widened_upper - widened_lower)),
                    "mean_widening_factor": float(np.mean(factor)),
                }
            )
            for group_col in ("shard_id", "speed_bin", "neighbor_count_bin", "object_mix_bin"):
                subgroup_rows.extend(
                    _coverage_by_groups(
                        split_df,
                        y_true=y_split,
                        lower=lower,
                        upper=upper,
                        group_col=group_col,
                        target=target,
                        horizon_label=horizon_label,
                        interval_kind=f"base_{split_name}",
                    )
                )
                subgroup_rows.extend(
                    _coverage_by_groups(
                        split_df,
                        y_true=y_split,
                        lower=widened_lower,
                        upper=widened_upper,
                        group_col=group_col,
                        target=target,
                        horizon_label=horizon_label,
                        interval_kind=f"widened_{split_name}",
                    )
                )

        base_name = f"{target}_{horizon_label}"
        bundle = {
            "target": target,
            "horizon_label": horizon_label,
            "feature_columns": feature_columns,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "median_model": median_model,
            "lower_model": lower_model,
            "upper_model": upper_model,
            "qhat": qhat,
        }
        model_path = models_path / f"{artifact_prefix}_{base_name}_bundle.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(bundle, handle)
        save_conformal(
            uncertainty_path / f"{artifact_prefix}_{base_name}_conformal.json",
            ci,
            meta={"target": target, "horizon_label": horizon_label, "qhat": qhat},
        )
        task_registry[base_name] = str(model_path)
        _emit_progress({"event": "av_training_task_done", "target": target, "horizon_label": horizon_label})

    pd.DataFrame(summary_rows).to_csv(reports_path / "training_summary.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(reports_path / "subgroup_coverage.csv", index=False)
    train_stats = {
        "feature_columns": feature_columns,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "runtime_step_feature_rows": step_row_count,
        "artifact_registry": task_registry,
        "artifact_prefix": artifact_prefix,
        "shift_aware_config": shift_cfg.to_dict(),
    }
    (reports_path / "feature_stats.json").write_text(json.dumps(train_stats, indent=2), encoding="utf-8")
    return {
        "training_summary_csv": str(reports_path / "training_summary.csv"),
        "subgroup_coverage_csv": str(reports_path / "subgroup_coverage.csv"),
        "feature_stats_json": str(reports_path / "feature_stats.json"),
        "shift_aware_config_json": str(shift_cfg_path),
        "artifact_registry": task_registry,
    }


def load_model_bundle(path: str | Path) -> ModelBundle:
    payload = load_pickle_artifact(path)
    return ModelBundle(
        target=str(payload["target"]),
        horizon_label=str(payload["horizon_label"]),
        feature_columns=list(payload["feature_columns"]),
        median_model=payload["median_model"],
        lower_model=payload["lower_model"],
        upper_model=payload["upper_model"],
        qhat=float(payload["qhat"]),
        feature_mean=dict(payload["feature_mean"]),
        feature_std=dict(payload["feature_std"]),
    )


def predict_interval_from_bundle(
    bundle: ModelBundle, feature_frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, _ = prepare_feature_matrix(feature_frame.copy(), feature_columns=bundle.feature_columns)
    lower = predict_gbm(bundle.lower_model, X) - bundle.qhat
    upper = predict_gbm(bundle.upper_model, X) + bundle.qhat
    center = predict_gbm(bundle.median_model, X)
    return center, lower, upper


def estimate_shift_score(bundle: ModelBundle, feature_frame: pd.DataFrame) -> np.ndarray:
    X, _ = prepare_feature_matrix(feature_frame.copy(), feature_columns=bundle.feature_columns)
    return _estimate_shift_scores(X, bundle.feature_mean, bundle.feature_std)
