"""Build ORIUS replay windows directly from nuPlan SQLite archives.

The downstream ORIUS AV training/runtime path consumes a source-neutral
``replay_windows.parquet`` table. This module converts nuPlan DB logs into that
contract while leaving the older Waymo parser available as a legacy source.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from orius.av_waymo.dataset import MAX_NEIGHBORS, NEIGHBOR_RADIUS_M, TOTAL_SCENARIO_STEPS
from orius.av_waymo.replay import _project_to_ego_frame, _slot_fields, compute_state_safety_metrics

DEFAULT_EGO_LENGTH_M = 4.8
DEFAULT_EGO_WIDTH_M = 2.0
DEFAULT_EGO_HEIGHT_M = 1.7
DEFAULT_SCENARIO_STRIDE = TOTAL_SCENARIO_STEPS
DEFAULT_TRAIN_GLOB = "nuplan-v*.zip"
SOURCE_MANIFEST_FILENAME = "nuplan_source_manifest.json"


@dataclass(slots=True)
class NuPlanSurfaceConfig:
    """Configuration for a bounded nuPlan replay-surface build."""

    train_zips: tuple[Path, ...]
    out_dir: Path
    archive_role: str = "train"
    maps_zip: Path | None = None
    temp_dir: Path | None = None
    max_dbs: int | None = None
    max_scenarios: int | None = None
    max_dbs_per_archive: int | None = None
    max_scenarios_per_archive: int | None = None
    scenario_stride: int = DEFAULT_SCENARIO_STRIDE
    batch_size: int = 8_192


@dataclass(frozen=True, slots=True)
class NuPlanArchive:
    path: Path
    archive_id: str
    sha256: str
    size_bytes: int
    entry_count: int
    db_count: int
    uncompressed_db_bytes: int
    first_db_entries: tuple[str, ...]


class _ReplayParquetWriter:
    def __init__(self, path: Path, *, schema: pa.Schema, batch_size: int) -> None:
        self.path = Path(path)
        self.temp_path = self.path.with_name(f"{self.path.name}.tmp")
        self.schema = schema
        self.batch_size = int(batch_size)
        self.rows: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.temp_path.exists():
            self.temp_path.unlink()
        self._sink = pa.OSFile(str(self.temp_path), "wb")
        self._writer = pq.ParquetWriter(self._sink, self.schema)
        self._closed = False

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self._writer.write_table(table)
        self.rows.clear()

    def close(self, *, finalize: bool = True) -> None:
        if self._closed:
            return
        self.flush()
        self._writer.close()
        self._sink.close()
        self._closed = True
        if finalize:
            pq.ParquetFile(self.temp_path)
            self.temp_path.replace(self.path)
        else:
            self.temp_path.unlink(missing_ok=True)


def _replay_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("scenario_id", pa.string()),
        pa.field("shard_id", pa.string()),
        pa.field("source_dataset", pa.string()),
        pa.field("record_index", pa.int64()),
        pa.field("step_index", pa.int64()),
        pa.field("timestamp_us", pa.int64()),
        pa.field("ts_utc", pa.string()),
        pa.field("ego_track_id", pa.int64()),
        pa.field("ego_x_m", pa.float64()),
        pa.field("ego_y_m", pa.float64()),
        pa.field("ego_speed_mps", pa.float64()),
        pa.field("ego_velocity_x_mps", pa.float64()),
        pa.field("ego_velocity_y_mps", pa.float64()),
        pa.field("ego_heading_rad", pa.float64()),
        pa.field("ego_length_m", pa.float64()),
        pa.field("ego_width_m", pa.float64()),
        pa.field("ego_valid", pa.bool_()),
        pa.field("speed_limit_mps", pa.float64()),
        pa.field("neighbor_count", pa.int64()),
        pa.field("object_mix_bin", pa.string()),
    ]
    for slot in range(MAX_NEIGHBORS):
        base = f"neighbor_slot_{slot}"
        fields.extend(
            [
                pa.field(f"{base}_track_id", pa.int64()),
                pa.field(f"{base}_x_m", pa.float64()),
                pa.field(f"{base}_y_m", pa.float64()),
                pa.field(f"{base}_speed_mps", pa.float64()),
                pa.field(f"{base}_length_m", pa.float64()),
                pa.field(f"{base}_width_m", pa.float64()),
                pa.field(f"{base}_rel_x_m", pa.float64()),
                pa.field(f"{base}_rel_y_m", pa.float64()),
                pa.field(f"{base}_rel_longitudinal_gap_m", pa.float64()),
                pa.field(f"{base}_rel_lateral_offset_m", pa.float64()),
                pa.field(f"{base}_valid", pa.bool_()),
            ]
        )
    fields.extend(
        [
            pa.field("lead_track_id", pa.int64()),
            pa.field("min_gap_m", pa.float64()),
            pa.field("lead_speed_mps", pa.float64()),
            pa.field("lead_rel_speed_mps", pa.float64()),
            pa.field("ttc_s", pa.float64()),
            pa.field("overlap", pa.bool_()),
            pa.field("true_constraint_violated", pa.bool_()),
            pa.field("true_margin", pa.float64()),
        ]
    )
    return pa.schema(fields)


def _token_hex(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _stable_token_int(value: Any) -> int:
    token = _token_hex(value)
    if not token:
        return 0
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _timestamp_to_iso(timestamp_us: int) -> str:
    return datetime.fromtimestamp(int(timestamp_us) / 1_000_000.0, tz=UTC).isoformat().replace("+00:00", "Z")


def _speed(vx: Any, vy: Any) -> float:
    vx_f = 0.0 if vx is None else float(vx)
    vy_f = 0.0 if vy is None else float(vy)
    return float(math.hypot(vx_f, vy_f))


def _clip_speed_limit(speeds: Iterable[float]) -> float:
    values = [float(value) for value in speeds if math.isfinite(float(value))]
    if not values:
        return 15.0
    return float(np.clip(max(values) + 2.0, 5.0, 40.0))


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _emit_progress(event: Mapping[str, Any]) -> None:
    pass


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _archive_id(path: Path, sha256: str) -> str:
    safe_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in Path(path).stem)
    return f"{safe_stem}-{sha256[:12]}"


def _nuplan_source_dataset(archive_path: str | Path, db_entry_name: str, location: Any = None) -> str:
    """Return a stable nuPlan source label from archive/member/log metadata."""

    archive_text = " ".join(
        str(part).lower() for part in (archive_path, db_entry_name) if part not in (None, "")
    )
    for needle, label in (
        ("pittsburgh", "nuplan_pittsburgh"),
        ("boston", "nuplan_boston"),
        ("singapore", "nuplan_singapore"),
        ("las_vegas", "nuplan_las_vegas"),
        ("las-vegas", "nuplan_las_vegas"),
        ("vegas", "nuplan_las_vegas"),
    ):
        if needle in archive_text:
            return label

    text = " ".join(str(part).lower() for part in (location or "",) if part not in (None, ""))
    if "pittsburgh" in text:
        return "nuplan_pittsburgh"
    if "singapore" in text or "sg-one-north" in text:
        return "nuplan_singapore"
    if "boston" in text:
        return "nuplan_boston"
    if "las_vegas" in text or "las-vegas" in text or "vegas" in text:
        return "nuplan_las_vegas"
    return "nuplan_unknown"


def _summarize_source_datasets(source_datasets: Iterable[str]) -> str:
    labels = sorted({str(label) for label in source_datasets if str(label)})
    if not labels:
        return "nuplan_unknown"
    if len(labels) == 1:
        return labels[0]
    return "nuplan_multi_city"


def _is_incomplete_download(path: Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(".crdownload") or name.startswith("unconfirmed ")


def _archive_record(archive: NuPlanArchive) -> dict[str, Any]:
    return {
        "path": str(archive.path),
        "archive_id": archive.archive_id,
        "sha256": archive.sha256,
        "size_bytes": int(archive.size_bytes),
        "entry_count": int(archive.entry_count),
        "db_count": int(archive.db_count),
        "uncompressed_db_bytes": int(archive.uncompressed_db_bytes),
        "first_db_entries": list(archive.first_db_entries),
    }


def _candidate_train_paths(
    *,
    train_zips: Sequence[str | Path] | None = None,
    train_dirs: Sequence[str | Path] | None = None,
    train_glob: str = DEFAULT_TRAIN_GLOB,
) -> list[Path]:
    candidates: list[Path] = []
    for path in train_zips or ():
        candidates.append(Path(path))
    for directory in train_dirs or ():
        root = Path(directory)
        if not root.exists():
            candidates.append(root / train_glob)
            continue
        candidates.extend(sorted(path for path in root.glob(train_glob) if path.is_file()))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def resolve_nuplan_train_archives(
    *,
    train_zips: Sequence[str | Path] | None = None,
    train_dirs: Sequence[str | Path] | None = None,
    train_glob: str = DEFAULT_TRAIN_GLOB,
    skip_incomplete: bool = True,
) -> tuple[list[NuPlanArchive], list[dict[str, Any]]]:
    """Validate candidate nuPlan train zips and return DB-bearing archives."""
    archives: list[NuPlanArchive] = []
    skipped: list[dict[str, Any]] = []
    for candidate in _candidate_train_paths(
        train_zips=train_zips, train_dirs=train_dirs, train_glob=train_glob
    ):
        path = Path(candidate)
        if _is_incomplete_download(path):
            skipped.append({"path": str(path), "reason": "incomplete_download"})
            if skip_incomplete:
                continue
            raise ValueError(f"Incomplete nuPlan download cannot be used: {path}")
        if not path.exists():
            skipped.append({"path": str(path), "reason": "missing"})
            if skip_incomplete:
                continue
            raise FileNotFoundError(f"Missing nuPlan archive: {path}")
        if not path.is_file():
            skipped.append({"path": str(path), "reason": "not_file"})
            continue
        if path.suffix.lower() != ".zip":
            skipped.append({"path": str(path), "reason": "not_zip"})
            continue
        if "map" in path.name.lower():
            skipped.append({"path": str(path), "reason": "maps_archive"})
            continue
        if not zipfile.is_zipfile(path):
            skipped.append({"path": str(path), "reason": "invalid_zip"})
            if skip_incomplete:
                continue
            raise ValueError(f"Invalid nuPlan zip archive: {path}")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            db_entries = [info for info in infos if info.filename.endswith(".db")]
            if not db_entries:
                skipped.append({"path": str(path), "reason": "no_db_entries"})
                continue
            digest = _sha256_file(path)
            archives.append(
                NuPlanArchive(
                    path=path,
                    archive_id=_archive_id(path, digest),
                    sha256=digest,
                    size_bytes=int(path.stat().st_size),
                    entry_count=int(len(infos)),
                    db_count=int(len(db_entries)),
                    uncompressed_db_bytes=int(sum(info.file_size for info in db_entries)),
                    first_db_entries=tuple(info.filename for info in db_entries[:20]),
                )
            )
    if not archives:
        skipped_reasons = ", ".join(f"{row['path']}:{row['reason']}" for row in skipped) or "no candidates"
        raise FileNotFoundError(
            f"No completed nuPlan train archives with DB entries were found ({skipped_reasons})."
        )
    return archives, skipped


def _zip_inventory(path: Path | None, *, db_only: bool = False) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        entries = [info for info in infos if not db_only or info.filename.endswith(".db")]
        return {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size),
            "entry_count": int(len(infos)),
            "selected_entry_count": int(len(entries)),
            "uncompressed_bytes": int(sum(info.file_size for info in entries)),
            "first_entries": [info.filename for info in entries[:20]],
        }


def inspect_nuplan_archives(
    train_zip: str | Path | Sequence[str | Path] | None = None,
    maps_zip: str | Path | None = None,
    *,
    train_zips: Sequence[str | Path] | None = None,
    train_dirs: Sequence[str | Path] | None = None,
    train_glob: str = DEFAULT_TRAIN_GLOB,
    skip_incomplete: bool = True,
) -> dict[str, Any]:
    """Return a lightweight manifest for local nuPlan zip archives."""
    explicit_train_zips: list[str | Path] = []
    if train_zip is not None:
        if isinstance(train_zip, str | Path):
            explicit_train_zips.append(train_zip)
        else:
            explicit_train_zips.extend(train_zip)
    explicit_train_zips.extend(train_zips or [])
    archives, skipped = resolve_nuplan_train_archives(
        train_zips=explicit_train_zips,
        train_dirs=train_dirs,
        train_glob=train_glob,
        skip_incomplete=skip_incomplete,
    )
    maps_path = Path(maps_zip) if maps_zip is not None else None
    if maps_path is not None and not maps_path.exists():
        raise FileNotFoundError(f"Missing nuPlan maps zip: {maps_path}")
    return {
        "train": _archive_record(archives[0]),
        "train_archives": [_archive_record(archive) for archive in archives],
        "skipped_train_archives": skipped,
        "total_db_count": int(sum(archive.db_count for archive in archives)),
        "maps": _zip_inventory(maps_path, db_only=False),
    }


def _fetch_lidar_rows(db_path: Path) -> list[sqlite3.Row]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return list(
            con.execute(
                """
                SELECT
                    lp.token AS lidar_pc_token,
                    lp.scene_token AS scene_token,
                    lp.timestamp AS timestamp_us,
                    ep.x AS ego_x,
                    ep.y AS ego_y,
                    ep.qw AS qw,
                    ep.qx AS qx,
                    ep.qy AS qy,
                    ep.qz AS qz,
                    ep.vx AS vx,
                    ep.vy AS vy,
                    ep.vz AS vz,
                    scene.name AS scene_name,
                    log.location AS location,
                    log.map_version AS map_version
                FROM lidar_pc AS lp
                JOIN ego_pose AS ep ON lp.ego_pose_token = ep.token
                LEFT JOIN scene ON lp.scene_token = scene.token
                LEFT JOIN log ON ep.log_token = log.token
                ORDER BY scene.name, lp.timestamp
                """
            )
        )
    finally:
        con.close()


def _fetch_boxes_for_tokens(db_path: Path, lidar_tokens: list[Any]) -> dict[str, list[sqlite3.Row]]:
    if not lidar_tokens:
        return {}
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in lidar_tokens)
        rows = list(
            con.execute(
                f"""
                SELECT
                    b.lidar_pc_token AS lidar_pc_token,
                    b.track_token AS track_token,
                    b.x AS x,
                    b.y AS y,
                    b.width AS width,
                    b.length AS length,
                    b.height AS height,
                    b.vx AS vx,
                    b.vy AS vy,
                    b.vz AS vz,
                    b.yaw AS yaw,
                    b.confidence AS confidence,
                    c.name AS category_name
                FROM lidar_box AS b
                LEFT JOIN track AS t ON b.track_token = t.token
                LEFT JOIN category AS c ON t.category_token = c.token
                WHERE b.lidar_pc_token IN ({placeholders})
                """,
                lidar_tokens,
            )
        )
    finally:
        con.close()
    by_lidar: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_lidar.setdefault(_token_hex(row["lidar_pc_token"]), []).append(row)
    return by_lidar


def _scene_groups(lidar_rows: list[sqlite3.Row]) -> list[list[sqlite3.Row]]:
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in lidar_rows:
        key = _token_hex(row["scene_token"]) or str(row["scene_name"] or "scene")
        grouped.setdefault(key, []).append(row)
    return [sorted(rows, key=lambda item: int(item["timestamp_us"])) for _, rows in sorted(grouped.items())]


def _candidate_windows(
    scene_rows: list[sqlite3.Row], *, stride: int
) -> Iterable[tuple[int, list[sqlite3.Row]]]:
    if len(scene_rows) < TOTAL_SCENARIO_STEPS:
        return
    step = max(1, int(stride))
    for start in range(0, len(scene_rows) - TOTAL_SCENARIO_STEPS + 1, step):
        yield start, scene_rows[start : start + TOTAL_SCENARIO_STEPS]


def _select_neighbors(
    row: sqlite3.Row, boxes: list[sqlite3.Row], *, ego_heading: float
) -> list[dict[str, Any]]:
    ego_x = float(row["ego_x"])
    ego_y = float(row["ego_y"])
    selected: list[dict[str, Any]] = []
    for box in boxes:
        actor_x = float(box["x"])
        actor_y = float(box["y"])
        rel_x = actor_x - ego_x
        rel_y = actor_y - ego_y
        distance = math.hypot(rel_x, rel_y)
        if distance > NEIGHBOR_RADIUS_M:
            continue
        longitudinal, lateral = _project_to_ego_frame(
            ego_x=ego_x,
            ego_y=ego_y,
            ego_heading_rad=ego_heading,
            actor_x=actor_x,
            actor_y=actor_y,
        )
        selected.append(
            {
                "track_id": _stable_token_int(box["track_token"]),
                "x": actor_x,
                "y": actor_y,
                "speed": _speed(box["vx"], box["vy"]),
                "length": float(box["length"] or 4.0),
                "width": float(box["width"] or 2.0),
                "rel_x": rel_x,
                "rel_y": rel_y,
                "longitudinal": float(longitudinal),
                "lateral": float(lateral),
                "category": str(box["category_name"] or "unknown"),
                "distance": float(distance),
            }
        )
    selected.sort(
        key=lambda item: (
            0 if item["longitudinal"] >= -5.0 else 1,
            abs(float(item["lateral"])),
            float(item["distance"]),
        )
    )
    return selected[:MAX_NEIGHBORS]


def _window_rows(
    *,
    source_archive_id: str,
    source_archive_path: Path,
    db_entry_name: str,
    db_record_index: int,
    window_index: int,
    window_start: int,
    window: list[sqlite3.Row],
    boxes_by_lidar: dict[str, list[sqlite3.Row]],
) -> list[dict[str, Any]]:
    scenario_name = str(window[0]["scene_name"] or "scene")
    shard_id = f"{source_archive_id}/{db_entry_name}"
    scenario_id = f"nuplan:{source_archive_id}:{Path(db_entry_name).stem}:{scenario_name}:{window_start}"
    source_dataset = _nuplan_source_dataset(source_archive_path, db_entry_name, window[0]["location"])
    speed_limit = _clip_speed_limit(_speed(row["vx"], row["vy"]) for row in window)
    rows: list[dict[str, Any]] = []
    for step_index, row in enumerate(window):
        ego_heading = _yaw_from_quaternion(
            float(row["qw"] or 1.0),
            float(row["qx"] or 0.0),
            float(row["qy"] or 0.0),
            float(row["qz"] or 0.0),
        )
        lidar_key = _token_hex(row["lidar_pc_token"])
        neighbors = _select_neighbors(row, boxes_by_lidar.get(lidar_key, []), ego_heading=ego_heading)
        categories = sorted({str(item["category"]) for item in neighbors})
        state: dict[str, Any] = {
            "scenario_id": scenario_id,
            "shard_id": shard_id,
            "source_dataset": source_dataset,
            "record_index": int(db_record_index * 1_000_000 + window_index),
            "step_index": int(step_index),
            "timestamp_us": int(row["timestamp_us"]),
            "ts_utc": _timestamp_to_iso(int(row["timestamp_us"])),
            "ego_track_id": 0,
            "ego_x_m": float(row["ego_x"]),
            "ego_y_m": float(row["ego_y"]),
            "ego_speed_mps": _speed(row["vx"], row["vy"]),
            "ego_velocity_x_mps": float(row["vx"] or 0.0),
            "ego_velocity_y_mps": float(row["vy"] or 0.0),
            "ego_heading_rad": ego_heading,
            "ego_length_m": DEFAULT_EGO_LENGTH_M,
            "ego_width_m": DEFAULT_EGO_WIDTH_M,
            "ego_valid": True,
            "speed_limit_mps": speed_limit,
            "neighbor_count": int(len(neighbors)),
            "object_mix_bin": f"nuplan_{row['location'] or 'unknown'}_n{len(neighbors)}_{','.join(categories) if categories else 'none'}",
            **_slot_fields("neighbor_slot_"),
        }
        for slot, neighbor in enumerate(neighbors):
            prefix = f"neighbor_slot_{slot}"
            state[f"{prefix}_track_id"] = int(neighbor["track_id"])
            state[f"{prefix}_x_m"] = float(neighbor["x"])
            state[f"{prefix}_y_m"] = float(neighbor["y"])
            state[f"{prefix}_speed_mps"] = float(neighbor["speed"])
            state[f"{prefix}_length_m"] = float(neighbor["length"])
            state[f"{prefix}_width_m"] = float(neighbor["width"])
            state[f"{prefix}_rel_x_m"] = float(neighbor["rel_x"])
            state[f"{prefix}_rel_y_m"] = float(neighbor["rel_y"])
            state[f"{prefix}_rel_longitudinal_gap_m"] = float(neighbor["longitudinal"])
            state[f"{prefix}_rel_lateral_offset_m"] = float(neighbor["lateral"])
            state[f"{prefix}_valid"] = True
        state.update(compute_state_safety_metrics(state))
        rows.append(state)
    return rows


def _extract_db_member(archive: zipfile.ZipFile, member: str, temp_root: Path) -> Path:
    temp_root.mkdir(parents=True, exist_ok=True)
    target = temp_root / Path(member).name
    with archive.open(member) as source, target.open("wb") as sink:
        shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    return target


def build_nuplan_replay_surface(
    *,
    train_zip: str | Path | Sequence[str | Path] | None = None,
    train_zips: Sequence[str | Path] | None = None,
    train_dirs: Sequence[str | Path] | None = None,
    train_glob: str = DEFAULT_TRAIN_GLOB,
    archive_role: str = "train",
    skip_incomplete: bool = True,
    out_dir: str | Path,
    maps_zip: str | Path | None = None,
    temp_dir: str | Path | None = None,
    max_dbs: int | None = None,
    max_scenarios: int | None = None,
    max_dbs_per_archive: int | None = None,
    max_scenarios_per_archive: int | None = None,
    scenario_stride: int = DEFAULT_SCENARIO_STRIDE,
    batch_size: int = 8_192,
) -> dict[str, Any]:
    """Convert local nuPlan DB logs into ORIUS ``replay_windows.parquet``."""
    explicit_train_zips: list[str | Path] = []
    if train_zip is not None:
        if isinstance(train_zip, str | Path):
            explicit_train_zips.append(train_zip)
        else:
            explicit_train_zips.extend(train_zip)
    explicit_train_zips.extend(train_zips or [])
    archives, skipped_archives = resolve_nuplan_train_archives(
        train_zips=explicit_train_zips,
        train_dirs=train_dirs,
        train_glob=train_glob,
        skip_incomplete=skip_incomplete,
    )
    config = NuPlanSurfaceConfig(
        train_zips=tuple(archive.path for archive in archives),
        maps_zip=Path(maps_zip) if maps_zip is not None else None,
        out_dir=Path(out_dir),
        archive_role=str(archive_role),
        temp_dir=Path(temp_dir) if temp_dir is not None else None,
        max_dbs=max_dbs,
        max_scenarios=max_scenarios,
        max_dbs_per_archive=max_dbs_per_archive,
        max_scenarios_per_archive=max_scenarios_per_archive,
        scenario_stride=scenario_stride,
        batch_size=batch_size,
    )
    archive_manifest = {
        "generated_at_utc": _utc_now_iso(),
        "archive_role": config.archive_role,
        "train": _archive_record(archives[0]),
        "train_archives": [_archive_record(archive) for archive in archives],
        "skipped_train_archives": skipped_archives,
        "total_db_count": int(sum(archive.db_count for archive in archives)),
        "maps": _zip_inventory(config.maps_zip, db_only=False),
    }
    out_path = config.out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    source_manifest_path = out_path / SOURCE_MANIFEST_FILENAME
    source_manifest_path.write_text(json.dumps(archive_manifest, indent=2), encoding="utf-8")

    replay_path = out_path / "replay_windows.parquet"
    writer = _ReplayParquetWriter(replay_path, schema=_replay_schema(), batch_size=config.batch_size)
    inventory_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    row_count = 0
    scenario_count = 0
    db_count = 0
    source_datasets_seen: set[str] = set()

    temp_owner: tempfile.TemporaryDirectory[str] | None = None
    if config.temp_dir is None:
        temp_owner = tempfile.TemporaryDirectory(prefix="orius_nuplan_")
        temp_root = Path(temp_owner.name)
    else:
        temp_root = config.temp_dir
        temp_root.mkdir(parents=True, exist_ok=True)

    completed_without_error = False
    try:
        for source_archive in archives:
            if config.max_dbs is not None and db_count >= int(config.max_dbs):
                break
            if config.max_scenarios is not None and scenario_count >= int(config.max_scenarios):
                break
            _emit_progress(
                {
                    "event": "nuplan_archive_start",
                    "archive": str(source_archive.path.name),
                    "archive_db_count": int(source_archive.db_count),
                    "global_db_count": int(db_count),
                    "global_scenario_count": int(scenario_count),
                }
            )
            archive_db_count = 0
            archive_scenario_count = 0
            with zipfile.ZipFile(source_archive.path) as archive:
                db_members = [info.filename for info in archive.infolist() if info.filename.endswith(".db")]
                for db_member in db_members:
                    if config.max_dbs is not None and db_count >= int(config.max_dbs):
                        break
                    if config.max_scenarios is not None and scenario_count >= int(config.max_scenarios):
                        break
                    if config.max_dbs_per_archive is not None and archive_db_count >= int(
                        config.max_dbs_per_archive
                    ):
                        break
                    if config.max_scenarios_per_archive is not None and archive_scenario_count >= int(
                        config.max_scenarios_per_archive
                    ):
                        break

                    extracted_db = _extract_db_member(archive, db_member, temp_root)
                    try:
                        lidar_rows = _fetch_lidar_rows(extracted_db)
                        db_scenario_count = 0
                        for _scene_index, scene_rows in enumerate(_scene_groups(lidar_rows)):
                            for window_start, window in _candidate_windows(
                                scene_rows, stride=config.scenario_stride
                            ):
                                if config.max_scenarios is not None and scenario_count >= int(
                                    config.max_scenarios
                                ):
                                    break
                                if (
                                    config.max_scenarios_per_archive is not None
                                    and archive_scenario_count >= int(config.max_scenarios_per_archive)
                                ):
                                    break
                                lidar_tokens = [row["lidar_pc_token"] for row in window]
                                boxes_by_lidar = _fetch_boxes_for_tokens(extracted_db, lidar_tokens)
                                window_records = _window_rows(
                                    source_archive_id=source_archive.archive_id,
                                    source_archive_path=source_archive.path,
                                    db_entry_name=db_member,
                                    db_record_index=db_count,
                                    window_index=db_scenario_count,
                                    window_start=window_start,
                                    window=window,
                                    boxes_by_lidar=boxes_by_lidar,
                                )
                                source_dataset = str(window_records[0]["source_dataset"])
                                source_datasets_seen.add(source_dataset)
                                for output_row in window_records:
                                    writer.append(output_row)
                                scenario_id = str(window_records[0]["scenario_id"])
                                scenario_rows.append(
                                    {
                                        "scenario_id": scenario_id,
                                        "shard_id": str(window_records[0]["shard_id"]),
                                        "record_index": int(window_records[0]["record_index"]),
                                        "source_archive_id": source_archive.archive_id,
                                        "source_archive_path": str(source_archive.path),
                                        "db_entry": db_member,
                                        "location": str(window[0]["location"] or ""),
                                        "map_version": str(window[0]["map_version"] or ""),
                                        "scene_name": str(window[0]["scene_name"] or ""),
                                        "window_start": int(window_start),
                                        "step_count": int(len(window_records)),
                                        "start_timestamp_us": int(window_records[0]["timestamp_us"]),
                                        "end_timestamp_us": int(window_records[-1]["timestamp_us"]),
                                        "source_dataset": source_dataset,
                                    }
                                )
                                row_count += len(window_records)
                                scenario_count += 1
                                archive_scenario_count += 1
                                db_scenario_count += 1
                            if config.max_scenarios is not None and scenario_count >= int(
                                config.max_scenarios
                            ):
                                break
                            if config.max_scenarios_per_archive is not None and archive_scenario_count >= int(
                                config.max_scenarios_per_archive
                            ):
                                break
                        inventory_rows.append(
                            {
                                "source_archive_id": source_archive.archive_id,
                                "source_archive_path": str(source_archive.path),
                                "db_entry": db_member,
                                "lidar_pc_rows": int(len(lidar_rows)),
                                "scenario_windows": int(db_scenario_count),
                            }
                        )
                    finally:
                        extracted_db.unlink(missing_ok=True)
                    db_count += 1
                    archive_db_count += 1
                    if archive_db_count == 1 or archive_db_count % 250 == 0:
                        _emit_progress(
                            {
                                "event": "nuplan_archive_progress",
                                "archive": str(source_archive.path.name),
                                "archive_dbs_processed": int(archive_db_count),
                                "archive_scenarios": int(archive_scenario_count),
                                "global_db_count": int(db_count),
                                "global_scenario_count": int(scenario_count),
                                "global_row_count": int(row_count),
                            }
                        )
            _emit_progress(
                {
                    "event": "nuplan_archive_done",
                    "archive": str(source_archive.path.name),
                    "archive_dbs_processed": int(archive_db_count),
                    "archive_scenarios": int(archive_scenario_count),
                    "global_db_count": int(db_count),
                    "global_scenario_count": int(scenario_count),
                    "global_row_count": int(row_count),
                }
            )
        completed_without_error = True
    finally:
        writer.close(finalize=completed_without_error and row_count > 0)
        if temp_owner is not None:
            temp_owner.cleanup()

    if row_count <= 0:
        replay_path.unlink(missing_ok=True)
        raise ValueError(
            "No nuPlan replay rows were generated. Check max_scenarios, max_dbs, and DB window length."
        )

    inventory_path = out_path / "nuplan_db_inventory.csv"
    scenario_index_path = out_path / "scenario_index.parquet"
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    pd.DataFrame(scenario_rows).to_parquet(scenario_index_path, index=False)

    report = {
        "source_dataset": _summarize_source_datasets(source_datasets_seen),
        "source_datasets": sorted(source_datasets_seen),
        "row_count": int(row_count),
        "scenario_count": int(scenario_count),
        "db_count": int(db_count),
        "scenario_steps": int(TOTAL_SCENARIO_STEPS),
        "scenario_stride": int(config.scenario_stride),
        "bounds": {
            "max_dbs": config.max_dbs,
            "max_scenarios": config.max_scenarios,
            "max_dbs_per_archive": config.max_dbs_per_archive,
            "max_scenarios_per_archive": config.max_scenarios_per_archive,
        },
        "artifacts": {
            "replay_windows": str(replay_path),
            "scenario_index": str(scenario_index_path),
            "db_inventory": str(inventory_path),
            "source_manifest": str(source_manifest_path),
        },
        "inputs": archive_manifest,
    }
    (out_path / "nuplan_surface_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
