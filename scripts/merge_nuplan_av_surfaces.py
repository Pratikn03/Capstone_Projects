#!/usr/bin/env python3
"""Merge validated nuPlan ORIUS AV surfaces and optionally rebuild features."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.av_nuplan import build_feature_tables
from orius.av_waymo.training import (
    GROUPED_ARCHIVE_DB_CITY_SPLIT,
    assign_balanced_splits,
    assign_grouped_archive_db_city_splits,
    assign_split,
)


FEATURE_BATCH_ROWS = 200_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge nuPlan replay surfaces into one ORIUS AV evidence surface")
    parser.add_argument("--surface-dir", type=Path, action="append", required=True, help="Processed nuPlan surface directory; repeat in merge order")
    parser.add_argument("--out-dir", type=Path, required=True, help="Merged output surface directory")
    parser.add_argument("--maps-zip", type=Path, default=None, help="Optional maps zip path recorded in the merged manifest")
    parser.add_argument("--archive-role", default="all_completed_grouped", help="Merged evidence role recorded in the manifest")
    parser.add_argument("--build-features", action="store_true", help="Build merged feature tables after replay/scenario merge")
    parser.add_argument(
        "--reuse-feature-tables",
        action="store_true",
        help="Merge existing per-surface feature tables and rewrite split labels instead of recomputing features",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("hash", "balanced", "all_test", "grouped_archive_db_city"),
        default="grouped_archive_db_city",
        help="Merged feature split strategy",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _surface_paths(surface_dir: Path) -> dict[str, Path]:
    paths = {
        "replay_windows": surface_dir / "replay_windows.parquet",
        "scenario_index": surface_dir / "scenario_index.parquet",
        "db_inventory": surface_dir / "nuplan_db_inventory.csv",
        "source_manifest": surface_dir / "nuplan_source_manifest.json",
        "surface_report": surface_dir / "nuplan_surface_report.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{surface_dir} is not a complete nuPlan surface; missing: {missing}")
    return paths


def _maps_inventory(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Missing maps zip: {path}")
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
    }


def _copy_parquet_streaming(sources: list[Path], dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_name(f"{dest.name}.tmp")
    temp_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for source in sources:
            parquet_file = pq.ParquetFile(source)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, parquet_file.schema_arrow)
            elif parquet_file.schema_arrow != writer.schema:
                raise ValueError(f"Parquet schema mismatch while merging {source}")
            for row_group_index in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_index)
                writer.write_table(table)
                row_count += int(table.num_rows)
        if writer is None:
            raise ValueError("No parquet sources were provided")
        writer.close()
        writer = None
        pq.ParquetFile(temp_path)
        temp_path.replace(dest)
    finally:
        if writer is not None:
            writer.close()
        temp_path.unlink(missing_ok=True)
    return row_count


def _emit_progress(event: dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)


def _dedupe_archives(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    archives: list[dict[str, Any]] = []
    for manifest in manifests:
        for archive in manifest.get("train_archives", []):
            if not isinstance(archive, dict):
                continue
            key = str(archive.get("archive_id") or archive.get("path"))
            if key in seen:
                continue
            seen.add(key)
            archives.append(dict(archive))
    return archives


def _merge_surfaces(surface_dirs: list[Path], out_dir: Path, *, maps_zip: Path | None, archive_role: str) -> dict[str, Any]:
    surface_dirs = [path.resolve() for path in surface_dirs]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_by_surface = [_surface_paths(path) for path in surface_dirs]
    reports = [_read_json(paths["surface_report"]) for paths in paths_by_surface]
    manifests = [_read_json(paths["source_manifest"]) for paths in paths_by_surface]

    replay_count = _copy_parquet_streaming([paths["replay_windows"] for paths in paths_by_surface], out_dir / "replay_windows.parquet")

    scenario_frames = [pd.read_parquet(paths["scenario_index"]) for paths in paths_by_surface]
    scenario_index = pd.concat(scenario_frames, ignore_index=True)
    duplicate_scenarios = scenario_index["scenario_id"].astype(str).duplicated()
    if duplicate_scenarios.any():
        examples = scenario_index.loc[duplicate_scenarios, "scenario_id"].astype(str).head(5).tolist()
        raise ValueError(f"Merged nuPlan scenario IDs are not unique; examples: {examples}")
    scenario_index.to_parquet(out_dir / "scenario_index.parquet", index=False)

    inventory_frames = [pd.read_csv(paths["db_inventory"]) for paths in paths_by_surface]
    db_inventory = pd.concat(inventory_frames, ignore_index=True)
    db_inventory.to_csv(out_dir / "nuplan_db_inventory.csv", index=False)

    archives = _dedupe_archives(manifests)
    skipped_archives: list[dict[str, Any]] = []
    for manifest in manifests:
        skipped = manifest.get("skipped_train_archives", [])
        if isinstance(skipped, list):
            skipped_archives.extend(row for row in skipped if isinstance(row, dict))

    source_datasets = sorted({str(value) for value in scenario_index.get("source_dataset", pd.Series(dtype=str)).dropna().unique()})
    source_dataset = source_datasets[0] if len(source_datasets) == 1 else "nuplan_multi_city"
    source_manifest = {
        "generated_at_utc": _utc_now_iso(),
        "archive_role": archive_role,
        "merge_source_dirs": [str(path) for path in surface_dirs],
        "train": archives[0] if archives else {},
        "train_archives": archives,
        "skipped_train_archives": skipped_archives,
        "total_db_count": int(sum(int(archive.get("db_count", 0)) for archive in archives)),
        "maps": _maps_inventory(maps_zip) or next((manifest.get("maps") for manifest in manifests if manifest.get("maps")), None),
    }
    _write_json(out_dir / "nuplan_source_manifest.json", source_manifest)

    report = {
        "source_dataset": source_dataset,
        "source_datasets": source_datasets,
        "row_count": int(replay_count),
        "scenario_count": int(len(scenario_index)),
        "db_count": int(len(db_inventory)),
        "scenario_steps": int(reports[0].get("scenario_steps", 91)) if reports else 91,
        "scenario_stride": int(reports[0].get("scenario_stride", 91)) if reports else 91,
        "bounds": {
            "merged_surface_count": int(len(surface_dirs)),
            "merged_surface_dirs": [str(path) for path in surface_dirs],
        },
        "artifacts": {
            "replay_windows": str(out_dir / "replay_windows.parquet"),
            "scenario_index": str(out_dir / "scenario_index.parquet"),
            "db_inventory": str(out_dir / "nuplan_db_inventory.csv"),
            "source_manifest": str(out_dir / "nuplan_source_manifest.json"),
        },
        "inputs": source_manifest,
        "merge_reports": reports,
    }
    _write_json(out_dir / "nuplan_surface_report.json", report)
    return report


def _split_map_for_strategy(scenario_index: pd.DataFrame, split_strategy: str) -> tuple[dict[str, str], dict[str, Any]]:
    scenario_ids = scenario_index["scenario_id"].astype(str).unique()
    if split_strategy == "all_test":
        return {str(scenario_id): "test" for scenario_id in scenario_ids}, {}
    if split_strategy == "balanced":
        return assign_balanced_splits(scenario_ids), {}
    if split_strategy == "hash":
        return {str(scenario_id): assign_split(str(scenario_id)) for scenario_id in scenario_ids}, {}
    if split_strategy == GROUPED_ARCHIVE_DB_CITY_SPLIT:
        return assign_grouped_archive_db_city_splits(scenario_index)
    raise ValueError(f"Unsupported split_strategy={split_strategy!r}")


def _merge_feature_tables(
    surface_dirs: list[Path],
    out_dir: Path,
    *,
    split_strategy: str,
) -> dict[str, Any]:
    scenario_index = pd.read_parquet(out_dir / "scenario_index.parquet")
    split_map, split_metadata = _split_map_for_strategy(scenario_index, split_strategy)

    feature_sources = []
    anchor_sources = []
    for surface_dir in surface_dirs:
        step_path = surface_dir / "step_features.parquet"
        anchor_path = surface_dir / "anchor_features.parquet"
        if not step_path.exists() or not anchor_path.exists():
            raise FileNotFoundError(
                f"{surface_dir} must contain step_features.parquet and anchor_features.parquet for --reuse-feature-tables"
            )
        feature_sources.append(step_path)
        anchor_sources.append(anchor_path)

    step_features_path = out_dir / "step_features.parquet"
    tmp_step_path = step_features_path.with_name(f"{step_features_path.name}.tmp")
    tmp_step_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    feature_schema: pa.Schema | None = None
    row_count = 0
    next_progress_row_count = 1_000_000
    try:
        for source in feature_sources:
            _emit_progress({"event": "nuplan_feature_merge_source_start", "source": str(source)})
            for batch in pq.ParquetFile(source).iter_batches(batch_size=FEATURE_BATCH_ROWS):
                frame = batch.to_pandas()
                frame["split"] = frame["scenario_id"].astype(str).map(split_map)
                if frame["split"].isna().any():
                    missing = frame.loc[frame["split"].isna(), "scenario_id"].astype(str).head(5).tolist()
                    raise ValueError(f"Feature split map is missing scenario IDs: {missing}")
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
                if row_count >= next_progress_row_count:
                    _emit_progress({"event": "nuplan_feature_merge_progress", "feature_rows": int(row_count)})
                    while row_count >= next_progress_row_count:
                        next_progress_row_count += 1_000_000
            _emit_progress({"event": "nuplan_feature_merge_source_done", "source": str(source), "feature_rows": int(row_count)})
        if writer is None:
            raise ValueError("No feature sources were merged")
        writer.close()
        writer = None
        pq.ParquetFile(tmp_step_path)
        tmp_step_path.replace(step_features_path)
    finally:
        if writer is not None:
            writer.close()
        tmp_step_path.unlink(missing_ok=True)

    anchor_frames = [pd.read_parquet(source) for source in anchor_sources]
    anchor_features = pd.concat(anchor_frames, ignore_index=True)
    anchor_features["split"] = anchor_features["scenario_id"].astype(str).map(split_map)
    if anchor_features["split"].isna().any():
        missing = anchor_features.loc[anchor_features["split"].isna(), "scenario_id"].astype(str).head(5).tolist()
        raise ValueError(f"Anchor split map is missing scenario IDs: {missing}")
    anchor_features_path = out_dir / "anchor_features.parquet"
    anchor_features.to_parquet(anchor_features_path, index=False)

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_path in splits_dir.glob("*.parquet"):
        split_path.unlink()
    for split_name, split_df in anchor_features.groupby("split", sort=False):
        split_df.to_parquet(splits_dir / f"{split_name}.parquet", index=False)

    report = {
        "row_count": int(row_count),
        "anchor_row_count": int(len(anchor_features)),
        "scenario_count": int(anchor_features["scenario_id"].nunique()),
        "split_strategy": split_strategy,
        "reused_feature_tables": True,
        "split_counts": {str(key): int(value) for key, value in anchor_features["split"].value_counts().sort_index().items()},
        **split_metadata,
        "artifacts": {
            "step_features": str(step_features_path),
            "anchor_features": str(anchor_features_path),
            "splits_dir": str(splits_dir),
        },
    }
    _write_json(out_dir / "feature_table_report.json", report)
    return report


def main() -> int:
    args = _parse_args()
    if args.build_features and args.reuse_feature_tables:
        raise ValueError("Use either --build-features or --reuse-feature-tables, not both.")
    report = _merge_surfaces(
        list(args.surface_dir),
        args.out_dir,
        maps_zip=args.maps_zip,
        archive_role=args.archive_role,
    )
    if args.build_features:
        report["features"] = build_feature_tables(
            replay_windows_path=args.out_dir / "replay_windows.parquet",
            out_dir=args.out_dir,
            split_strategy=args.split_strategy,
        )
        _write_json(args.out_dir / "nuplan_surface_report.json", report)
    if args.reuse_feature_tables:
        report["features"] = _merge_feature_tables(
            list(args.surface_dir),
            args.out_dir,
            split_strategy=args.split_strategy,
        )
        _write_json(args.out_dir / "nuplan_surface_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
