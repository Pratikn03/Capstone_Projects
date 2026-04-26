#!/usr/bin/env python3
"""Build an ORIUS AV replay surface from local nuPlan zip archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.av_nuplan import DEFAULT_TRAIN_GLOB, build_feature_tables, build_nuplan_replay_surface, inspect_nuplan_archives


DEFAULT_TRAIN_ZIP = REPO_ROOT / "nuplan-v1.1_train_singapore.zip"
DEFAULT_MAPS_ZIP = REPO_ROOT / "nuplan-maps-v1.0.zip"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "orius_av" / "av" / "processed_nuplan_singapore"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ORIUS replay windows from nuPlan Singapore DB logs")
    parser.add_argument("--train-zip", type=Path, action="append", default=None, help="nuPlan train split zip containing .db logs; repeat for multiple archives")
    parser.add_argument("--train-dir", type=Path, action="append", default=None, help="Directory containing completed nuPlan train zip archives; repeat as needed")
    parser.add_argument("--train-glob", type=str, default=DEFAULT_TRAIN_GLOB, help="Glob used inside each --train-dir")
    parser.add_argument("--archive-role", choices=("train", "val", "test", "trainval"), default="train", help="Evidence role recorded in the nuPlan source manifest")
    parser.add_argument("--skip-incomplete", action=argparse.BooleanOptionalAction, default=True, help="Skip .crdownload, invalid, or incomplete archives")
    parser.add_argument("--hf-dataset", type=str, default=None, help="Private HF dataset repo containing nuPlan zip archives")
    parser.add_argument("--hf-cache-dir", type=Path, default=None, help="Optional Hugging Face snapshot cache directory")
    parser.add_argument("--maps-zip", type=Path, default=DEFAULT_MAPS_ZIP, help="nuPlan maps zip; inventoried for provenance")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for ORIUS AV parquet artifacts")
    parser.add_argument("--temp-dir", type=Path, default=None, help="Optional extraction scratch directory")
    parser.add_argument("--max-dbs", type=int, default=None, help="Bound the number of DB logs converted")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Bound generated 91-step scenario windows")
    parser.add_argument("--max-dbs-per-archive", type=int, default=None, help="Bound DB logs converted from each completed archive")
    parser.add_argument(
        "--max-scenarios-per-archive",
        type=int,
        default=None,
        help="Bound generated scenario windows from each completed archive",
    )
    parser.add_argument("--scenario-stride", type=int, default=91, help="Stride between generated windows within each scene")
    parser.add_argument("--batch-size", type=int, default=8192, help="Parquet writer batch size")
    parser.add_argument("--build-features", action="store_true", help="Also build ORIUS step and anchor feature tables")
    parser.add_argument(
        "--split-strategy",
        choices=("hash", "balanced", "all_test", "grouped_archive_db_city"),
        default="grouped_archive_db_city",
        help="Feature split policy when --build-features is set; use all_test for held-out val/test surfaces",
    )
    parser.add_argument("--inspect-only", action="store_true", help="Only print archive manifests without extracting DB files")
    return parser.parse_args()


def _download_hf_dataset(repo_id: str, *, cache_dir: Path | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required for --hf-dataset") from exc
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=["*.zip", "**/*.zip", "*.json", "**/*.json"],
        ignore_patterns=["*.crdownload", "**/*.crdownload"],
    )
    return Path(snapshot_path)


def main() -> int:
    args = _parse_args()
    train_dirs = list(args.train_dir or [])
    train_zips = list(args.train_zip or [])
    hf_temp: tempfile.TemporaryDirectory[str] | None = None
    if args.hf_dataset is not None:
        if args.hf_cache_dir is None:
            hf_temp = tempfile.TemporaryDirectory(prefix="orius_nuplan_hf_")
            args.hf_cache_dir = Path(hf_temp.name)
        hf_snapshot_dir = _download_hf_dataset(args.hf_dataset, cache_dir=args.hf_cache_dir)
        train_dirs.append(hf_snapshot_dir)
        if not args.maps_zip.exists():
            maps_candidates = sorted(hf_snapshot_dir.glob("**/nuplan-maps*.zip"))
            if maps_candidates:
                args.maps_zip = maps_candidates[0]
    if not train_zips and not train_dirs:
        train_zips = [DEFAULT_TRAIN_ZIP]

    if args.inspect_only:
        print(
            json.dumps(
                inspect_nuplan_archives(
                    train_zips=train_zips,
                    train_dirs=train_dirs,
                    train_glob=args.train_glob,
                    skip_incomplete=args.skip_incomplete,
                    maps_zip=args.maps_zip,
                ),
                indent=2,
            )
        )
        if hf_temp is not None:
            hf_temp.cleanup()
        return 0

    try:
        report = build_nuplan_replay_surface(
            train_zips=train_zips,
            train_dirs=train_dirs,
            train_glob=args.train_glob,
            archive_role=args.archive_role,
            skip_incomplete=args.skip_incomplete,
            maps_zip=args.maps_zip,
            out_dir=args.out_dir,
            temp_dir=args.temp_dir,
            max_dbs=args.max_dbs,
            max_scenarios=args.max_scenarios,
            max_dbs_per_archive=args.max_dbs_per_archive,
            max_scenarios_per_archive=args.max_scenarios_per_archive,
            scenario_stride=args.scenario_stride,
            batch_size=args.batch_size,
        )
        if args.build_features:
            report["features"] = build_feature_tables(
                replay_windows_path=args.out_dir / "replay_windows.parquet",
                out_dir=args.out_dir,
                split_strategy=args.split_strategy,
            )
            (args.out_dir / "nuplan_surface_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if hf_temp is not None:
            hf_temp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
