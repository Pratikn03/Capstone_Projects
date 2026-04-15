#!/usr/bin/env python3
"""Run the 1k-scenario Stage 1–5 Waymo AV dry run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orius.av_waymo import (
    build_feature_tables,
    build_replay_surface,
    build_subset_manifest,
    build_validation_surface,
    run_runtime_dry_run,
    train_dry_run_models,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "orius_av" / "raw" / "waymo_motion" / "validation"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "orius_av" / "av" / "processed"
DEFAULT_MODELS = REPO_ROOT / "artifacts" / "models_orius_av_dryrun"
DEFAULT_UNCERTAINTY = REPO_ROOT / "artifacts" / "uncertainty" / "orius_av_dryrun"
DEFAULT_REPORTS = REPO_ROOT / "reports" / "orius_av" / "dry_run"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Waymo ORIUS AV dry run")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--max-validation-shards", type=int, default=None)
    parser.add_argument("--max-validation-scenarios", type=int, default=None)
    parser.add_argument("--skip-actor-tracks", action="store_true")
    parser.add_argument("--max-runtime-scenarios", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--uncertainty-dir", type=Path, default=DEFAULT_UNCERTAINTY)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()

    report: dict[str, object] = {}
    if not args.skip_validation:
        report["validation"] = build_validation_surface(
            raw_dir=args.raw_dir,
            out_dir=args.processed_dir,
            max_shards=args.max_validation_shards,
            max_scenarios=args.max_validation_scenarios,
            write_actor_tracks=not args.skip_actor_tracks,
        )

    report["subset"] = build_subset_manifest(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        target_count=args.subset_size,
    )
    subset_manifest_path = args.processed_dir / "dry_run_subset_manifest.parquet"
    report["replay"] = build_replay_surface(
        raw_dir=args.raw_dir,
        subset_manifest_path=subset_manifest_path,
        out_dir=args.processed_dir,
    )
    report["features"] = build_feature_tables(
        replay_windows_path=args.processed_dir / "replay_windows.parquet",
        out_dir=args.processed_dir,
    )

    if not args.skip_training:
        report["training"] = train_dry_run_models(
            anchor_features_path=args.processed_dir / "anchor_features.parquet",
            step_features_path=args.processed_dir / "step_features.parquet",
            models_dir=args.models_dir,
            uncertainty_dir=args.uncertainty_dir,
            reports_dir=args.reports_dir,
        )

    if not args.skip_runtime:
        report["runtime"] = run_runtime_dry_run(
            replay_windows_path=args.processed_dir / "replay_windows.parquet",
            step_features_path=args.processed_dir / "step_features.parquet",
            models_dir=args.models_dir,
            out_dir=args.reports_dir,
            max_scenarios=args.max_runtime_scenarios,
        )

    out_path = args.reports_dir / "dry_run_report.json"
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
