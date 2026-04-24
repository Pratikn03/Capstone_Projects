#!/usr/bin/env python3
"""Run the battery and AV training surfaces as one bounded pipeline."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SRC_DIR, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import build_waymo_av_dry_run_report as av_report_script
import build_battery_av_closure_artifacts as closure_script
import build_orius_ieee_assets as ieee_assets_script
import build_orius_monograph_assets as monograph_assets_script
import run_battery_deep_novelty as battery_script
from orius.av_nuplan import build_nuplan_replay_surface
from orius.av_waymo import (
    build_feature_tables,
    build_replay_surface,
    build_subset_manifest,
    build_validation_surface,
    run_runtime_dry_run,
    train_dry_run_models,
)


DEFAULT_OUT_ROOT = REPO_ROOT / "reports" / "battery_av"
DEFAULT_BATTERY_OUT = DEFAULT_OUT_ROOT / "battery"
DEFAULT_BATTERY_MODEL_DIR = REPO_ROOT / "artifacts" / "battery_av" / "deep_oqe"
DEFAULT_BATTERY_PAPER_TABLE_DIR = DEFAULT_BATTERY_OUT / "paper_assets" / "tables" / "generated"
DEFAULT_BATTERY_PAPER_FIG_DIR = DEFAULT_BATTERY_OUT / "paper_assets" / "figures"

DEFAULT_AV_RAW = REPO_ROOT / "data" / "orius_av" / "raw" / "waymo_motion" / "validation"
DEFAULT_AV_PROCESSED = REPO_ROOT / "data" / "orius_av" / "av" / "processed_full_corpus"
DEFAULT_NUPLAN_PROCESSED = REPO_ROOT / "data" / "orius_av" / "av" / "processed_nuplan_singapore"
DEFAULT_NUPLAN_TRAIN_ZIP = REPO_ROOT / "nuplan-v1.1_train_singapore.zip"
DEFAULT_NUPLAN_MAPS_ZIP = REPO_ROOT / "nuplan-maps-v1.0.zip"
DEFAULT_AV_MODELS = REPO_ROOT / "artifacts" / "models_orius_av_full_corpus"
DEFAULT_AV_UNCERTAINTY = REPO_ROOT / "artifacts" / "uncertainty" / "orius_av_full_corpus"
DEFAULT_AV_REPORTS = REPO_ROOT / "reports" / "orius_av" / "full_corpus"
DEFAULT_OVERALL = DEFAULT_OUT_ROOT / "overall"
DEFAULT_AV_SOURCE = os.environ.get("ORIUS_AV_SOURCE", "nuplan_singapore")


def _is_appledouble(path: Path) -> bool:
    return any(part.startswith("._") for part in path.parts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the battery + AV pipelines together")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Root reports directory for the combined battery+AV surface")
    parser.add_argument("--overall-dir", type=Path, default=DEFAULT_OVERALL, help="Directory for the combined manifest and summary")
    parser.add_argument("--skip-battery", action="store_true", help="Skip the battery training surface")
    parser.add_argument("--skip-av", action="store_true", help="Skip the AV training surface")
    parser.add_argument("--submission-scope", type=str, default="battery_av_only")

    parser.add_argument("--battery-out-dir", type=Path, default=DEFAULT_BATTERY_OUT)
    parser.add_argument("--battery-model-dir", type=Path, default=DEFAULT_BATTERY_MODEL_DIR)
    parser.add_argument("--battery-paper-table-dir", type=Path, default=DEFAULT_BATTERY_PAPER_TABLE_DIR)
    parser.add_argument("--battery-paper-fig-dir", type=Path, default=DEFAULT_BATTERY_PAPER_FIG_DIR)
    parser.add_argument("--battery-features", type=Path, default=battery_script.FEATURES_PATH)
    parser.add_argument("--battery-engineered-baseline", type=Path, default=battery_script.ENGINEERED_BASELINE_PATH)
    parser.add_argument("--battery-deep-oqe-epochs", type=int, default=12)
    parser.add_argument("--battery-forecast-epochs", type=int, default=8)
    parser.add_argument("--battery-batch-size", type=int, default=128)
    parser.add_argument("--battery-seq-len", type=int, default=8)
    parser.add_argument("--battery-lookback", type=int, default=168)
    parser.add_argument("--battery-horizon", type=int, default=24)
    parser.add_argument("--battery-train-stride", type=int, default=6)
    parser.add_argument("--battery-eval-stride", type=int, default=12)

    parser.add_argument("--av-source", choices=["nuplan", "nuplan_singapore", "waymo_motion"], default=DEFAULT_AV_SOURCE)
    parser.add_argument("--av-raw-dir", type=Path, default=DEFAULT_AV_RAW)
    parser.add_argument("--av-processed-dir", type=Path, default=None)
    parser.add_argument("--av-models-dir", type=Path, default=DEFAULT_AV_MODELS)
    parser.add_argument("--av-uncertainty-dir", type=Path, default=DEFAULT_AV_UNCERTAINTY)
    parser.add_argument("--av-reports-dir", type=Path, default=DEFAULT_AV_REPORTS)
    parser.add_argument("--nuplan-train-zip", type=Path, action="append", default=None)
    parser.add_argument("--nuplan-train-dir", type=Path, action="append", default=None)
    parser.add_argument("--nuplan-train-glob", type=str, default="nuplan-v*.zip")
    parser.add_argument("--nuplan-archive-role", choices=["train", "val", "test", "trainval"], default="train")
    parser.add_argument("--nuplan-skip-incomplete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nuplan-maps-zip", type=Path, default=DEFAULT_NUPLAN_MAPS_ZIP)
    parser.add_argument("--nuplan-temp-dir", type=Path, default=None)
    parser.add_argument("--nuplan-max-dbs", type=int, default=None)
    parser.add_argument("--nuplan-max-scenarios", type=int, default=None)
    parser.add_argument("--nuplan-max-dbs-per-archive", type=int, default=None)
    parser.add_argument("--nuplan-max-scenarios-per-archive", type=int, default=None)
    parser.add_argument("--nuplan-scenario-stride", type=int, default=91)
    parser.add_argument("--nuplan-split-strategy", choices=["balanced", "hash", "all_test"], default="balanced")
    parser.add_argument("--av-subset-size", type=int, default=1000, help="Subset size when using --av-subset")
    parser.add_argument("--av-full-corpus", dest="av_full_corpus", action="store_true", default=True, help="Use every scenario in scenario_index.parquet across the 7 validation shards")
    parser.add_argument("--av-subset", dest="av_full_corpus", action="store_false", help="Use a bounded subset instead of the canonical full-corpus AV surface")
    parser.add_argument("--av-max-validation-shards", type=int, default=None)
    parser.add_argument("--av-max-validation-scenarios", type=int, default=None)
    parser.add_argument("--av-max-runtime-scenarios", type=int, default=None)
    parser.add_argument("--av-skip-actor-tracks", action="store_true")
    parser.add_argument("--av-skip-validation", action="store_true")
    parser.add_argument("--av-skip-training", action="store_true")
    parser.add_argument("--av-skip-runtime", action="store_true")
    parser.add_argument("--av-skip-report", action="store_true")
    args = parser.parse_args()
    if args.out_root != DEFAULT_OUT_ROOT:
        if args.battery_out_dir == DEFAULT_BATTERY_OUT:
            args.battery_out_dir = args.out_root / "battery"
        if args.battery_paper_table_dir == DEFAULT_BATTERY_PAPER_TABLE_DIR:
            args.battery_paper_table_dir = args.out_root / "battery" / "paper_assets" / "tables" / "generated"
        if args.battery_paper_fig_dir == DEFAULT_BATTERY_PAPER_FIG_DIR:
            args.battery_paper_fig_dir = args.out_root / "battery" / "paper_assets" / "figures"
        if args.overall_dir == DEFAULT_OVERALL:
            args.overall_dir = args.out_root / "overall"
    if args.av_processed_dir is None:
        args.av_processed_dir = DEFAULT_NUPLAN_PROCESSED if args.av_source == "nuplan_singapore" else DEFAULT_AV_PROCESSED
    if args.nuplan_train_zip is None and args.nuplan_train_dir is None:
        args.nuplan_train_zip = [DEFAULT_NUPLAN_TRAIN_ZIP]
    return args


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _collect_file_paths(value: Any, sink: set[Path]) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _collect_file_paths(nested, sink)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _collect_file_paths(nested, sink)
        return
    if not isinstance(value, str):
        return
    candidate = Path(value)
    if candidate.exists() and candidate.is_file() and not _is_appledouble(candidate):
        sink.add(candidate.resolve())


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["domain", "status", "out_dir", "key_report", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _input_hashes(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {"battery": {}, "av": {}}

    battery_features = getattr(args, "battery_features", None)
    if isinstance(battery_features, Path) and battery_features.exists() and battery_features.is_file():
        payload["battery"][str(battery_features)] = _sha256_file(battery_features)
    battery_baseline = getattr(args, "battery_engineered_baseline", None)
    if isinstance(battery_baseline, Path) and battery_baseline.exists() and battery_baseline.is_file():
        payload["battery"][str(battery_baseline)] = _sha256_file(battery_baseline)

    av_processed_dir = getattr(args, "av_processed_dir", None)
    if isinstance(av_processed_dir, Path):
        subset_manifest = av_processed_dir / "dry_run_subset_manifest.json"
        subset_payload = _load_json_file(subset_manifest)
        raw_hashes = subset_payload.get("raw_file_hashes")
        if isinstance(raw_hashes, dict):
            payload["av"]["raw_file_hashes"] = dict(raw_hashes)
        if subset_manifest.exists():
            payload["av"]["subset_manifest_sha256"] = _sha256_file(subset_manifest)
    av_shift_cfg = getattr(args, "av_reports_dir", None)
    if isinstance(av_shift_cfg, Path):
        shift_cfg_path = av_shift_cfg / "shift_aware_config.json"
        if shift_cfg_path.exists() and shift_cfg_path.is_file():
            payload["av"]["shift_aware_config_sha256"] = _sha256_file(shift_cfg_path)

    return payload


def _resolve_full_corpus_count(processed_dir: Path) -> int:
    scenario_index_path = processed_dir / "scenario_index.parquet"
    if not scenario_index_path.exists():
        raise FileNotFoundError(
            f"Full-corpus AV mode requires {scenario_index_path}. "
            "Run validation first or remove --av-skip-validation."
        )
    scenario_index = pd.read_parquet(scenario_index_path, columns=["scenario_id"])
    if scenario_index.empty:
        raise ValueError(f"{scenario_index_path} contains no scenarios")
    return int(len(scenario_index))


def run_av_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {}
    args.av_reports_dir.mkdir(parents=True, exist_ok=True)
    av_source = getattr(args, "av_source", "waymo_motion")

    if av_source in {"nuplan", "nuplan_singapore"}:
        report["source"] = "nuplan"
        if not args.av_skip_validation:
            report["replay"] = build_nuplan_replay_surface(
                train_zips=getattr(args, "nuplan_train_zip", [DEFAULT_NUPLAN_TRAIN_ZIP]),
                train_dirs=getattr(args, "nuplan_train_dir", None),
                train_glob=getattr(args, "nuplan_train_glob", "nuplan-v*.zip"),
                archive_role=getattr(args, "nuplan_archive_role", "train"),
                skip_incomplete=getattr(args, "nuplan_skip_incomplete", True),
                maps_zip=getattr(args, "nuplan_maps_zip", DEFAULT_NUPLAN_MAPS_ZIP),
                out_dir=args.av_processed_dir,
                temp_dir=getattr(args, "nuplan_temp_dir", None),
                max_dbs=getattr(args, "nuplan_max_dbs", None),
                max_scenarios=getattr(args, "nuplan_max_scenarios", None),
                max_dbs_per_archive=getattr(args, "nuplan_max_dbs_per_archive", None),
                max_scenarios_per_archive=getattr(args, "nuplan_max_scenarios_per_archive", None),
                scenario_stride=getattr(args, "nuplan_scenario_stride", 91),
            )
        else:
            report["replay"] = _load_json_file(args.av_processed_dir / "nuplan_surface_report.json")
        replay_source = str(report.get("replay", {}).get("source_dataset") or av_source)
        report["source"] = replay_source
        report["subset_mode"] = replay_source
        report["subset_size"] = int(report.get("replay", {}).get("scenario_count") or _resolve_full_corpus_count(args.av_processed_dir))
        report["features"] = build_feature_tables(
            replay_windows_path=args.av_processed_dir / "replay_windows.parquet",
            out_dir=args.av_processed_dir,
            split_strategy=getattr(args, "nuplan_split_strategy", "balanced"),
        )

        if not args.av_skip_training:
            report["training"] = train_dry_run_models(
                anchor_features_path=args.av_processed_dir / "anchor_features.parquet",
                step_features_path=args.av_processed_dir / "step_features.parquet",
                models_dir=args.av_models_dir,
                uncertainty_dir=args.av_uncertainty_dir,
                reports_dir=args.av_reports_dir,
                artifact_prefix="nuplan_av",
            )

        if not args.av_skip_runtime:
            report["runtime"] = run_runtime_dry_run(
                replay_windows_path=args.av_processed_dir / "replay_windows.parquet",
                step_features_path=args.av_processed_dir / "step_features.parquet",
                models_dir=args.av_models_dir,
                out_dir=args.av_reports_dir,
                max_scenarios=args.av_max_runtime_scenarios,
                artifact_prefix="nuplan_av",
            )

        if not args.av_skip_report:
            report["report"] = av_report_script.build_report(
                processed_dir=args.av_processed_dir,
                reports_dir=args.av_reports_dir,
                models_dir=args.av_models_dir,
                uncertainty_dir=args.av_uncertainty_dir,
            )
        return report

    if not args.av_skip_validation:
        report["validation"] = build_validation_surface(
            raw_dir=args.av_raw_dir,
            out_dir=args.av_processed_dir,
            max_shards=args.av_max_validation_shards,
            max_scenarios=args.av_max_validation_scenarios,
            write_actor_tracks=not args.av_skip_actor_tracks,
        )

    subset_size = _resolve_full_corpus_count(args.av_processed_dir) if args.av_full_corpus else int(args.av_subset_size)
    report["subset_mode"] = "full_corpus" if args.av_full_corpus else "subset"
    report["subset_size"] = subset_size
    report["subset"] = build_subset_manifest(
        raw_dir=args.av_raw_dir,
        processed_dir=args.av_processed_dir,
        target_count=subset_size,
    )
    subset_manifest_path = args.av_processed_dir / "dry_run_subset_manifest.parquet"

    report["replay"] = build_replay_surface(
        raw_dir=args.av_raw_dir,
        subset_manifest_path=subset_manifest_path,
        out_dir=args.av_processed_dir,
    )
    report["features"] = build_feature_tables(
        replay_windows_path=args.av_processed_dir / "replay_windows.parquet",
        out_dir=args.av_processed_dir,
    )

    if not args.av_skip_training:
        report["training"] = train_dry_run_models(
            anchor_features_path=args.av_processed_dir / "anchor_features.parquet",
            step_features_path=args.av_processed_dir / "step_features.parquet",
            models_dir=args.av_models_dir,
            uncertainty_dir=args.av_uncertainty_dir,
            reports_dir=args.av_reports_dir,
        )

    if not args.av_skip_runtime:
        report["runtime"] = run_runtime_dry_run(
            replay_windows_path=args.av_processed_dir / "replay_windows.parquet",
            step_features_path=args.av_processed_dir / "step_features.parquet",
            models_dir=args.av_models_dir,
            out_dir=args.av_reports_dir,
            max_scenarios=args.av_max_runtime_scenarios,
        )

    if not args.av_skip_report:
        report["report"] = av_report_script.build_report(
            processed_dir=args.av_processed_dir,
            reports_dir=args.av_reports_dir,
            models_dir=args.av_models_dir,
            uncertainty_dir=args.av_uncertainty_dir,
        )

    return report


def run_battery_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    return battery_script.run_pipeline(
        out_dir=args.battery_out_dir,
        model_dir=args.battery_model_dir,
        paper_table_dir=args.battery_paper_table_dir,
        paper_fig_dir=args.battery_paper_fig_dir,
        features_path=args.battery_features,
        engineered_baseline_path=args.battery_engineered_baseline,
        deep_oqe_epochs=args.battery_deep_oqe_epochs,
        forecast_epochs=args.battery_forecast_epochs,
        batch_size=args.battery_batch_size,
        seq_len=args.battery_seq_len,
        lookback=args.battery_lookback,
        horizon=args.battery_horizon,
        train_stride=args.battery_train_stride,
        eval_stride=args.battery_eval_stride,
    )


def main() -> int:
    args = _parse_args()
    overall_dir = args.overall_dir
    overall_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domains": {},
    }
    summary_rows: list[dict[str, object]] = []

    if not args.skip_battery:
        battery_report = run_battery_pipeline(args)
        combined["domains"]["battery"] = battery_report

    if not args.skip_av:
        av_report = run_av_pipeline(args)
        combined["domains"]["av"] = av_report
    closure_report = closure_script.build_closure(
        battery_dir=args.battery_out_dir,
        av_dir=args.av_reports_dir,
        overall_dir=overall_dir,
        submission_scope=args.submission_scope,
    )
    combined["closure"] = closure_report
    monograph_assets_script.build(submission_scope=args.submission_scope)
    ieee_assets_script.build()
    combined["publication_assets"] = {
        "submission_scope": args.submission_scope,
        "parity_matrix": str(REPO_ROOT / "reports" / "publication" / "orius_equal_domain_parity_matrix.csv"),
        "domain_closure_matrix": str(REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv"),
        "maturity_matrix": str(REPO_ROOT / "reports" / "publication" / "orius_maturity_matrix.csv"),
        "executive_summary": str(REPO_ROOT / "docs" / "executive_summary.md"),
        "claim_ledger": str(REPO_ROOT / "docs" / "claim_ledger.md"),
    }

    if not args.skip_battery:
        summary_rows.append(
            {
                "domain": "battery",
                "status": closure_report["battery"]["status"],
                "out_dir": str(args.battery_out_dir),
                "key_report": str(args.battery_out_dir / "summary.json"),
                "notes": "witness_row",
            }
        )
    if not args.skip_av:
        summary_rows.append(
            {
                "domain": "av",
                "status": closure_report["av"]["status"],
                "out_dir": str(args.av_reports_dir),
                "key_report": str(args.av_reports_dir / "summary.json"),
                "notes": av_report["subset_mode"],
            }
        )

    combined["domain_count"] = len(combined["domains"])
    combined["summary_csv"] = str(overall_dir / "domain_summary.csv")

    artifact_paths: set[Path] = set()
    _collect_file_paths(combined["domains"], artifact_paths)
    manifest = {
        "created_at_utc": combined["created_at_utc"],
        "domains": sorted(combined["domains"].keys()),
        "artifacts": {str(path): _sha256_file(path) for path in sorted(artifact_paths) if not _is_appledouble(path)},
        "input_hashes": _input_hashes(args),
    }

    _write_summary_csv(Path(combined["summary_csv"]), summary_rows)
    manifest_path = overall_dir / "battery_av_manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    existing_manifest.setdefault("artifacts", {})
    existing_manifest["artifacts"].update(manifest["artifacts"])
    existing_manifest.setdefault("pipeline_artifacts", {})
    existing_manifest["pipeline_artifacts"].update(manifest["artifacts"])
    existing_manifest["input_hashes"] = manifest["input_hashes"]
    manifest_path.write_text(json.dumps(existing_manifest, indent=2), encoding="utf-8")
    combined["manifest"] = str(manifest_path)

    summary_path = overall_dir / "battery_av_pipeline.json"
    summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
