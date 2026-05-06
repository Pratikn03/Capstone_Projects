#!/usr/bin/env python3
"""Unified release orchestrator: one command, one splits hash, all 10 models.

Pipeline (each stage is independently skippable, all status captured in the
release manifest):

    1. preflight   - verify features parquet, deps, output dirs
    2. carve       - deterministic train/cal/val/test from features.parquet
                     using the canonical config, write SHA256s
    3. legacy      - subprocess `orius.forecasting.train` for GBM/LSTM/TCN/
                     N-BEATS/TFT/PatchTST against the carved features
    4. advanced    - in-process `train_advanced.run_advanced_baselines` for
                     Prophet/NGBoost/Darts-N-BEATS/FLAML on the carved splits
    5. extract     - stage per-model .npz predictions for significance
    6. table       - rebuild the publication baseline comparison table
    7. significance - DM + paired bootstrap + Holm across the 10-model family
    8. manifest    - write release_manifest.json with every integrity field

The final manifest at
``artifacts/runs/{region}/{release_id}/release_manifest.json`` is the single
artifact a reviewer reads to answer "did all models see the same data and is
this run reproducible?".
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.forecasting.train_advanced import (
    AdvancedTrainerConfig,
    SplitPaths,
    run_advanced_baselines,
)
from orius.release.extract_predictions import (
    LEGACY_MODEL_KEYS,
    link_advanced_predictions,
    materialize_legacy_predictions,
)
from orius.release.manifest import (
    ReleaseManifest,
    StepStatus,
    collect_environment,
    utc_now_iso,
    write_release_manifest,
)
from orius.release.splits import (
    carve_splits,
    splits_config_from_yaml,
)

logger = logging.getLogger("release")

DEFAULT_TARGETS = ("load_mw", "wind_mw", "solar_mw")
DEFAULT_SEEDS = (42, 123, 456, 789, 2024, 1337, 7777, 9999)
DEFAULT_ADVANCED_MODELS = ("prophet", "ngboost", "nbeats_darts", "flaml")
DEFAULT_BASELINES_FOR_SIGNIFICANCE = (
    "lstm",
    "tcn",
    "nbeats",
    "tft",
    "patchtst",
    "prophet",
    "nbeats_darts",
    "ngboost",
    "flaml",
)


def _start_step(name: str) -> tuple[StepStatus, float]:
    started = utc_now_iso()
    return StepStatus(name=name, status="running", started_at=started), time.perf_counter()


def _finish_step(
    step: StepStatus,
    t0: float,
    *,
    status: str,
    detail: dict[str, Any] | None = None,
    error: str | None = None,
) -> StepStatus:
    step.status = status
    step.finished_at = utc_now_iso()
    step.duration_seconds = round(time.perf_counter() - t0, 3)
    if detail:
        step.detail.update(detail)
    if error:
        step.error = error
    return step


def _resolve_region_paths(*, region: str, out_root: Path, release_id: str) -> dict[str, Path]:
    region_lower = region.lower()
    base = out_root / "artifacts" / "runs" / region_lower / release_id
    return {
        "release_root": base,
        "splits_dir": base / "splits",
        "models_dir": base / "models",
        "reports_dir": base / "reports",
        "uncertainty_dir": base / "uncertainty",
        "backtests_dir": base / "backtests",
        "advanced_runs_dir": base / "advanced_baselines",
        "predictions_dir": base / "predictions",
        "logs_dir": base / "logs",
        "manifest_path": base / "release_manifest.json",
        "metrics_json": base / "reports" / "week2_metrics.json",
    }


def _preflight(features_path: Path, config_path: Path, paths: dict[str, Path]) -> dict[str, Any]:
    if not features_path.exists():
        raise FileNotFoundError(f"features parquet missing: {features_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config missing: {config_path}")
    for key in (
        "splits_dir",
        "models_dir",
        "reports_dir",
        "uncertainty_dir",
        "backtests_dir",
        "advanced_runs_dir",
        "predictions_dir",
        "logs_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    return {
        "features_path": str(features_path),
        "config_path": str(config_path),
        "release_root": str(paths["release_root"]),
    }


def _run_legacy_trainer(
    *,
    config_path: Path,
    features_path: Path,
    paths: dict[str, Path],
    targets: tuple[str, ...],
    seed: int,
    log_path: Path,
    extra_args: list[str],
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "orius.forecasting.train",
        "--config",
        str(config_path),
        "--features",
        str(features_path),
        "--out-dir",
        str(paths["models_dir"]),
        "--reports-dir",
        str(paths["reports_dir"]),
        "--seed",
        str(int(seed)),
        "--targets",
        ",".join(targets),
    ]
    cmd.extend(extra_args)
    env = {"PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONHASHSEED": "0"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"# command: {' '.join(cmd)}\n")
        proc = subprocess.run(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**env, "PATH": "/usr/bin:/bin"},
            check=False,
            cwd=str(REPO_ROOT),
        )
    return {"command": cmd, "returncode": int(proc.returncode), "log_path": str(log_path)}


def _run_advanced(
    *,
    region: str,
    release_id: str,
    paths: dict[str, Path],
    splits_paths: SplitPaths,
    targets: tuple[str, ...],
    seeds: tuple[int, ...],
    horizon: int,
    lookback: int,
    alpha: float,
    holiday_country: str | None,
    enabled_models: tuple[str, ...],
    flaml_time_budget: int,
    out_root: Path,
) -> dict[str, Any]:
    cfg = AdvancedTrainerConfig(
        region=region,
        release_id=release_id,
        splits=splits_paths,
        out_root=out_root,
        targets=targets,
        seeds=seeds,
        horizon=horizon,
        lookback=lookback,
        alpha=alpha,
        holiday_country=holiday_country,
        metrics_json=paths["metrics_json"],
        conformal_dir=paths["uncertainty_dir"],
        enabled_models=enabled_models,
        flaml_time_budget=flaml_time_budget,
    )
    summary = run_advanced_baselines(cfg)
    return summary


def _run_table_builder(*, release_id: str, paths: dict[str, Path], log_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_baseline_comparison_table.py"),
        "--release-id",
        release_id,
        "--out-dir",
        str(paths["release_root"]),
    ]
    env = {"PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"# command: {' '.join(cmd)}\n")
        proc = subprocess.run(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, check=False, cwd=str(REPO_ROOT)
        )
    return {"command": cmd, "returncode": int(proc.returncode), "log_path": str(log_path)}


def _run_significance(
    *,
    region: str,
    release_id: str,
    paths: dict[str, Path],
    targets: tuple[str, ...],
    reference: str,
    baselines: tuple[str, ...],
    horizon: int,
    n_resamples: int,
    seed: int,
    log_path: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_baseline_significance.py"),
        "--region",
        region,
        "--release-id",
        release_id,
        "--predictions-dir",
        str(paths["predictions_dir"]),
        "--out-dir",
        str(paths["release_root"]),
        "--targets",
        *targets,
        "--reference",
        reference,
        "--baselines",
        *baselines,
        "--horizon",
        str(int(horizon)),
        "--n-resamples",
        str(int(n_resamples)),
        "--seed",
        str(int(seed)),
    ]
    env = {"PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"# command: {' '.join(cmd)}\n")
        proc = subprocess.run(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, check=False, cwd=str(REPO_ROOT)
        )
    return {"command": cmd, "returncode": int(proc.returncode), "log_path": str(log_path)}


def run_release(
    *,
    region: str,
    release_id: str,
    features_path: Path,
    config_path: Path,
    out_root: Path,
    targets: tuple[str, ...],
    seeds: tuple[int, ...],
    advanced_models: tuple[str, ...],
    significance_baselines: tuple[str, ...],
    holiday_country: str | None,
    legacy_seed: int,
    skip_legacy: bool,
    skip_advanced: bool,
    skip_table: bool,
    skip_significance: bool,
    n_resamples: int,
    flaml_time_budget: int,
    legacy_extra_args: list[str],
) -> ReleaseManifest:
    paths = _resolve_region_paths(region=region, out_root=out_root, release_id=release_id)
    manifest = ReleaseManifest(
        region=region,
        release_id=release_id,
        started_at=utc_now_iso(),
        environment=collect_environment(),
    )

    step, t0 = _start_step("preflight")
    try:
        info = _preflight(features_path, config_path, paths)
        manifest.inputs.update(info)
        manifest.steps.append(_finish_step(step, t0, status="ok", detail=info))
    except Exception as exc:
        manifest.steps.append(_finish_step(step, t0, status="failed", error=str(exc)))
        manifest.finished_at = utc_now_iso()
        write_release_manifest(manifest, paths["manifest_path"])
        raise

    step, t0 = _start_step("carve_splits")
    try:
        cfg_dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        splits_cfg = splits_config_from_yaml(cfg_dict)
        carved = carve_splits(features_path=features_path, out_dir=paths["splits_dir"], cfg=splits_cfg)
        manifest.inputs.update(
            {
                "features_sha256": carved.features_sha256,
                "splits_sha256": carved.splits_sha256,
                "splits_boundaries": carved.boundaries,
                "splits_paths": {
                    "train": str(carved.train_path),
                    "calibration": str(carved.calibration_path),
                    "val": str(carved.val_path),
                    "test": str(carved.test_path),
                },
            }
        )
        manifest.steps.append(
            _finish_step(
                step,
                t0,
                status="ok",
                detail={
                    "n_rows": carved.n_rows,
                    "splits_sha256": carved.splits_sha256,
                    "features_sha256": carved.features_sha256,
                },
            )
        )
    except Exception as exc:
        manifest.steps.append(_finish_step(step, t0, status="failed", error=str(exc)))
        manifest.finished_at = utc_now_iso()
        write_release_manifest(manifest, paths["manifest_path"])
        raise

    if not skip_legacy:
        step, t0 = _start_step("legacy_trainer")
        try:
            log_path = paths["logs_dir"] / "legacy_train.log"
            result = _run_legacy_trainer(
                config_path=config_path,
                features_path=features_path,
                paths=paths,
                targets=targets,
                seed=legacy_seed,
                log_path=log_path,
                extra_args=legacy_extra_args,
            )
            status = "ok" if result["returncode"] == 0 else "failed"
            manifest.steps.append(_finish_step(step, t0, status=status, detail=result))
        except Exception as exc:
            manifest.steps.append(_finish_step(step, t0, status="failed", error=str(exc)))
    else:
        manifest.steps.append(StepStatus(name="legacy_trainer", status="skipped"))

    if not skip_advanced:
        step, t0 = _start_step("advanced_trainer")
        try:
            splits_paths = SplitPaths(
                train=carved.train_path,
                calibration=carved.calibration_path,
                test=carved.test_path,
            )
            summary = _run_advanced(
                region=region,
                release_id=release_id,
                paths=paths,
                splits_paths=splits_paths,
                targets=targets,
                seeds=seeds,
                horizon=splits_cfg.horizon,
                lookback=splits_cfg.lookback,
                alpha=0.10,
                holiday_country=holiday_country,
                enabled_models=advanced_models,
                flaml_time_budget=flaml_time_budget,
                out_root=out_root,
            )
            advanced_runs_dir = paths["release_root"] / "advanced_baselines"
            staged = link_advanced_predictions(
                advanced_runs_dir=advanced_runs_dir,
                predictions_dir=paths["predictions_dir"],
            )
            ok = any(
                block.get("status") == "ok"
                for blocks in summary.get("models", {}).values()
                for block in blocks.values()
            )
            manifest.steps.append(
                _finish_step(
                    step,
                    t0,
                    status="ok" if ok else "failed",
                    detail={"models_summary": summary, "predictions_staged": staged},
                )
            )
        except Exception as exc:
            manifest.steps.append(_finish_step(step, t0, status="failed", error=str(exc)))
    else:
        manifest.steps.append(StepStatus(name="advanced_trainer", status="skipped"))

    step, t0 = _start_step("extract_predictions")
    try:
        legacy_extracted = materialize_legacy_predictions(
            backtests_dir=paths["backtests_dir"],
            predictions_dir=paths["predictions_dir"],
            targets=targets,
            seed=legacy_seed,
            models=LEGACY_MODEL_KEYS,
        )
        manifest.steps.append(
            _finish_step(
                step,
                t0,
                status="ok",
                detail={"legacy_predictions_written": len(legacy_extracted)},
            )
        )
    except Exception as exc:
        manifest.steps.append(_finish_step(step, t0, status="failed", error=str(exc)))

    if not skip_table:
        step, t0 = _start_step("publication_table")
        try:
            log_path = paths["logs_dir"] / "table_builder.log"
            result = _run_table_builder(release_id=release_id, paths=paths, log_path=log_path)
            manifest.steps.append(
                _finish_step(
                    step,
                    t0,
                    status="ok" if result["returncode"] == 0 else "failed",
                    detail=result,
                )
            )
        except Exception as exc:
            manifest.steps.append(_finish_step(step, t0, status="failed", error=str(exc)))
    else:
        manifest.steps.append(StepStatus(name="publication_table", status="skipped"))

    if not skip_significance:
        step, t0 = _start_step("significance")
        try:
            log_path = paths["logs_dir"] / "significance.log"
            result = _run_significance(
                region=region,
                release_id=release_id,
                paths=paths,
                targets=targets,
                reference="gbm",
                baselines=significance_baselines,
                horizon=splits_cfg.horizon,
                n_resamples=n_resamples,
                seed=legacy_seed,
                log_path=log_path,
            )
            manifest.steps.append(
                _finish_step(
                    step,
                    t0,
                    status="ok" if result["returncode"] == 0 else "failed",
                    detail=result,
                )
            )
        except Exception as exc:
            manifest.steps.append(_finish_step(step, t0, status="failed", error=str(exc)))
    else:
        manifest.steps.append(StepStatus(name="significance", status="skipped"))

    manifest.artifacts.update(
        {
            "release_root": str(paths["release_root"]),
            "splits_manifest": str(paths["splits_dir"] / "splits_manifest.json"),
            "metrics_json": str(paths["metrics_json"]),
            "predictions_dir": str(paths["predictions_dir"]),
            "publication_table_csv": str(paths["release_root"] / "baseline_comparison_all.csv"),
            "publication_table_tex": str(paths["release_root"] / "baseline_comparison_all.tex"),
            "significance_csv": str(paths["release_root"] / "baseline_significance.csv"),
        }
    )
    manifest.summary = {
        "n_steps": len(manifest.steps),
        "n_failed_steps": sum(1 for s in manifest.steps if s.status == "failed"),
        "n_skipped_steps": sum(1 for s in manifest.steps if s.status == "skipped"),
        "n_ok_steps": sum(1 for s in manifest.steps if s.status == "ok"),
    }
    manifest.finished_at = utc_now_iso()
    write_release_manifest(manifest, paths["manifest_path"])
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified release orchestrator: train every model on identical splits and publish."
    )
    p.add_argument("--region", required=True, help="DE, US, AV, etc.")
    p.add_argument("--release-id", required=True)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out-root", default=REPO_ROOT, type=Path)
    p.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--advanced-models", nargs="+", default=list(DEFAULT_ADVANCED_MODELS))
    p.add_argument("--significance-baselines", nargs="+", default=list(DEFAULT_BASELINES_FOR_SIGNIFICANCE))
    p.add_argument("--holiday-country", default=None)
    p.add_argument("--legacy-seed", type=int, default=42)
    p.add_argument("--skip-legacy", action="store_true")
    p.add_argument("--skip-advanced", action="store_true")
    p.add_argument("--skip-table", action="store_true")
    p.add_argument("--skip-significance", action="store_true")
    p.add_argument("--n-resamples", type=int, default=10_000)
    p.add_argument("--flaml-time-budget", type=int, default=600)
    p.add_argument("--legacy-extra", nargs=argparse.REMAINDER, default=[])
    p.add_argument("--smoke", action="store_true", help="One target, one seed, advanced-only, fast.")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    targets = tuple(args.targets)
    seeds = tuple(args.seeds)
    advanced_models = tuple(args.advanced_models)
    significance_baselines = tuple(args.significance_baselines)
    skip_legacy = args.skip_legacy
    n_resamples = int(args.n_resamples)
    flaml_budget = int(args.flaml_time_budget)
    if args.smoke:
        targets = (targets[0],)
        seeds = (seeds[0],)
        advanced_models = ("ngboost",)
        significance_baselines = ("ngboost",)
        skip_legacy = True
        n_resamples = 500
        flaml_budget = 30

    legacy_extra = list(args.legacy_extra)
    if legacy_extra and legacy_extra[0] == "--":
        legacy_extra = legacy_extra[1:]

    manifest = run_release(
        region=str(args.region),
        release_id=str(args.release_id),
        features_path=Path(args.features).resolve(),
        config_path=Path(args.config).resolve(),
        out_root=Path(args.out_root).resolve(),
        targets=targets,
        seeds=seeds,
        advanced_models=advanced_models,
        significance_baselines=significance_baselines,
        holiday_country=args.holiday_country,
        legacy_seed=int(args.legacy_seed),
        skip_legacy=skip_legacy,
        skip_advanced=args.skip_advanced,
        skip_table=args.skip_table,
        skip_significance=args.skip_significance,
        n_resamples=n_resamples,
        flaml_time_budget=flaml_budget,
        legacy_extra_args=legacy_extra,
    )
    print(json.dumps(manifest.summary, indent=2))
    failed = manifest.summary.get("n_failed_steps", 0)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
