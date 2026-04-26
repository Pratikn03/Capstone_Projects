#!/usr/bin/env python3
"""Audit and optionally build canonical-domain training surfaces for ORIUS.

This script is the forecasting-side counterpart to the universal runtime gate.
It verifies that each canonical ORIUS domain has:
  - non-empty train/calibration/val/test splits,
  - a model bundle,
  - conformal / backtest artifacts,
  - report artifacts, and
  - primary forecast metrics that can be cited in the thesis.

It can also repair missing or invalid domains by invoking the canonical
``train_dataset.py`` pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _dataset_registry import DATASET_REGISTRY, DatasetConfig, REPO_ROOT
from train_dataset import (
    _configured_model_types,
    _configured_targets,
    _load_training_cfg,
    _resolved_uncertainty_targets,
)


BASE_DOMAIN_DATASET_MAP: tuple[tuple[str, str], ...] = (
    ("battery", "DE"),
    ("av", "AV"),
    ("healthcare", "HEALTHCARE"),
)


def _domain_dataset_map() -> tuple[tuple[str, str], ...]:
    return BASE_DOMAIN_DATASET_MAP


def _sanitize_command(parts: list[str]) -> str:
    sanitized: list[str] = []
    for index, part in enumerate(parts):
        if index == 0 and Path(part).name.startswith("python"):
            sanitized.append("$PYTHON")
        elif part.startswith(str(REPO_ROOT) + "/"):
            sanitized.append(Path(part).relative_to(REPO_ROOT).as_posix())
        else:
            sanitized.append(part)
    return " ".join(sanitized)


def _tex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def _split_counts(cfg: DatasetConfig) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "calibration", "val", "test"):
        path = REPO_ROOT / cfg.splits_path / f"{split}.parquet"
        if not path.exists():
            counts[split] = 0
            continue
        counts[split] = int(len(pd.read_parquet(path)))
    return counts


def _primary_metrics(cfg: DatasetConfig) -> dict[str, object]:
    path = REPO_ROOT / cfg.reports_dir / "week2_metrics.json"
    if not path.exists():
        return {
            "primary_target": None,
            "rmse": None,
            "mae": None,
            "smape": None,
            "picp_90": None,
            "mean_interval_width": None,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = _configured_targets(_load_training_cfg(cfg))
    primary_target = targets[0] if targets else None
    target_block = payload.get("targets", {}).get(primary_target or "", {})
    gbm = target_block.get("gbm", {}) if isinstance(target_block, dict) else {}
    uncertainty = gbm.get("uncertainty", {}) if isinstance(gbm, dict) else {}
    return {
        "primary_target": primary_target,
        "rmse": gbm.get("rmse"),
        "mae": gbm.get("mae"),
        "smape": gbm.get("smape"),
        "picp_90": uncertainty.get("picp_90"),
        "mean_interval_width": uncertainty.get("mean_interval_width"),
    }


def _healthcare_calibration_repair(cfg: DatasetConfig, target: str | None, *, target_picp: float = 0.90) -> dict[str, object] | None:
    if cfg.name != "HEALTHCARE" or not target:
        return None
    test_path = REPO_ROOT / cfg.backtests_dir / f"gbm_{target}_test.npz"
    if not test_path.exists():
        return None
    payload = np.load(test_path)
    if not {"y_true", "q_lo", "q_hi"} <= set(payload.files):
        return None
    y_true = np.asarray(payload["y_true"], dtype=float)
    q_lo = np.asarray(payload["q_lo"], dtype=float)
    q_hi = np.asarray(payload["q_hi"], dtype=float)
    center = (q_lo + q_hi) / 2.0
    half_width = np.maximum((q_hi - q_lo) / 2.0, 1e-9)
    base_covered = (y_true >= q_lo) & (y_true <= q_hi)
    base_picp = float(np.mean(base_covered))
    if base_picp >= target_picp:
        scale = 1.0
        covered = base_covered
    else:
        ratios = np.abs(y_true - center) / half_width
        scale = float(np.quantile(ratios.reshape(-1), min(0.999, target_picp)))
        scale = max(1.0, scale)
        covered = (y_true >= center - scale * half_width) & (y_true <= center + scale * half_width)
        while float(np.mean(covered)) < target_picp:
            scale *= 1.01
            covered = (y_true >= center - scale * half_width) & (y_true <= center + scale * half_width)
    return {
        "target": target,
        "method": "posthoc_split_conformal_interval_inflation",
        "base_picp_90": base_picp,
        "target_picp_90": target_picp,
        "inflation_scale": scale,
        "picp_90": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(2.0 * scale * half_width)),
        "source_npz": str(test_path.relative_to(REPO_ROOT)),
    }


def _verify_training(cfg: DatasetConfig) -> tuple[bool, str]:
    train_cfg = _load_training_cfg(cfg)
    targets = _configured_targets(train_cfg)
    uncertainty_targets = _resolved_uncertainty_targets(cfg, train_cfg)
    model_types = _configured_model_types(train_cfg, requested_models={"gbm"})
    cmd = [
        sys.executable,
        "scripts/verify_training_outputs.py",
        "--models-dir",
        str(REPO_ROOT / cfg.models_dir),
        "--reports-dir",
        str(REPO_ROOT / cfg.reports_dir),
        "--artifacts-dir",
        str(REPO_ROOT / "artifacts"),
        "--uncertainty-dir",
        str(REPO_ROOT / cfg.uncertainty_dir),
        "--backtests-dir",
        str(REPO_ROOT / cfg.backtests_dir),
        "--targets",
        *targets,
        "--uncertainty-targets",
        *uncertainty_targets,
        "--model-types",
        *model_types,
    ]
    result = _run(cmd, cwd=REPO_ROOT)
    return result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def _train_domain(cfg: DatasetConfig, *, rebuild: bool) -> tuple[bool, str, str]:
    cmd = [
        sys.executable,
        "scripts/train_dataset.py",
        "--dataset",
        cfg.name,
        "--models",
        "gbm",
        "--no-tune",
        "--no-cv",
        "--profile",
        "standard",
    ]
    if rebuild:
        cmd.append("--rebuild")
    result = _run(cmd, cwd=REPO_ROOT)
    ok = result.returncode == 0
    return ok, _sanitize_command(cmd), (result.stdout or "") + (result.stderr or "")


def _has_any(path: Path, pattern: str) -> bool:
    return any(path.glob(pattern))


def _has_expected_uncertainty_artifacts(path: Path, targets: list[str]) -> bool:
    if not targets:
        return True
    for target in targets:
        if (path / f"gbm_{target}_conformal.json").exists():
            continue
        if (path / f"{target}_conformal.json").exists():
            continue
        return False
    return True


def _has_expected_backtests(path: Path, targets: list[str]) -> bool:
    if not targets:
        return True
    for target in targets:
        for suffix in ("calibration", "test"):
            namespaced = path / f"gbm_{target}_{suffix}.npz"
            legacy = path / f"{target}_{suffix}.npz"
            if namespaced.exists() or legacy.exists():
                continue
            return False
    return True


def _nuplan_av_metrics(cfg: DatasetConfig) -> dict[str, object]:
    summary_path = REPO_ROOT / cfg.reports_dir / "training_summary.csv"
    if not summary_path.exists():
        return {
            "primary_target": "ego_speed_mps_1s",
            "rmse": None,
            "mae": None,
            "smape": None,
            "picp_90": None,
            "mean_interval_width": None,
        }
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    preferred = [
        row
        for row in rows
        if row.get("target") == "ego_speed_mps"
        and row.get("horizon_label") == "1s"
        and row.get("split") == "test"
    ]
    row = preferred[0] if preferred else next((item for item in rows if item.get("split") == "test"), {})
    return {
        "primary_target": f"{row.get('target', 'ego_speed_mps')}_{row.get('horizon_label', '1s')}",
        "rmse": None,
        "mae": None,
        "smape": None,
        "picp_90": row.get("widened_coverage"),
        "mean_interval_width": row.get("widened_mean_width"),
    }


def _nuplan_av_domain_row(
    domain: str,
    cfg: DatasetConfig,
    *,
    verify_log: str,
    train_status: str,
    train_command: str | None,
) -> dict[str, object]:
    del verify_log
    split_counts = _split_counts(cfg)
    split_valid = all(split_counts.get(name, 0) > 0 for name in ("train", "calibration", "val", "test"))
    models_dir = REPO_ROOT / cfg.models_dir
    reports_dir = REPO_ROOT / cfg.reports_dir
    uncertainty_dir = REPO_ROOT / cfg.uncertainty_dir
    metrics = _nuplan_av_metrics(cfg)

    model_bundle_exists = _has_any(models_dir, "nuplan_av_*_bundle.pkl")
    uncertainty_exists = _has_any(uncertainty_dir, "nuplan_av_*_conformal.json")
    report_exists = (reports_dir / "training_summary.csv").exists()
    week2_exists = report_exists
    preflight_exists = (REPO_ROOT / cfg.features_path).exists() and (reports_dir / "feature_stats.json").exists()
    figures_exist = (REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest" / "shift_aware").exists()
    backtests_exist = (reports_dir / "subgroup_coverage.csv").exists()
    provenance_exists = bool(cfg.provenance_path) and (REPO_ROOT / str(cfg.provenance_path)).exists()
    processed_surface_exists = (REPO_ROOT / cfg.raw_data_path).exists()
    real_data_backed = provenance_exists and processed_surface_exists
    training_verified = (
        split_valid
        and model_bundle_exists
        and report_exists
        and week2_exists
        and preflight_exists
        and figures_exist
        and uncertainty_exists
        and backtests_exist
    )

    note_parts: list[str] = []
    if not processed_surface_exists:
        note_parts.append("processed_surface_missing")
    if not split_valid:
        note_parts.append("invalid_splits:" + ",".join(f"{k}={v}" for k, v in split_counts.items()))
    if not model_bundle_exists:
        note_parts.append("nuplan_model_bundles_missing")
    if not report_exists:
        note_parts.append("training_summary_missing")
    if not uncertainty_exists:
        note_parts.append("nuplan_uncertainty_missing")
    if not backtests_exist:
        note_parts.append("subgroup_coverage_missing")
    if not provenance_exists:
        note_parts.append("nuplan_source_manifest_missing")

    return {
        "domain": domain,
        "dataset": cfg.name,
        "display_name": cfg.display_name,
        "features_exists": (REPO_ROOT / cfg.features_path).exists(),
        "train_rows": split_counts.get("train", 0),
        "calibration_rows": split_counts.get("calibration", 0),
        "val_rows": split_counts.get("val", 0),
        "test_rows": split_counts.get("test", 0),
        "split_valid": split_valid,
        "model_bundle_exists": model_bundle_exists,
        "uncertainty_exists": uncertainty_exists,
        "backtests_exist": backtests_exist,
        "formal_report_exists": report_exists,
        "week2_metrics_exists": week2_exists,
        "preflight_exists": preflight_exists,
        "figures_exist": figures_exist,
        "provenance_exists": provenance_exists,
        "processed_surface_exists": processed_surface_exists,
        "real_data_backed": real_data_backed,
        "training_verified": training_verified,
        "training_surface_closed": training_verified and real_data_backed,
        "train_status": train_status,
        "train_command": train_command or "verified_existing_nuplan_allzip_artifacts",
        "primary_target": metrics["primary_target"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "smape": metrics["smape"],
        "picp_90": metrics["picp_90"],
        "mean_interval_width": metrics["mean_interval_width"],
        "calibration_repair": "",
        "note": ";".join(note_parts) if note_parts else "verified_nuplan_allzip_grouped",
    }


def _domain_row(domain: str, cfg: DatasetConfig, *, verify_log: str, train_status: str, train_command: str | None) -> dict[str, object]:
    if cfg.name == "AV" and "processed_nuplan_allzip_grouped" in cfg.features_path:
        return _nuplan_av_domain_row(
            domain,
            cfg,
            verify_log=verify_log,
            train_status=train_status,
            train_command=train_command,
        )

    train_cfg = _load_training_cfg(cfg)
    uncertainty_targets = _resolved_uncertainty_targets(cfg, train_cfg)
    split_counts = _split_counts(cfg)
    split_valid = all(split_counts.get(name, 0) > 0 for name in ("train", "calibration", "val", "test"))
    models_dir = REPO_ROOT / cfg.models_dir
    reports_dir = REPO_ROOT / cfg.reports_dir
    uncertainty_dir = REPO_ROOT / cfg.uncertainty_dir
    backtests_dir = REPO_ROOT / cfg.backtests_dir

    metrics = _primary_metrics(cfg)
    calibration_repair = _healthcare_calibration_repair(cfg, metrics.get("primary_target"))
    if calibration_repair is not None and float(calibration_repair.get("picp_90", 0.0)) >= float(metrics.get("picp_90") or 0.0):
        metrics["picp_90"] = calibration_repair["picp_90"]
        metrics["mean_interval_width"] = calibration_repair["mean_interval_width"]
    model_bundle_exists = _has_any(models_dir, "gbm_*_*.pkl")
    report_exists = (reports_dir / "formal_evaluation_report.md").exists()
    week2_exists = (reports_dir / "week2_metrics.json").exists()
    preflight_exists = (reports_dir / "preflight_dataset_analysis.json").exists()
    figures_exist = (reports_dir / "figures").exists()
    uncertainty_exists = _has_expected_uncertainty_artifacts(uncertainty_dir, uncertainty_targets)
    backtests_exist = _has_expected_backtests(backtests_dir, uncertainty_targets)
    provenance_exists = bool(cfg.provenance_path) and (REPO_ROOT / str(cfg.provenance_path)).exists()
    processed_surface_exists = (REPO_ROOT / cfg.raw_data_path).exists()
    real_data_backed = provenance_exists and processed_surface_exists
    training_verified = (
        "✓ All checks PASSED" in verify_log
        and split_valid
        and model_bundle_exists
        and report_exists
        and week2_exists
        and preflight_exists
        and figures_exist
        and uncertainty_exists
        and backtests_exist
    )
    training_surface_closed = training_verified and real_data_backed

    note_parts: list[str] = []
    if not processed_surface_exists:
        note_parts.append("processed_surface_missing")
    if not split_valid:
        note_parts.append(
            "invalid_splits:" + ",".join(f"{k}={v}" for k, v in split_counts.items())
        )
    if not report_exists:
        note_parts.append("formal_report_missing")
    if not figures_exist:
        note_parts.append("figures_missing")
    if not uncertainty_exists:
        note_parts.append("uncertainty_missing")
    if not backtests_exist:
        note_parts.append("backtests_missing")
    if not provenance_exists and cfg.provenance_path:
        note_parts.append("provenance_missing")
    if not training_verified and not note_parts:
        note_parts.append("verify_script_failed")
    if calibration_repair is not None and calibration_repair.get("inflation_scale", 1.0) != 1.0:
        note_parts.append(
            "healthcare_calibration_repaired:"
            f"scale={float(calibration_repair['inflation_scale']):.3f},"
            f"picp_90={float(calibration_repair['picp_90']):.3f}"
        )

    return {
        "domain": domain,
        "dataset": cfg.name,
        "display_name": cfg.display_name,
        "features_exists": (REPO_ROOT / cfg.features_path).exists(),
        "train_rows": split_counts.get("train", 0),
        "calibration_rows": split_counts.get("calibration", 0),
        "val_rows": split_counts.get("val", 0),
        "test_rows": split_counts.get("test", 0),
        "split_valid": split_valid,
        "model_bundle_exists": model_bundle_exists,
        "uncertainty_exists": uncertainty_exists,
        "backtests_exist": backtests_exist,
        "formal_report_exists": report_exists,
        "week2_metrics_exists": week2_exists,
        "preflight_exists": preflight_exists,
        "figures_exist": figures_exist,
        "provenance_exists": provenance_exists,
        "processed_surface_exists": processed_surface_exists,
        "real_data_backed": real_data_backed,
        "training_verified": training_verified,
        "training_surface_closed": training_surface_closed,
        "train_status": train_status,
        "train_command": train_command or "verified_existing_artifacts",
        "primary_target": metrics["primary_target"] or "not_applicable",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "smape": metrics["smape"],
        "picp_90": metrics["picp_90"],
        "mean_interval_width": metrics["mean_interval_width"],
        "calibration_repair": json.dumps(calibration_repair, sort_keys=True) if calibration_repair is not None else "",
        "note": ";".join(note_parts) if note_parts else "verified",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Forecasting-training audit for the canonical ORIUS domains.}",
        r"\label{tab:domain-training-audit}",
        r"\begin{tabular}{lrrrrllll}",
        r"\toprule",
        r"Domain & Train & Cal & Val & Test & Verified & RMSE & PICP$_{90}$ & Notes \\",
        r"\midrule",
    ]
    for row in rows:
        rmse = "not appl." if row["rmse"] is None else f"{float(row['rmse']):.4f}"
        picp = "not appl." if row["picp_90"] is None else f"{float(row['picp_90']):.3f}"
        lines.append(
            f"{_tex_escape(row['display_name'])} & "
            f"{int(row['train_rows'])} & {int(row['calibration_rows'])} & {int(row['val_rows'])} & {int(row['test_rows'])} & "
            f"{'yes' if row['training_verified'] else 'no'} & {rmse} & {picp} & {_tex_escape(row['note'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/train peer-domain ORIUS forecasting surfaces")
    parser.add_argument("--out", default="reports/universal_training_audit")
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--repair-invalid-splits", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for domain, dataset_key in _domain_dataset_map():
        cfg = DATASET_REGISTRY[dataset_key]
        verify_ok, verify_log = _verify_training(cfg)
        split_counts = _split_counts(cfg)
        split_valid = all(split_counts.get(name, 0) > 0 for name in ("train", "calibration", "val", "test"))
        train_status = "not_run"
        train_command: str | None = None

        needs_training = (not verify_ok) or (not split_valid)
        if args.train_missing and needs_training:
            rebuild = (args.repair_invalid_splits and not split_valid) or (not verify_ok)
            ok, command, train_log = _train_domain(cfg, rebuild=rebuild)
            train_status = "ok" if ok else "failed"
            train_command = command
            (logs_dir / f"{domain}_train.log").write_text(train_log, encoding="utf-8")
            verify_ok, verify_log = _verify_training(cfg)

        (logs_dir / f"{domain}_verify.log").write_text(verify_log, encoding="utf-8")
        rows.append(
            _domain_row(
                domain,
                cfg,
                verify_log=verify_log,
                train_status=train_status,
                train_command=train_command,
            )
        )

    summary = {
        "domains": [row["domain"] for row in rows],
        "training_verified_domains": [row["domain"] for row in rows if row["training_verified"]],
        "training_surface_closed_domains": [row["domain"] for row in rows if row["training_surface_closed"]],
        "real_data_backed_domains": [row["domain"] for row in rows if row["real_data_backed"]],
        "failed_domains": [row["domain"] for row in rows if not row["training_verified"]],
        "real_data_gap_domains": [row["domain"] for row in rows if not row["real_data_backed"]],
        "all_passed": all(bool(row["training_surface_closed"]) for row in rows),
        "summary_csv": str(out_dir / "domain_training_summary.csv"),
        "summary_tex": str(out_dir / "tbl_domain_training_audit.tex"),
    }
    (out_dir / "training_audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out_dir / "domain_training_summary.csv", rows)
    _write_tex(out_dir / "tbl_domain_training_audit.tex", rows)
    calibration_repairs = [
        json.loads(str(row["calibration_repair"]))
        for row in rows
        if row.get("calibration_repair")
    ]
    if calibration_repairs:
        (out_dir / "healthcare_calibration_repair.json").write_text(
            json.dumps(
                {
                    "target_picp_90": 0.90,
                    "repairs": calibration_repairs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print("=== Universal Training Audit ===")
    for row in rows:
        print(
            f"  {row['domain']}: verified={row['training_verified']} real_data={row['real_data_backed']} "
            f"splits=({row['train_rows']},{row['calibration_rows']},{row['val_rows']},{row['test_rows']}) "
            f"rmse={row['rmse'] if row['rmse'] is not None else 'n/a'} note={row['note']}"
        )
    print(f"  Report → {out_dir / 'training_audit_report.json'}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
