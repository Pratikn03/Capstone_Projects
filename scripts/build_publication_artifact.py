#!/usr/bin/env python3
"""One-command artifact builder for CPSBench/DC3S publication outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.cpsbench_iot.runner import run_suite, run_single
from orius.cpsbench_iot.scenarios import DEFAULT_SCENARIOS
from orius.dc3s.calibration import calibrate_ambiguity_lambda
from scripts.build_conference_assets import (
    build_calibration_tradeoff,
    build_dataset_cards,
    build_figure_inventory as build_conference_figure_inventory,
    build_transfer_generalization,
)
from scripts.build_baseline_comparison_table import build_release_baseline_comparison
from scripts.build_cost_safety_pareto import build_cost_safety_pareto
from scripts.build_dataset_summary_table import build_dataset_summary_rows
from scripts.run_ablations import run_dc3s_ablation_matrix
from scripts.train_distributional_load import train_distributional_load
from scripts.train_regime_cqr import train_regime_cqr_artifacts


REQUIRED_PUBLICATION = (
    "dc3s_main_table.csv",
    "dc3s_fault_breakdown.csv",
    "fig_true_soc_violation_vs_dropout.png",
    "fig_true_soc_severity_p95_vs_dropout.png",
    "table2_ablations.csv",
    "stats_summary.json",
    "cqr_group_coverage.csv",
    "cqr_calibration_summary.json",
    "fig_cqr_group_coverage.png",
    "rac_cert_summary.json",
    "fig_rac_sensitivity_vs_width.png",
    "transfer_stress.csv",
    "table5_transfer.csv",
    "table_cqr_distributional_compare.csv",
    "fig_distributional_vs_cqr.png",
    "ambiguity_calibration_summary.json",
    "cost_safety_pareto.csv",
    "fig_cost_safety_pareto.png",
    "dataset_cards.csv",
    "fig_region_dataset_cards.png",
    "fig_calibration_tradeoff.png",
    "fig_transfer_generalization.png",
    "conference_figure_inventory.json",
    "baseline_comparison_all.csv",
    "baseline_comparison_status.json",
    "release_manifest.json",
)

RELEASE_DATASETS = ("DE", "US_MISO", "US_PJM", "US_ERCOT")

PAPER_ASSET_SOURCE_MAP = {
    "FIG01_ARCHITECTURE": {"source_artifact": "reports/figures/architecture.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG02_DC3S_STEP": {"source_artifact": "reports/publication/figures/fig11_dispatch_comparison.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG03_TRUE_SOC_VIOLATION": {"source_artifact": "reports/publication/fig_true_soc_violation_vs_dropout.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG04_TRUE_SOC_SEVERITY": {"source_artifact": "reports/publication/fig_true_soc_severity_p95_vs_dropout.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG05_CQR_GROUP_COVERAGE": {"source_artifact": "reports/publication/fig_cqr_group_coverage.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG06_TRANSFER_COVERAGE": {"source_artifact": "reports/publication/fig_transfer_coverage.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG07_COST_SAFETY_FRONTIER": {"source_artifact": "reports/publication/fig_cost_safety_pareto.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG08_RAC_SENSITIVITY_WIDTH": {"source_artifact": "reports/publication/fig_rac_sensitivity_vs_width.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG09_REGION_DATASET_CARDS": {"source_artifact": "reports/publication/fig_region_dataset_cards.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG10_CALIBRATION_TRADEOFF": {"source_artifact": "reports/publication/fig_calibration_tradeoff.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "FIG11_TRANSFER_GENERALIZATION": {"source_artifact": "reports/publication/fig_transfer_generalization.png", "build_command": "bash scripts/export_paper_assets.sh"},
    "TBL01_MAIN_RESULTS": {"source_artifact": "reports/publication/dc3s_main_table.csv", "build_command": "bash scripts/export_paper_assets.sh"},
    "TBL02_ABLATIONS": {"source_artifact": "reports/publication/table2_ablations.csv", "build_command": "bash scripts/export_paper_assets.sh"},
    "TBL03_CQR_GROUP_COVERAGE": {"source_artifact": "reports/publication/cqr_group_coverage.csv", "build_command": "bash scripts/export_paper_assets.sh"},
    "TBL04_TRANSFER_STRESS": {"source_artifact": "reports/publication/transfer_stress.csv", "build_command": "bash scripts/export_paper_assets.sh"},
    "TBL05_DATASET_SUMMARY": {"source_artifact": "reports/publication/tables/table1_dataset_summary.csv", "build_command": "bash scripts/export_paper_assets.sh"},
    "TBL06_HYPERPARAMS": {"source_artifact": "configs/train_forecast.yaml", "build_command": "bash scripts/export_paper_assets.sh"},
    "TBL07_DATASET_CARDS": {"source_artifact": "reports/publication/dataset_cards.csv", "build_command": "bash scripts/export_paper_assets.sh"},
    "TBL08_FORECAST_BASELINES": {"source_artifact": "reports/publication/baseline_comparison_all.csv", "build_command": "bash scripts/export_paper_assets.sh"},
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full publication artifact package")
    p.add_argument("--out-dir", default="reports/publication")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--release-id", default=None)
    p.add_argument("--release-family", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _load_uncertainty_cfg() -> dict[str, Any]:
    path = REPO_ROOT / "configs" / "uncertainty.yaml"
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_release_source_runs(release_id: str) -> dict[str, dict[str, Any]]:
    source_runs: dict[str, dict[str, Any]] = {}
    for dataset in RELEASE_DATASETS:
        dataset_lower = dataset.lower()
        manifest_path = REPO_ROOT / "artifacts" / "runs" / dataset_lower / release_id / "registry" / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing candidate run manifest for {dataset}: {manifest_path}. "
                "Build publication artifacts only from a verified single release family."
            )
        payload = _load_json(manifest_path)
        manifest_release_id = str(payload.get("release_id", ""))
        if manifest_release_id != release_id:
            raise ValueError(
                f"Run manifest {manifest_path} belongs to release_id={manifest_release_id!r}, expected {release_id!r}"
            )
        if payload.get("accepted") is not True:
            raise ValueError(f"Run manifest {manifest_path} is not accepted.")
        artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
        source_runs[dataset] = {
            "release_id": manifest_release_id,
            "run_id": payload.get("run_id"),
            "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "reports_dir": artifacts.get("reports_dir"),
            "models_dir": artifacts.get("models_dir"),
            "uncertainty_dir": artifacts.get("uncertainty_dir"),
            "backtests_dir": artifacts.get("backtests_dir"),
            "selection_summary_path": payload.get("selection_summary_path"),
            "accepted": payload.get("accepted"),
        }
    return source_runs


def _ensure_cpsbench_release_source(release_id: str) -> str:
    cpsbench_dir = REPO_ROOT / "reports" / "runs" / "cpsbench" / release_id
    if not cpsbench_dir.exists():
        raise FileNotFoundError(
            f"Missing CPSBench source directory for release_id={release_id}: {cpsbench_dir}"
        )
    return str(cpsbench_dir.relative_to(REPO_ROOT))


def _build_paper_asset_manifest(out_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = REPO_ROOT / "paper" / "manifest.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    paper_assets: dict[str, dict[str, str]] = {}
    for section in ("figures", "tables"):
        section_payload = payload.get(section, {}) if isinstance(payload.get(section), dict) else {}
        for token, entry in section_payload.items():
            path = entry.get("path") if isinstance(entry, dict) else None
            mapping = PAPER_ASSET_SOURCE_MAP.get(token, {})
            source_artifact = str(mapping.get("source_artifact", ""))
            if source_artifact.startswith("reports/publication/"):
                source_artifact = str(out_dir / source_artifact.removeprefix("reports/publication/"))
            paper_assets[token] = {
                "paper_path": str(path or ""),
                "source_artifact": source_artifact,
                "build_command": str(mapping.get("build_command", "")),
            }
    return paper_assets


def _load_dc3s_cfg() -> dict[str, Any]:
    path = REPO_ROOT / "configs" / "dc3s.yaml"
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    dc3s = payload.get("dc3s", {})
    return dc3s if isinstance(dc3s, dict) else {}


def _resolve_split_paths(unc_cfg: dict[str, Any]) -> dict[str, Path]:
    split_cfg = unc_cfg.get("publication_splits", {}) if isinstance(unc_cfg.get("publication_splits"), dict) else {}
    train = Path(str(split_cfg.get("train", "data/processed/splits/train.parquet")))
    cal = Path(str(split_cfg.get("calibration", "data/processed/splits/calibration.parquet")))
    test = Path(str(split_cfg.get("test", "data/processed/splits/test.parquet")))
    if not train.is_absolute():
        train = REPO_ROOT / train
    if not cal.is_absolute():
        cal = REPO_ROOT / cal
    if not test.is_absolute():
        test = REPO_ROOT / test
    for p in (train, cal, test):
        if not p.exists():
            raise FileNotFoundError(f"Missing publication split file: {p}")
    return {"train": train, "calibration": cal, "test": test}


def _prepare_regime_cqr(out_dir: Path, split_paths: dict[str, Path], unc_cfg: dict[str, Any]) -> dict[str, Any]:
    regime_cfg = unc_cfg.get("regime_cqr", {}) if isinstance(unc_cfg.get("regime_cqr"), dict) else {}
    if not bool(regime_cfg.get("enabled", True)):
        raise RuntimeError("regime_cqr.enabled must be true for publication runs")
    payload = train_regime_cqr_artifacts(
        train_path=split_paths["train"],
        cal_path=split_paths["calibration"],
        test_path=split_paths["test"],
        out_dir=out_dir,
        target="load_mw",
        alpha=float(unc_cfg.get("conformal", {}).get("alpha", 0.10)),
        bins=int(regime_cfg.get("n_bins", 3)),
        vol_window=int(regime_cfg.get("vol_window", 24)),
        backend_policy=str(regime_cfg.get("quantile_backend_policy", "strict")),
        quantile_backend=str(regime_cfg.get("quantile_backend", "lightgbm")),
    )
    regime_path = Path(str(regime_cfg.get("artifact_path", "artifacts/uncertainty/{target}_regime_cqr.json")).format(target="load_mw"))
    if not regime_path.exists():
        raise FileNotFoundError(f"Regime CQR artifact missing after training: {regime_path}")
    rac_cfg = unc_cfg.get("rac_cert", {}) if isinstance(unc_cfg.get("rac_cert"), dict) else {}
    rac_path = Path(str(rac_cfg.get("artifact_path", "artifacts/uncertainty/{target}_rac_cert.json")).format(target="load_mw"))
    if not rac_path.exists():
        raise FileNotFoundError(f"RAC-Cert artifact missing after training: {rac_path}")
    payload["regime_path_resolved"] = str(regime_path)
    payload["rac_path_resolved"] = str(rac_path)
    return payload


def _prepare_distributional(out_dir: Path, split_paths: dict[str, Path]) -> dict[str, Any]:
    return train_distributional_load(
        train_path=split_paths["train"],
        cal_path=split_paths["calibration"],
        test_path=split_paths["test"],
        out_dir=out_dir,
        target="load_mw",
    )


def _build_cqr_group_coverage(out_dir: Path) -> dict[str, Any]:
    cov_path = out_dir / "cqr_group_coverage.csv"
    if not cov_path.exists():
        raise FileNotFoundError(
            f"Missing RegimeCQR coverage file: {cov_path}. "
            "train_regime_cqr.py must run before publication build."
        )
    cov_df = pd.read_csv(cov_path)
    if "group" not in cov_df.columns:
        raise ValueError("cqr_group_coverage.csv missing 'group' column")
    cov_df["group"] = cov_df["group"].astype(str).replace({"mid": "med"})
    cov_df.to_csv(cov_path, index=False, float_format="%.6f")

    cov_alias = out_dir / "table3_group_coverage.csv"
    cov_df.to_csv(cov_alias, index=False, float_format="%.6f")

    fig_primary = out_dir / "fig_cqr_group_coverage.png"
    fig_tradeoff = out_dir / "fig_coverage_width_tradeoff.png"
    fig_alias = out_dir / "fig_coverage_width.png"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if cov_df.empty:
        ax.text(0.5, 0.5, "No CQR group coverage rows", ha="center", va="center")
    else:
        for _, row in cov_df.iterrows():
            if not np.isfinite(float(row.get("mean_width", np.nan))) or not np.isfinite(float(row.get("picp_90", np.nan))):
                continue
            ax.scatter(float(row["mean_width"]), float(row["picp_90"]), s=80)
            ax.text(float(row["mean_width"]), float(row["picp_90"]), f" {row['group']}")
        ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean Interval Width")
    ax.set_ylabel("PICP@90")
    ax.set_title("Coverage vs Width by Volatility Group (load_mw)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_primary, dpi=220)
    fig.savefig(fig_tradeoff, dpi=220)
    fig.savefig(fig_alias, dpi=220)
    plt.close(fig)

    summary_path = out_dir / "cqr_calibration_summary.json"
    if not summary_path.exists():
        summary_payload = {
            "source": "build_publication_artifact._build_cqr_group_coverage",
            "target": "load_mw",
            "rows": int(len(cov_df)),
            "groups": sorted(cov_df["group"].astype(str).unique().tolist()) if not cov_df.empty else [],
            "overall_picp_90": float(cov_df["picp_90"].mean()) if not cov_df.empty else None,
            "overall_mean_width": float(cov_df["mean_width"].mean()) if not cov_df.empty else None,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "rows": int(len(cov_df)),
        "path": str(cov_path),
        "alias": str(cov_alias),
        "figure": str(fig_primary),
        "summary": str(summary_path),
    }


def _build_rac_diagnostics(out_dir: Path) -> dict[str, Any]:
    main_path = out_dir / "dc3s_main_table.csv"
    if not main_path.exists():
        raise FileNotFoundError(f"Missing main table for RAC diagnostics: {main_path}")
    df = pd.read_csv(main_path)

    required_cols = {
        "rac_sensitivity_mean",
        "rac_sensitivity_p95",
        "rac_q_multiplier_mean",
        "rac_q_multiplier_p95",
        "rac_inflation_mean",
        "adaptive_width_mean",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Main table missing RAC diagnostics columns: {sorted(missing)}")

    summary = {
        "rows": int(len(df)),
        "controllers": sorted(df["controller"].astype(str).unique().tolist()) if "controller" in df.columns else [],
        "rac_sensitivity_mean_global": float(pd.to_numeric(df["rac_sensitivity_mean"], errors="coerce").mean()),
        "rac_q_multiplier_mean_global": float(pd.to_numeric(df["rac_q_multiplier_mean"], errors="coerce").mean()),
        "rac_inflation_mean_global": float(pd.to_numeric(df["rac_inflation_mean"], errors="coerce").mean()),
    }
    summary_path = out_dir / "rac_cert_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    fig_path = out_dir / "fig_rac_sensitivity_vs_width.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in df.groupby("controller", sort=True):
        x = pd.to_numeric(sub["rac_sensitivity_mean"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["adaptive_width_mean"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.scatter(x[mask], y[mask], alpha=0.8, label=str(controller))
    ax.set_xlabel("RAC Sensitivity Mean")
    ax.set_ylabel("Adaptive Width Mean")
    ax.set_title("RAC Sensitivity vs Interval Width")
    ax.grid(alpha=0.3)
    if len(df):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return {"summary_path": str(summary_path), "figure_path": str(fig_path)}


def _calibrate_ambiguity_from_splits(
    *,
    out_dir: Path,
    split_paths: dict[str, Path],
    dc3s_cfg: dict[str, Any],
) -> dict[str, Any]:
    cal_df = pd.read_parquet(split_paths["calibration"]) if split_paths["calibration"].suffix == ".parquet" else pd.read_csv(split_paths["calibration"])
    if "load_mw" not in cal_df.columns:
        raise ValueError("Calibration split missing load_mw needed for ambiguity calibration")
    load = cal_df["load_mw"].to_numpy(dtype=float)
    if len(load) < 2:
        raise ValueError("Calibration split too short for ambiguity calibration")
    residuals = np.abs(np.diff(load))

    ambiguity_cfg = dc3s_cfg.get("ambiguity", {}) if isinstance(dc3s_cfg.get("ambiguity"), dict) else {}
    q = float(ambiguity_cfg.get("lambda_quantile", 0.95))
    scale = float(ambiguity_cfg.get("lambda_scale", 1.0))
    min_lambda = float(ambiguity_cfg.get("lambda_min_mw", 0.0))
    max_lambda = float(ambiguity_cfg["lambda_max_mw"]) if "lambda_max_mw" in ambiguity_cfg else None
    lambda_mw = calibrate_ambiguity_lambda(
        residuals_mw=residuals,
        quantile=q,
        scale=scale,
        min_lambda=min_lambda,
        max_lambda=max_lambda,
    )

    summary = {
        "residual_count": int(len(residuals)),
        "residual_quantile": float(q),
        "lambda_scale": float(scale),
        "lambda_mw_calibrated": float(lambda_mw),
        "residual_mean_mw": float(np.mean(residuals)),
        "residual_p95_mw": float(np.quantile(residuals, 0.95)),
    }
    summary_path = out_dir / "ambiguity_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    os.environ["ORIUS_DC3S_LAMBDA_MW"] = str(lambda_mw)
    os.environ["ORIUS_DC3S_LEARN_LAMBDA"] = "false"
    return {"summary_path": str(summary_path), **summary}


def _build_transfer_stress(out_dir: Path, seeds: list[int], horizon: int) -> dict[str, Any]:
    transfer_cases: dict[str, dict[str, float]] = {
        "DE_to_US_no_retrain": {
            "load_scale": 1.18,
            "renewables_scale": 0.82,
            "seasonal_shift_hours": 6,
            "load_bias_mw": 600.0,
        },
        "US_to_DE_no_retrain": {
            "load_scale": 0.86,
            "renewables_scale": 1.14,
            "seasonal_shift_hours": -6,
            "load_bias_mw": -450.0,
        },
        "DE_season_shift": {
            "seasonal_shift_hours": 12,
            "load_scale": 1.04,
            "renewables_scale": 0.94,
        },
        "US_season_shift": {
            "seasonal_shift_hours": -12,
            "load_scale": 0.98,
            "renewables_scale": 1.05,
        },
    }

    sweep_specs: list[tuple[str, str, float | str, dict[str, Any]]] = []
    sweep_specs.append(("nominal", "nominal", 0.0, {}))
    for p in (0.0, 0.10, 0.20, 0.30):
        sweep_specs.append(("dropout", "dropout", p, {"dropout_rate": float(p), "soc_dropout_prob": float(p)}))
    for p in (0.0, 0.10, 0.20):
        sweep_specs.append(("stale", "nominal", p, {"soc_stale_prob": float(p)}))
    sweep_specs.append(("delay", "delay_jitter", 0, {"delay_seconds": 0.0, "delay_rate": 0.0, "soc_stale_prob": 0.0}))
    sweep_specs.append(("delay", "delay_jitter", "high", {"delay_seconds": 15.0, "delay_rate": 0.50, "soc_stale_prob": 0.35}))

    rows: list[dict[str, Any]] = []
    for case_name, case_overrides in transfer_cases.items():
        for sweep_type, scenario, level, sweep_overrides in sweep_specs:
            overrides = {**case_overrides, **sweep_overrides}
            for seed in seeds:
                payload = run_single(
                    scenario=scenario,
                    seed=int(seed),
                    horizon=int(horizon),
                    fault_overrides=overrides,
                )
                for row in payload["main_rows"]:
                    rows.append(
                        {
                            "transfer_case": case_name,
                            "sweep_type": sweep_type,
                            "sweep_value": level,
                            "scenario": scenario,
                            "seed": int(seed),
                            "controller": row.get("controller"),
                            "picp_90": row.get("picp_90"),
                            "mean_width": row.get("mean_interval_width"),
                            "true_soc_violation_rate": row.get("true_soc_violation_rate"),
                            "true_soc_violation_severity_p95_mwh": row.get(
                                "true_soc_violation_severity_p95_mwh",
                                row.get("true_soc_violation_severity_p95"),
                            ),
                            "cost_delta_pct": row.get("cost_delta_pct"),
                        }
                    )

    transfer_df = pd.DataFrame(rows)
    transfer_path = out_dir / "transfer_stress.csv"
    alias_path = out_dir / "table5_transfer.csv"
    transfer_df.to_csv(transfer_path, index=False, float_format="%.6f")
    transfer_df.to_csv(alias_path, index=False, float_format="%.6f")

    fig_path = out_dir / "fig_transfer_coverage.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = transfer_df[
        (transfer_df["controller"] == "dc3s_wrapped")
        & (transfer_df["sweep_type"].isin(["nominal", "dropout"]))
    ].copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No transfer rows", ha="center", va="center")
    else:
        agg = (
            plot_df.groupby(["transfer_case", "sweep_type"], as_index=False)["picp_90"]
            .mean(numeric_only=True)
            .sort_values(["transfer_case", "sweep_type"])
        )
        for case, sub in agg.groupby("transfer_case", sort=True):
            x = np.arange(len(sub))
            ax.plot(x, sub["picp_90"].to_numpy(dtype=float), marker="o", label=str(case))
        ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["dropout", "nominal"])
        ax.legend(fontsize=8)
    ax.set_title("Transfer Coverage (DC3S)")
    ax.set_ylabel("PICP@90")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    return {
        "rows": int(len(transfer_df)),
        "path": str(transfer_path),
        "alias": str(alias_path),
        "figure": str(fig_path),
    }


def _build_dataset_summary_table(out_dir: Path) -> dict[str, Any]:
    rows = build_dataset_summary_rows()
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "table1_dataset_summary.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    return {"rows": int(len(rows)), "path": str(table_path)}


def _verify_outputs(out_dir: Path) -> None:
    missing: list[str] = []
    for name in REQUIRED_PUBLICATION:
        p = out_dir / name
        if (not p.exists()) or p.stat().st_size == 0:
            missing.append(str(p))
    if missing:
        raise RuntimeError(f"Missing required publication artifacts: {missing}")

    main = pd.read_csv(out_dir / "dc3s_main_table.csv")
    required_cols = {
        "true_soc_violation_severity_mean",
        "true_soc_violation_severity_p95",
        "true_soc_violation_severity_mean_mwh",
        "true_soc_violation_severity_p95_mwh",
    }
    if not required_cols.issubset(set(main.columns)):
        raise RuntimeError(f"Main table missing severity compatibility columns: {sorted(required_cols - set(main.columns))}")

    transfer = pd.read_csv(out_dir / "transfer_stress.csv")
    required_cases = {
        "DE_to_US_no_retrain",
        "US_to_DE_no_retrain",
        "DE_season_shift",
        "US_season_shift",
    }
    if not required_cases.issubset(set(transfer["transfer_case"].astype(str).unique().tolist())):
        raise RuntimeError("transfer_stress.csv missing one or more required transfer cases")
    table5 = pd.read_csv(out_dir / "table5_transfer.csv")
    if "transfer_case" in table5.columns and table5["transfer_case"].astype(str).str.contains("pending_transfer_artifacts", na=False).any():
        raise RuntimeError("table5_transfer.csv contains placeholder rows")
    stats = json.loads((out_dir / "stats_summary.json").read_text(encoding="utf-8"))
    stats_blob = json.dumps(stats, sort_keys=True)
    if "/tmp/" in stats_blob:
        raise RuntimeError("stats_summary.json contains /tmp paths")
    figure_inventory = json.loads((out_dir / "conference_figure_inventory.json").read_text(encoding="utf-8"))
    summary = figure_inventory.get("summary", {}) if isinstance(figure_inventory.get("summary"), dict) else {}
    if not bool(summary.get("ready", False)):
        raise RuntimeError("conference_figure_inventory.json reports missing required conference figures")

    baseline_table = out_dir / "baseline_comparison_all.csv"
    baseline_status_path = out_dir / "baseline_comparison_status.json"
    baseline_text = baseline_table.read_text(encoding="utf-8")
    if "NaN" in baseline_text:
        raise RuntimeError("baseline_comparison_all.csv contains NaN; missing values must render as ---")
    baseline_status = json.loads(baseline_status_path.read_text(encoding="utf-8"))
    if not bool(baseline_status.get("full_table_complete", False)):
        raise RuntimeError("baseline_comparison_status.json reports an incomplete six-model DE/US forecast table")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_metrics_manifest_scope() -> dict[str, Any]:
    path = REPO_ROOT / "paper" / "metrics_manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        "metric_policy": payload.get("metric_policy", {}),
        "run_ids": payload.get("run_ids", {}),
        "dataset_profiles": payload.get("dataset_profiles", {}),
        "release_family": (
            payload.get("metric_policy", {}).get("release_family", {})
            if isinstance(payload.get("metric_policy"), dict)
            else {}
        ),
    }


def _write_release_manifest(
    *,
    out_dir: Path,
    release_id: str,
    seeds: list[int],
    scenarios: list[str],
    horizon: int,
    command: str,
    source_runs: dict[str, dict[str, Any]],
    cpsbench_source_dir: str,
    paper_assets: dict[str, dict[str, str]],
    conference_figure_inventory: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        )
    except Exception:
        commit = "unknown"
    artifact_hashes: dict[str, str] = {}
    for name in REQUIRED_PUBLICATION:
        if name == "release_manifest.json":
            continue
        p = out_dir / name
        if p.exists() and p.is_file():
            artifact_hashes[name] = _sha256_file(p)
    metrics_scope = _load_metrics_manifest_scope()
    controller_list: list[str] = []
    main_table_path = out_dir / "dc3s_main_table.csv"
    if main_table_path.exists():
        main_df = pd.read_csv(main_table_path)
        if "controller" in main_df.columns:
            controller_list = sorted(str(value) for value in main_df["controller"].dropna().unique().tolist())
    manifest = {
        "release_id": release_id,
        "git_commit": commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_family": release_id,
        "command": command,
        "seeds": seeds,
        "scenarios": scenarios,
        "horizon": int(horizon),
        "source_runs": source_runs,
        "benchmark_sources": {"cpsbench": cpsbench_source_dir},
        "paper_assets": paper_assets,
        "artifact_hashes_sha256": artifact_hashes,
        "paper_metric_policy": metrics_scope.get("metric_policy", {}),
        "paper_run_ids": metrics_scope.get("run_ids", {}),
        "dataset_profiles": metrics_scope.get("dataset_profiles", {}),
        "controllers_present": controller_list,
        "conference_figure_inventory": conference_figure_inventory or {},
    }
    manifest_path = out_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    release_id = args.release_id or args.release_family
    if not release_id:
        raise SystemExit("build_publication_artifact.py requires --release-id for normalized release-family builds")

    seeds = list(args.seeds or list(range(10)))
    scenarios = list(args.scenarios or DEFAULT_SCENARIOS)
    source_runs = _load_release_source_runs(release_id)
    cpsbench_source_dir = _ensure_cpsbench_release_source(release_id)
    paper_assets = _build_paper_asset_manifest(out_dir)

    unc_cfg = _load_uncertainty_cfg()
    split_paths = _resolve_split_paths(unc_cfg)
    regime_summary = _prepare_regime_cqr(out_dir=out_dir, split_paths=split_paths, unc_cfg=unc_cfg)
    coverage_summary = _build_cqr_group_coverage(out_dir)
    distributional_summary = _prepare_distributional(out_dir=out_dir, split_paths=split_paths)
    ambiguity_summary = _calibrate_ambiguity_from_splits(
        out_dir=out_dir,
        split_paths=split_paths,
        dc3s_cfg=_load_dc3s_cfg(),
    )
    os.environ["ORIUS_REQUIRE_REGIME_CQR"] = "true"
    os.environ["ORIUS_REQUIRE_RAC_CERT"] = "true"

    cpsbench_summary = run_suite(
        scenarios=scenarios,
        seeds=seeds,
        out_dir=out_dir,
        horizon=int(args.horizon),
    )

    ablation_summary = run_dc3s_ablation_matrix(
        output_dir=out_dir,
        seeds=seeds,
        scenario="drift_combo",
        horizon=int(args.horizon),
        precomputed_main_csv=out_dir / "dc3s_main_table.csv",
        precomputed_sweep_csv=out_dir / "cpsbench_merged_sweep.csv",
    )

    transfer_summary = _build_transfer_stress(out_dir, seeds=seeds, horizon=int(args.horizon))
    pareto_summary = build_cost_safety_pareto(main_table_path=out_dir / "dc3s_main_table.csv", out_dir=out_dir)
    rac_summary = _build_rac_diagnostics(out_dir)
    dataset_summary = _build_dataset_summary_table(out_dir)
    baseline_summary = build_release_baseline_comparison(release_id=release_id, out_dir=out_dir)
    dataset_cards = build_dataset_cards(dataset_summary_path=out_dir / "tables" / "table1_dataset_summary.csv", out_dir=out_dir)
    calibration_tradeoff = build_calibration_tradeoff(out_dir=out_dir)
    transfer_generalization = build_transfer_generalization(out_dir=out_dir)
    conference_figure_inventory = build_conference_figure_inventory(out_dir=out_dir)

    release_manifest = _write_release_manifest(
        out_dir=out_dir,
        release_id=release_id,
        seeds=seeds,
        scenarios=scenarios,
        horizon=int(args.horizon),
        command="python3 scripts/build_publication_artifact.py "
        + f"--release-id {release_id} "
        + f"--out-dir {out_dir} --horizon {int(args.horizon)} --seeds {' '.join(str(s) for s in seeds)} "
        + f"--scenarios {' '.join(str(s) for s in scenarios)}",
        source_runs=source_runs,
        cpsbench_source_dir=cpsbench_source_dir,
        paper_assets=paper_assets,
        conference_figure_inventory=conference_figure_inventory.get("summary", {}),
    )
    _verify_outputs(out_dir)

    summary = {
        "cpsbench": cpsbench_summary,
        "ablations": ablation_summary,
        "group_coverage": coverage_summary,
        "regime_training": regime_summary,
        "distributional": distributional_summary,
        "ambiguity_calibration": ambiguity_summary,
        "rac_diagnostics": rac_summary,
        "transfer": transfer_summary,
        "pareto": pareto_summary,
        "dataset_summary_table": dataset_summary,
        "baseline_comparison": baseline_summary,
        "dataset_cards": dataset_cards,
        "calibration_tradeoff": calibration_tradeoff,
        "transfer_generalization": transfer_generalization,
        "conference_figure_inventory": conference_figure_inventory,
        "release_manifest": release_manifest,
        "required_outputs": [str(out_dir / x) for x in REQUIRED_PUBLICATION],
    }
    (out_dir / "publication_artifact_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
