#!/usr/bin/env python3
"""One-command artifact builder for CPSBench/DC3S publication outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gridpulse.cpsbench_iot.runner import run_suite, run_single
from gridpulse.cpsbench_iot.scenarios import DEFAULT_SCENARIOS
from gridpulse.dc3s.calibration import calibrate_ambiguity_lambda
from scripts.build_cost_safety_pareto import build_cost_safety_pareto
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
    "transfer_stress.csv",
    "table_cqr_distributional_compare.csv",
    "fig_distributional_vs_cqr.png",
    "ambiguity_calibration_summary.json",
    "cost_safety_pareto.csv",
    "fig_cost_safety_pareto.png",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full publication artifact package")
    p.add_argument("--out-dir", default="reports/publication")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    return p.parse_args()


def _load_uncertainty_cfg() -> dict[str, Any]:
    path = REPO_ROOT / "configs" / "uncertainty.yaml"
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


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
    payload["regime_path_resolved"] = str(regime_path)
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

    os.environ["GRIDPULSE_DC3S_LAMBDA_MW"] = str(lambda_mw)
    os.environ["GRIDPULSE_DC3S_LEARN_LAMBDA"] = "false"
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


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(args.seeds or list(range(10)))
    scenarios = list(args.scenarios or DEFAULT_SCENARIOS)

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
    os.environ["GRIDPULSE_REQUIRE_REGIME_CQR"] = "true"

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
    )

    transfer_summary = _build_transfer_stress(out_dir, seeds=seeds, horizon=int(args.horizon))
    pareto_summary = build_cost_safety_pareto(main_table_path=out_dir / "dc3s_main_table.csv", out_dir=out_dir)

    _verify_outputs(out_dir)

    summary = {
        "cpsbench": cpsbench_summary,
        "ablations": ablation_summary,
        "group_coverage": coverage_summary,
        "regime_training": regime_summary,
        "distributional": distributional_summary,
        "ambiguity_calibration": ambiguity_summary,
        "transfer": transfer_summary,
        "pareto": pareto_summary,
        "required_outputs": [str(out_dir / x) for x in REQUIRED_PUBLICATION],
    }
    (out_dir / "publication_artifact_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
