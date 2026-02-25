#!/usr/bin/env python3
"""One-command UMN artifact builder for CPSBench/DC3S publication outputs."""
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gridpulse.cpsbench_iot.runner import run_suite, run_single
from gridpulse.cpsbench_iot.scenarios import DEFAULT_SCENARIOS
from gridpulse.forecasting.uncertainty.conformal import load_conformal
from scripts.run_ablations import run_dc3s_ablation_matrix


REQUIRED_PUBLICATION = (
    "dc3s_main_table.csv",
    "dc3s_fault_breakdown.csv",
    "fig_true_soc_violation_vs_dropout.png",
    "fig_true_soc_severity_p95_vs_dropout.png",
    "table2_ablations.csv",
    "stats_summary.json",
    "cqr_group_coverage.csv",
    "transfer_stress.csv",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full UMN admission artifact package")
    p.add_argument("--out-dir", default="reports/publication")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    return p.parse_args()


def _build_cqr_group_coverage(out_dir: Path) -> dict[str, Any]:
    target = "load_mw"
    unc_cfg_path = REPO_ROOT / "configs" / "uncertainty.yaml"
    artifacts_dir = Path("artifacts/uncertainty")
    test_npz = Path("artifacts/backtests/load_mw_test.npz")
    if unc_cfg_path.exists():
        try:
            import yaml

            unc_cfg = yaml.safe_load(unc_cfg_path.read_text(encoding="utf-8")) or {}
            artifacts_dir = Path(unc_cfg.get("artifacts_dir", artifacts_dir))
            test_tmpl = str(unc_cfg.get("test_npz", "artifacts/backtests/{target}_test.npz"))
            if "{target}" in test_tmpl:
                test_npz = Path(test_tmpl.format(target=target))
            else:
                test_npz = Path(test_tmpl)
        except Exception:
            pass

    conf_path = artifacts_dir / f"{target}_conformal.json"
    if not conf_path.exists():
        raise FileNotFoundError(f"Missing conformal artifact: {conf_path}")
    if not test_npz.exists():
        fallback = Path("artifacts/backtests/test.npz")
        if fallback.exists():
            test_npz = fallback
        else:
            raise FileNotFoundError(f"Missing test backtest arrays: {test_npz}")

    ci = load_conformal(conf_path)
    payload = np.load(test_npz)
    y_true = np.asarray(payload["y_true"], dtype=float)
    if "q_lo" in payload.files and "q_hi" in payload.files:
        q_lo = np.asarray(payload["q_lo"], dtype=float)
        q_hi = np.asarray(payload["q_hi"], dtype=float)
        lower, upper = ci.predict_interval_cqr(q_lo, q_hi)
    else:
        y_pred = np.asarray(payload["y_pred"], dtype=float)
        lower, upper = ci.predict_interval(y_pred)

    y = y_true.reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)

    width = hi - lo
    covered = ((y >= lo) & (y <= hi)).astype(float)
    vol = pd.Series(y).rolling(window=24, min_periods=6).std().fillna(0.0).to_numpy()
    q1, q2 = np.quantile(vol, [1.0 / 3.0, 2.0 / 3.0]) if len(vol) > 2 else (0.0, 0.0)
    labels = np.where(vol <= q1, "low", np.where(vol <= q2, "med", "high"))

    rows: list[dict[str, Any]] = []
    for label in ("low", "med", "high"):
        mask = labels == label
        if not np.any(mask):
            continue
        rows.append(
            {
                "target": target,
                "group": label,
                "picp_90": float(np.mean(covered[mask])),
                "mean_width": float(np.mean(width[mask])),
                "sample_count": int(np.sum(mask)),
            }
        )

    cov_df = pd.DataFrame(rows)
    cov_path = out_dir / "cqr_group_coverage.csv"
    cov_alias = out_dir / "table3_group_coverage.csv"
    cov_df.to_csv(cov_path, index=False, float_format="%.6f")
    cov_df.to_csv(cov_alias, index=False, float_format="%.6f")

    fig_tradeoff = out_dir / "fig_coverage_width_tradeoff.png"
    fig_alias = out_dir / "fig_coverage_width.png"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if cov_df.empty:
        ax.text(0.5, 0.5, "No CQR group coverage rows", ha="center", va="center")
    else:
        for _, row in cov_df.iterrows():
            ax.scatter(float(row["mean_width"]), float(row["picp_90"]), s=80)
            ax.text(float(row["mean_width"]), float(row["picp_90"]), f" {row['group']}")
        ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean Interval Width")
    ax.set_ylabel("PICP@90")
    ax.set_title("Coverage vs Width by Volatility Group (load_mw)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_tradeoff, dpi=220)
    fig.savefig(fig_alias, dpi=220)
    plt.close(fig)

    return {
        "rows": int(len(cov_df)),
        "path": str(cov_path),
        "alias": str(cov_alias),
        "figure": str(fig_tradeoff),
    }


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

    coverage_summary = _build_cqr_group_coverage(out_dir)
    transfer_summary = _build_transfer_stress(out_dir, seeds=seeds, horizon=int(args.horizon))

    _verify_outputs(out_dir)

    summary = {
        "cpsbench": cpsbench_summary,
        "ablations": ablation_summary,
        "group_coverage": coverage_summary,
        "transfer": transfer_summary,
        "required_outputs": [str(out_dir / x) for x in REQUIRED_PUBLICATION],
    }
    (out_dir / "umn_artifact_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
