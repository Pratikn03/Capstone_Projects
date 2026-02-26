"""
Cross-Region Transfer Evaluation for DC³S.

Tests whether DC³S conformal calibration generalises from Germany (DE/OPSD)
to the USA (US/MISO) without re-calibration — i.e., a zero-shot domain transfer.

This directly validates the paper's claim of *region-agnostic* safety guarantees
by running CPSBench with US-scale load profiles against DE-tuned DC³S parameters.

Methodology
-----------
1. "Source" run:    DE-scale scenarios (nominal params).
2. "Transfer" run:  US-scale scenarios using `load_scale` + `load_bias_mw`
                    overrides already supported by scenarios.py.
3. Compare violation_rate and PICP across both regions for each controller.

The key research question: does DC³S maintain PICP ≥ 90% and 0% violation rate
under cross-region domain shift without any recalibration?

Usage
-----
    python scripts/cross_region_transfer.py
    python scripts/cross_region_transfer.py --seeds 11 22 33 44 55 --horizon 168

Outputs
-------
    reports/publication/cross_region_transfer.csv
    reports/publication/cross_region_transfer.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.cpsbench_iot.runner import run_suite


# ---------------------------------------------------------------------------
# US-scale load parameters (EIA-930 MISO approximation)
# Derived from README metrics: MISO mean load ~13800 MW vs OPSD Germany ~50000 MW
# Ratio ≈ 0.276 (scale) + small positive bias for MISO baseline floor
# ---------------------------------------------------------------------------
US_FAULT_OVERRIDES = {
    "load_scale":       0.28,       # MISO load is ~28% of OPSD Germany
    "renewables_scale": 0.35,       # MISO renewable mix is lower
    "load_bias_mw":     500.0,      # MISO baseline floor
    "price_scale":      0.85,       # USD vs EUR rough parity after exchange
    "carbon_scale":     1.10,       # US grid is slightly more carbon-intensive
}

DE_SCENARIOS = ["nominal", "dropout", "spikes", "drift_combo"]
US_SCENARIOS = ["nominal", "dropout", "spikes", "drift_combo"]

DEFAULT_SEEDS   = [11, 22, 33, 44, 55]
DEFAULT_HORIZON = 168


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-region DC³S transfer evaluation")
    p.add_argument("--seeds",   nargs="*", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--out-dir", default="reports/publication")
    return p.parse_args()


def _run_region(
    region: str,
    scenarios: list[str],
    seeds: list[int],
    horizon: int,
    out_dir: Path,
    fault_overrides: dict | None = None,
) -> pd.DataFrame:
    """Run CPSBench suite for one region and return results as DataFrame."""
    tmp_dir = out_dir / f"_transfer_{region.lower()}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run_suite(
        scenarios=scenarios,
        seeds=seeds,
        out_dir=tmp_dir,
        horizon=horizon,
        fault_overrides=fault_overrides,
    )

    main_csv = tmp_dir / "dc3s_main_table.csv"
    if main_csv.exists() and main_csv.stat().st_size > 0:
        df = pd.read_csv(main_csv)
        df["region"] = region
        return df
    return pd.DataFrame()


def _build_transfer_table(de_df: pd.DataFrame, us_df: pd.DataFrame) -> pd.DataFrame:
    """Combine DE and US results, compute per-region mean/std."""
    combined = pd.concat([de_df, us_df], ignore_index=True)
    metrics = [c for c in [
        "true_soc_violation_rate", "intervention_rate",
        "picp_90", "mean_interval_width",
        "true_soc_violation_severity_p95"
    ] if c in combined.columns]

    summary = (
        combined
        .groupby(["region", "controller", "scenario"], sort=True)[metrics]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def _build_transfer_plot(transfer_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot.")
        return

    if "controller" not in transfer_df.columns:
        return

    controllers = ["dc3s_wrapped", "naive_safe_clip", "deterministic_lp", "robust_fixed_interval"]
    available = [c for c in controllers if c in transfer_df["controller"].unique()]
    if not available:
        return

    metric_col = next(
        (c for c in ["true_soc_violation_rate_mean", "violation_rate_mean", "picp_90_mean"]
         if c in transfer_df.columns),
        None,
    )
    if metric_col is None:
        return

    regions = ["DE", "US"]
    x = np.arange(len(available))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {"DE": "#4c78a8", "US": "#f58518"}
    for i, region in enumerate(regions):
        sub = transfer_df[transfer_df["region"] == region]
        vals = [
            sub[sub["controller"] == ctrl][metric_col].mean()
            if ctrl in sub["controller"].values else float("nan")
            for ctrl in available
        ]
        ax.bar(x + (i - 0.5) * width, vals, width, label=f"{region}", color=colors[region], alpha=0.85)

    metric_label = metric_col.replace("_mean", "").replace("_", " ").title()
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in available], fontsize=9)
    ax.set_ylabel(metric_label)
    ax.set_title(f"Cross-Region Transfer: {metric_label} by Controller (DE vs US)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "cross_region_transfer.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def _print_transfer_summary(de_df: pd.DataFrame, us_df: pd.DataFrame) -> None:
    """Print a compact comparison table for dc3s_wrapped across regions."""
    key_metrics = [c for c in [
        "true_soc_violation_rate", "intervention_rate", "picp_90"
    ] if c in de_df.columns or c in us_df.columns]

    print("\n" + "=" * 70)
    print("CROSS-REGION TRANSFER SUMMARY  (dc3s_wrapped)")
    print("=" * 70)
    print(f"{'Metric':<30} {'DE (source)':>16} {'US (transfer)':>16}")
    print("-" * 70)
    for metric in key_metrics:
        de_val = de_df[de_df["controller"] == "dc3s_wrapped"][metric].mean() \
            if "controller" in de_df.columns and metric in de_df.columns else float("nan")
        us_val = us_df[us_df["controller"] == "dc3s_wrapped"][metric].mean() \
            if "controller" in us_df.columns and metric in us_df.columns else float("nan")
        label = metric.replace("true_soc_", "").replace("_", " ")
        print(f"  {label:<28} {de_val:>16.4f} {us_val:>16.4f}")
    print("=" * 70)
    print("\nInterpretation:")
    print("  If violation_rate ≈ 0 for both regions and PICP ≥ 0.90 for both,")
    print("  DC³S safety guarantees transfer without re-calibration.  ✅")


def main() -> None:
    args  = _parse_args()
    seeds   = args.seeds   or DEFAULT_SEEDS
    horizon = args.horizon or DEFAULT_HORIZON
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nCross-Region Transfer Evaluation")
    print(f"  Seeds  : {seeds}")
    print(f"  Horizon: {horizon}h")
    print(f"  DE scenarios: {DE_SCENARIOS}")
    print(f"  US overrides: load_scale={US_FAULT_OVERRIDES['load_scale']}, "
          f"renewables_scale={US_FAULT_OVERRIDES['renewables_scale']}\n")

    print("▶ Running DE (source region) ...")
    de_df = _run_region("DE", DE_SCENARIOS, seeds, horizon, out_dir, fault_overrides=None)

    print("▶ Running US (transfer region) ...")
    us_df = _run_region("US", US_SCENARIOS, seeds, horizon, out_dir,
                        fault_overrides=US_FAULT_OVERRIDES)

    if de_df.empty and us_df.empty:
        print("No data collected — check CPSBench runner.")
        return

    _print_transfer_summary(de_df, us_df)

    # Save combined CSV
    combined_df = pd.concat([de_df, us_df], ignore_index=True)
    out_csv = out_dir / "cross_region_transfer.csv"
    combined_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\n📊 Full results ({len(combined_df)} rows) → {out_csv}")

    # Build and save summary
    if not (de_df.empty or us_df.empty):
        try:
            transfer_summary = _build_transfer_table(de_df, us_df)
            summary_path = out_dir / "cross_region_transfer_summary.csv"
            transfer_summary.to_csv(summary_path, index=False, float_format="%.6f")
            print(f"📋 Summary table → {summary_path}")
            _build_transfer_plot(transfer_summary, out_dir)
        except Exception as exc:
            print(f"  Could not build summary/plot: {exc}")

    print("\nCross-region transfer evaluation complete.")


if __name__ == "__main__":
    main()
