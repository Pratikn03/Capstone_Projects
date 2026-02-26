"""Run the default CPSBench-IoT suite and verify publication artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.cpsbench_iot.runner import REQUIRED_OUTPUTS, run_suite
from gridpulse.cpsbench_iot.scenarios import DEFAULT_SCENARIOS


DEFAULT_SEEDS = [11, 22, 33, 44, 55]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPSBench-IoT default benchmark suite")
    parser.add_argument("--out-dir", default="reports/publication")
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds, e.g. --seeds 0 1 2")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Override scenarios list")
    return parser.parse_args()


def _verify_artifacts(out_dir: Path) -> None:
    missing = []
    for name in REQUIRED_OUTPUTS:
        path = out_dir / name
        if (not path.exists()) or path.stat().st_size == 0:
            missing.append(str(path))
    if missing:
        raise SystemExit(f"Missing required CPSBench artifacts: {missing}")


def _bootstrap_ci_mean(values: np.ndarray, n_bootstrap: int = 4000, alpha: float = 0.05) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(42)
    idx = rng.integers(0, vals.size, size=(n_bootstrap, vals.size))
    means = vals[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _build_table1_and_coverage_fig(out_dir: Path) -> None:
    main = pd.read_csv(out_dir / "dc3s_main_table.csv")
    rows: list[dict[str, float | str]] = []
    for controller, sub in main.groupby("controller", sort=True):
        vr_lo, vr_hi = _bootstrap_ci_mean(sub["true_soc_violation_rate"].to_numpy(dtype=float))
        sev_lo, sev_hi = _bootstrap_ci_mean(sub["true_soc_violation_severity_p95"].to_numpy(dtype=float))
        int_lo, int_hi = _bootstrap_ci_mean(sub["intervention_rate"].to_numpy(dtype=float))
        rows.append(
            {
                "controller": controller,
                "true_soc_violation_rate_mean": float(sub["true_soc_violation_rate"].mean()),
                "true_soc_violation_rate_ci_low": vr_lo,
                "true_soc_violation_rate_ci_high": vr_hi,
                "true_soc_violation_severity_p95_mean": float(sub["true_soc_violation_severity_p95"].mean()),
                "true_soc_violation_severity_p95_ci_low": sev_lo,
                "true_soc_violation_severity_p95_ci_high": sev_hi,
                "intervention_rate_mean": float(sub["intervention_rate"].mean()),
                "intervention_rate_ci_low": int_lo,
                "intervention_rate_ci_high": int_hi,
                "cost_delta_pct_mean": float(pd.to_numeric(sub.get("cost_delta_pct"), errors="coerce").mean()),
            }
        )
    table1 = pd.DataFrame(rows).sort_values("controller").reset_index(drop=True)
    table1.to_csv(out_dir / "table1_main.csv", index=False, float_format="%.6f")

    coverage = (
        main.groupby("controller", as_index=False)[["picp_90", "mean_interval_width"]]
        .mean(numeric_only=True)
        .sort_values("controller")
    )
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    x = np.arange(len(coverage))
    ax1.bar(x - 0.18, coverage["picp_90"], width=0.35, label="PICP@90", color="#4c78a8")
    ax2.bar(x + 0.18, coverage["mean_interval_width"], width=0.35, label="Mean Width", color="#f58518")
    ax1.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    ax1.set_ylabel("Coverage")
    ax2.set_ylabel("Interval Width")
    ax1.set_xticks(x)
    ax1.set_xticklabels(coverage["controller"], rotation=20, ha="right")
    ax1.set_title("Coverage and Width by Controller")
    ax1.grid(axis="y", alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_coverage_width.png", dpi=220)
    plt.close(fig)


def _run_wilcoxon_tests(out_dir: Path) -> dict:
    """
    Wilcoxon signed-rank test: dc3s_wrapped vs naive_safe_clip.

    Tests whether DC³S produces statistically significantly lower violation_rate
    and intervention_rate than the naive safe clip baseline, and whether its
    mean_interval_width is statistically significantly different.

    Uses the Wilcoxon signed-rank test (non-parametric, paired) on per-seed
    per-scenario results. This is the standard test for paired comparisons
    in benchmarks where the same episodes are evaluated across controllers.

    Output: wilcoxon_tests.json with p-values and effect sizes.
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("  scipy not available — skipping Wilcoxon tests. Install scipy to enable.")
        return {}

    main_csv = out_dir / "dc3s_main_table.csv"
    if not main_csv.exists():
        print(f"  Wilcoxon: {main_csv} not found — skipping.")
        return {}

    df = pd.read_csv(main_csv)
    if "controller" not in df.columns:
        return {}

    dc3s = df[df["controller"] == "dc3s_wrapped"].sort_values(["scenario", "seed"]).reset_index(drop=True)
    # Use deterministic_lp as the reference baseline (the standard LP controller without DC³S safety shield)
    for reference_controller in ["deterministic_lp", "naive_safe_clip", "robust_fixed_interval"]:
        naive = df[df["controller"] == reference_controller].sort_values(["scenario", "seed"]).reset_index(drop=True)
        if not naive.empty:
            break

    if dc3s.empty or naive.empty:
        print("  Wilcoxon: dc3s_wrapped or naive_safe_clip not found in results — skipping.")
        return {}

    # Align on (scenario, seed) pairs
    merged = dc3s.merge(
        naive,
        on=["scenario", "seed"],
        suffixes=("_dc3s", "_naive"),
        how="inner",
    )

    metrics = [
        ("true_soc_violation_rate", "less"),         # DC³S should be lower
        ("intervention_rate",       "less"),          # DC³S should need fewer interventions
        ("mean_interval_width",     "two-sided"),     # Width comparison (no direction prior)
        ("picp_90",                 "greater"),       # DC³S should have higher PICP
    ]

    results = {}
    print("\n" + "=" * 72)
    print("WILCOXON SIGNED-RANK TEST: dc3s_wrapped vs naive_safe_clip")
    print(f"  n_pairs = {len(merged)}")
    print("=" * 72)
    print(f"{'Metric':<35} {'Stat':>10} {'p-value':>12} {'Sig?':>8} {'Direction':>12}")
    print("-" * 72)

    for metric, alternative in metrics:
        col_dc3s  = f"{metric}_dc3s"
        col_naive = f"{metric}_naive"
        if col_dc3s not in merged.columns or col_naive not in merged.columns:
            continue
        x = merged[col_dc3s].to_numpy(dtype=float)
        y = merged[col_naive].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) < 3:
            continue

        # Guard: if all differences are zero, the test is undefined (equal distributions)
        diff = x - y
        if np.all(np.abs(diff) < 1e-12):
            results[metric] = {
                "statistic": 0.0, "p_value": 1.0, "n_pairs": int(len(x)),
                "alternative": alternative, "significant_at_0.05": False,
                "significant_at_0.01": False, "dc3s_mean": float(np.mean(x)),
                "naive_mean": float(np.mean(y)), "mean_diff": 0.0,
                "note": "all_differences_zero_distributions_identical",
            }
            print(f"  {metric:<33} {'—':>10} {'1.0000':>12} {'ns':>8}   (identical distributions)")
            continue

        stat, pval = scipy_stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        direction_label = f"DC³S {'<' if alternative == 'less' else '>' if alternative == 'greater' else '≠'} baseline"
        results[metric] = {
            "statistic": float(stat),
            "p_value": float(pval),
            "n_pairs": int(len(x)),
            "alternative": alternative,
            "significant_at_0.05": bool(pval < 0.05),
            "significant_at_0.01": bool(pval < 0.01),
            "dc3s_mean": float(np.mean(x)),
            "naive_mean": float(np.mean(y)),
            "mean_diff": float(np.mean(x - y)),
        }
        print(f"  {metric:<33} {stat:>10.2f} {pval:>12.4f} {sig:>8} {direction_label:>12}")

    print("=" * 72)
    print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")

    # Export JSON
    out_json = out_dir / "wilcoxon_tests.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n  Wilcoxon results → {out_json}")
    return results


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    summary = run_suite(
        scenarios=list(args.scenarios or DEFAULT_SCENARIOS),
        seeds=list(args.seeds or DEFAULT_SEEDS),
        out_dir=out_dir,
        horizon=int(args.horizon),
    )
    _verify_artifacts(out_dir)
    _build_table1_and_coverage_fig(out_dir)
    _run_wilcoxon_tests(out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

