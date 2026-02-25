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
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
