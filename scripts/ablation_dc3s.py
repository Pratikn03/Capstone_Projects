"""
DC³S Parameter Ablation Sweep.

Sweeps (k_quality, k_drift, infl_max) across CPSBench-IoT scenarios and seeds
to generate a publication-quality sensitivity table.

Usage
-----
    # Quick run (2 seeds, subset scenarios):
    python scripts/ablation_dc3s.py --quick

    # Full run from dc3s_ablation.yaml:
    python scripts/ablation_dc3s.py

    # Custom grid:
    python scripts/ablation_dc3s.py \\
        --k-quality 0.4 0.8 1.2 \\
        --k-drift 0.0 0.6 \\
        --infl-max 2.0 3.0 \\
        --seeds 11 22 33 \\
        --scenarios nominal dropout spikes drift_combo

Outputs
-------
    reports/publication/ablation_table.csv
    reports/publication/ablation_sensitivity.png  (heatmap)

The table contains one row per (k_quality, k_drift, infl_max, scenario, seed)
and summary statistics across seeds per param config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.cpsbench_iot.runner import run_suite
from gridpulse.dc3s.coverage_theorem import verify_inflation_geq_one


# ---------------------------------------------------------------------------
# Defaults (overridden by dc3s_ablation.yaml or CLI)
# ---------------------------------------------------------------------------
DEFAULT_K_QUALITY  = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
DEFAULT_K_DRIFT    = [0.0, 0.3, 0.6, 0.9]
DEFAULT_INFL_MAX   = [2.0, 3.0, 5.0]
DEFAULT_SEEDS      = [11, 22, 33]
DEFAULT_SCENARIOS  = ["nominal", "dropout", "spikes", "drift_combo"]
DEFAULT_HORIZON    = 168
BASELINE_CONTROLLER = "dc3s_wrapped"
REFERENCE_CONTROLLER = "naive_safe_clip"


def _load_ablation_cfg() -> dict[str, Any]:
    cfg_path = repo_root / "configs" / "dc3s_ablation.yaml"
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return raw.get("ablation", {})
    return {}


def _parse_args() -> argparse.Namespace:
    cfg = _load_ablation_cfg()
    p = argparse.ArgumentParser(description="DC³S ablation sweep over k_quality, k_drift, infl_max")
    p.add_argument("--quick", action="store_true",
                   help="Quick run: 2 seeds, 4 scenarios, small grid")
    p.add_argument("--k-quality", nargs="*", type=float, default=None)
    p.add_argument("--k-drift",   nargs="*", type=float, default=None)
    p.add_argument("--infl-max",  nargs="*", type=float, default=None)
    p.add_argument("--seeds",     nargs="*", type=int,   default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--horizon",   type=int, default=None)
    p.add_argument("--out-dir",   default=str(cfg.get("out_dir", "reports/publication")))
    return p.parse_args()


def _resolve_grid(args: argparse.Namespace, cfg: dict) -> dict[str, list]:
    """Merge CLI args > yaml config > hardcoded defaults."""
    if args.quick:
        return {
            "k_quality":  [0.4, 0.8, 1.2],
            "k_drift":    [0.0, 0.6],
            "infl_max":   [2.0, 3.0],
            "seeds":      [11, 22],
            "scenarios":  ["nominal", "dropout", "drift_combo"],
            "horizon":    84,
        }
    return {
        "k_quality": args.k_quality  or cfg.get("k_quality_values",  DEFAULT_K_QUALITY),
        "k_drift":   args.k_drift    or cfg.get("k_drift_values",    DEFAULT_K_DRIFT),
        "infl_max":  args.infl_max   or cfg.get("infl_max_values",   DEFAULT_INFL_MAX),
        "seeds":     args.seeds      or cfg.get("seeds",             DEFAULT_SEEDS),
        "scenarios": args.scenarios  or cfg.get("scenarios",         DEFAULT_SCENARIOS),
        "horizon":   args.horizon    or int(cfg.get("horizon",       DEFAULT_HORIZON)),
    }


def _run_one_config(
    k_quality: float,
    k_drift: float,
    infl_max: float,
    grid: dict,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Run CPSBench suite for one (k_quality, k_drift, infl_max) config."""
    # Verify invariant: infl at worst case (w_t=min_w=0.05, drift=True) must stay ≥ 1
    infl_worst = min(infl_max, 1.0 + k_quality * (1.0 - 0.05) + k_drift * 1.0)
    verify_inflation_geq_one(max(1.0, infl_worst))

    dc3s_param_overrides = {
        "k_quality": k_quality,
        "k_drift":   k_drift,
        "infl_max":  infl_max,
    }

    # run_suite returns summary dict; we need per-row CSV data
    # We call it with a temp subdir so ablation runs don't clobber the main table
    tmp_dir = out_dir / "_ablation_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Also pre-create the output dir in case it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_suite(
        scenarios=grid["scenarios"],
        seeds=grid["seeds"],
        out_dir=tmp_dir,
        horizon=grid["horizon"],
        dc3s_param_overrides=dc3s_param_overrides,
    )

    rows: list[dict[str, Any]] = []
    main_csv = tmp_dir / "dc3s_main_table.csv"
    if main_csv.exists() and main_csv.stat().st_size > 0:
        df = pd.read_csv(main_csv)
        df["k_quality"] = k_quality
        df["k_drift"]   = k_drift
        df["infl_max"]  = infl_max
        rows = df.to_dict(orient="records")
    else:
        # Fallback: build rows from summary dict
        for scenario, s_data in summary.get("scenarios", {}).items():
            for ctrl, metrics in s_data.items():
                if not isinstance(metrics, dict):
                    continue
                for seed in grid["seeds"]:
                    rows.append({
                        "k_quality":   k_quality,
                        "k_drift":     k_drift,
                        "infl_max":    infl_max,
                        "scenario":    scenario,
                        "controller":  ctrl,
                        "seed":        seed,
                        "violation_rate":   metrics.get("violation_rate", float("nan")),
                        "intervention_rate": metrics.get("intervention_rate", float("nan")),
                        "mean_interval_width": metrics.get("mean_interval_width", float("nan")),
                        "picp_90":     metrics.get("picp_90", float("nan")),
                    })
    return rows


def _build_sensitivity_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate a heatmap of violation_rate vs k_quality × k_drift for dc3s_wrapped."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping heatmap.")
        return

    dc3s = df[df["controller"] == BASELINE_CONTROLLER].copy()
    if dc3s.empty:
        return

    pivot = (
        dc3s.groupby(["k_quality", "k_drift"])["violation_rate"]
        .mean()
        .unstack(level="k_drift")
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel("k_drift")
    ax.set_ylabel("k_quality")
    ax.set_title("DC³S Violation Rate — k_quality × k_drift Sensitivity")
    plt.colorbar(im, ax=ax, label="Mean Violation Rate")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_sensitivity.png", dpi=180)
    plt.close(fig)
    print(f"  Heatmap saved → {out_dir / 'ablation_sensitivity.png'}")


def _build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed rows to mean ± std per (k_quality, k_drift, infl_max, scenario, controller)."""
    numeric_cols = ["violation_rate", "intervention_rate", "mean_interval_width", "picp_90"]
    available = [c for c in numeric_cols if c in df.columns]
    grouped = df.groupby(
        ["k_quality", "k_drift", "infl_max", "scenario", "controller"],
        sort=True,
    )
    agg: dict[str, Any] = {}
    for col in available:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"]  = (col, "std")
    agg["n_seeds"] = ("seed", "count") if "seed" in df.columns else (available[0], "count")
    summary = grouped.agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in agg.items()})
    return summary.reset_index()


def main() -> None:
    args = _parse_args()
    cfg  = _load_ablation_cfg()
    grid = _resolve_grid(args, cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_quality_vals = grid["k_quality"]
    k_drift_vals   = grid["k_drift"]
    infl_max_vals  = grid["infl_max"]
    total = len(k_quality_vals) * len(k_drift_vals) * len(infl_max_vals)

    print(f"\nDC³S Ablation Sweep")
    print(f"  Grid: k_quality={k_quality_vals}")
    print(f"        k_drift  ={k_drift_vals}")
    print(f"        infl_max ={infl_max_vals}")
    print(f"  Scenarios: {grid['scenarios']}")
    print(f"  Seeds:     {grid['seeds']}")
    print(f"  Horizon:   {grid['horizon']}h")
    print(f"  Total configs: {total}\n")

    all_rows: list[dict[str, Any]] = []
    done = 0
    for k_q in k_quality_vals:
        for k_d in k_drift_vals:
            for i_max in infl_max_vals:
                done += 1
                print(f"  [{done}/{total}] k_quality={k_q} k_drift={k_d} infl_max={i_max} ...", end=" ", flush=True)
                try:
                    rows = _run_one_config(k_q, k_d, i_max, grid, out_dir)
                    all_rows.extend(rows)
                    print(f"→ {len(rows)} rows")
                except Exception as exc:
                    print(f"→ ERROR: {exc}")

    if not all_rows:
        print("No results collected — check CPSBench runner.")
        return

    df = pd.DataFrame(all_rows)
    raw_path = out_dir / "ablation_table.csv"
    df.to_csv(raw_path, index=False, float_format="%.6f")
    print(f"\n📊 Raw ablation table ({len(df)} rows) → {raw_path}")

    # Summary table — mean ± std across seeds
    try:
        summary = _build_summary_table(df)
        summary_path = out_dir / "ablation_summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.6f")
        print(f"📋 Summary table ({len(summary)} rows) → {summary_path}")
    except Exception as exc:
        print(f"  Could not build summary table: {exc}")

    # Heatmap
    _build_sensitivity_heatmap(df, out_dir)

    # Print top-10 configs by narrowest interval width at or above 90% coverage
    dc3s_rows = df[df.get("controller", pd.Series(dtype=str)) == BASELINE_CONTROLLER] if "controller" in df.columns else df
    if not dc3s_rows.empty and "picp_90" in dc3s_rows.columns and "mean_interval_width" in dc3s_rows.columns:
        meets = dc3s_rows[pd.to_numeric(dc3s_rows["picp_90"], errors="coerce") >= 0.88]
        if not meets.empty:
            top = (
                meets.groupby(["k_quality", "k_drift", "infl_max"])["mean_interval_width"]
                .mean()
                .sort_values()
                .head(5)
            )
            print("\n✅ Top-5 configs (narrowest width at PICP≥88%):")
            print(top.to_string())

    print("\nAblation sweep complete.")


if __name__ == "__main__":
    main()
