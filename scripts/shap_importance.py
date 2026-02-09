#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis
=================================
Generates SHAP summary plots and feature-importance tables for
all GBM models (both DE and US).

Usage:
  python scripts/shap_importance.py            # both regions
  python scripts/shap_importance.py --region de
  python scripts/shap_importance.py --region us
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
}
plt.rcParams.update(STYLE)

TARGETS = ["load_mw", "wind_mw", "solar_mw"]

TARGET_LABELS = {
    "load_mw": "Load (MW)",
    "wind_mw": "Wind (MW)",
    "solar_mw": "Solar (MW)",
}


@dataclass
class RegionConfig:
    name: str
    label: str
    features_path: Path
    models_dir: Path
    figures_dir: Path
    tables_dir: Path


def _get_regions() -> dict[str, RegionConfig]:
    return {
        "de": RegionConfig(
            name="de",
            label="Germany (OPSD)",
            features_path=REPO / "data" / "processed" / "features.parquet",
            models_dir=REPO / "artifacts" / "models",
            figures_dir=REPO / "reports" / "figures",
            tables_dir=REPO / "reports" / "tables",
        ),
        "us": RegionConfig(
            name="us",
            label="USA (EIA-930)",
            features_path=REPO / "data" / "processed" / "us_eia930" / "features.parquet",
            models_dir=REPO / "artifacts" / "models_eia930",
            figures_dir=REPO / "reports" / "eia930" / "figures",
            tables_dir=REPO / "reports" / "eia930" / "tables",
        ),
    }


def load_gbm_bundle(path: Path) -> dict | None:
    """Load a pickled GBM model bundle."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_gbm_path(models_dir: Path, target: str) -> Path | None:
    """Find the GBM pkl file for a target."""
    for p in models_dir.glob(f"gbm_*_{target}.pkl"):
        return p
    return None


def shap_for_target(
    region: RegionConfig,
    target: str,
    background_size: int = 200,
    test_size: int = 500,
) -> tuple[np.ndarray, list[str]] | None:
    """Compute SHAP values for a single target's GBM model."""
    import shap  # type: ignore

    gbm_path = _find_gbm_path(region.models_dir, target)
    if gbm_path is None:
        print(f"  [skip] No GBM model for {target} in {region.models_dir}")
        return None

    bundle = load_gbm_bundle(gbm_path)
    if bundle is None:
        return None

    model = bundle["model"]
    feat_cols = bundle["feature_cols"]

    # Load features and split test set (last 15%).
    df = pd.read_parquet(region.features_path).sort_values("timestamp")
    drop = {"timestamp", *TARGETS, "price_eur_mwh"}
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].to_numpy()
    n = len(X)
    X_test = X[int(n * 0.85):]

    # Subsample for performance.
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_test), min(background_size, len(X_test)), replace=False)
    test_idx = rng.choice(len(X_test), min(test_size, len(X_test)), replace=False)

    X_bg = X_test[bg_idx]
    X_explain = X_test[test_idx]

    # TreeExplainer is fast and exact for GBM.
    explainer = shap.TreeExplainer(model, data=X_bg)
    shap_values = explainer.shap_values(X_explain, check_additivity=False)

    return shap_values, available, X_explain


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    X_explain: np.ndarray,
    target: str,
    region: RegionConfig,
    top_k: int = 20,
):
    """Generate and save SHAP summary (beeswarm) plot."""
    import shap  # type: ignore

    region.figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Use shap's built-in plot with our figure.
    shap.summary_plot(
        shap_values,
        features=X_explain,
        feature_names=feature_names,
        max_display=top_k,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Feature Importance — {TARGET_LABELS.get(target, target)}\n({region.label})",
              fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    out_path = region.figures_dir / f"shap_summary_{target}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved {out_path}")
    return out_path


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    target: str,
    region: RegionConfig,
    top_k: int = 20,
):
    """Generate mean |SHAP| bar chart."""
    region.figures_dir.mkdir(parents=True, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 8))
    names = [feature_names[i] for i in idx]
    vals = mean_abs[idx]
    ax.barh(range(len(names)), vals[::-1], color="#1f77b4", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Mean |SHAP value|", fontweight="bold")
    ax.set_title(f"Top-{top_k} Features — {TARGET_LABELS.get(target, target)}\n({region.label})",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out_path = region.figures_dir / f"shap_bar_{target}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def save_importance_table(
    shap_values: np.ndarray,
    feature_names: list[str],
    target: str,
    region: RegionConfig,
):
    """Save top features as CSV table."""
    region.tables_dir.mkdir(parents=True, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    rows = []
    for rank, i in enumerate(order, 1):
        rows.append({
            "rank": rank,
            "feature": feature_names[i],
            "mean_abs_shap": round(float(mean_abs[i]), 6),
        })

    df = pd.DataFrame(rows)
    out_path = region.tables_dir / f"shap_importance_{target}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")
    return df


def run_region(region: RegionConfig):
    """Run SHAP analysis for all targets in a region."""
    print(f"\n{'='*60}")
    print(f"SHAP Analysis — {region.label}")
    print(f"{'='*60}")

    if not region.features_path.exists():
        print(f"  [skip] Features not found: {region.features_path}")
        return

    summary = {}
    for target in TARGETS:
        print(f"\n  Target: {target}")
        result = shap_for_target(region, target)
        if result is None:
            continue
        shap_values, feat_names, X_explain = result

        plot_shap_summary(shap_values, feat_names, X_explain, target, region)
        plot_shap_bar(shap_values, feat_names, target, region)
        df = save_importance_table(shap_values, feat_names, target, region)

        # Top-5 for summary JSON.
        summary[target] = df.head(5).to_dict(orient="records")

    # Save summary JSON.
    if summary:
        out = region.figures_dir.parent / "shap_summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  Summary JSON: {out}")


def main():
    parser = argparse.ArgumentParser(description="SHAP Feature Importance Analysis")
    parser.add_argument("--region", choices=["de", "us", "all"], default="all",
                        help="Region to analyze (default: all)")
    args = parser.parse_args()

    regions = _get_regions()
    if args.region == "all":
        for r in regions.values():
            run_region(r)
    else:
        run_region(regions[args.region])

    print("\nDone.")


if __name__ == "__main__":
    main()
