#!/usr/bin/env python3
"""
Compute conformal prediction intervals and coverage (PICP) for US EIA-930 data.

This fills the gap in US coverage metrics that was identified in gap analysis.

Usage:
    python scripts/compute_us_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gridpulse.forecasting.uncertainty.conformal import (
    ConformalConfig,
    ConformalInterval,
    save_conformal,
)
from gridpulse.forecasting.ml_gbm import predict_gbm


def _load_model(models_dir: Path, target: str):
    """Load a trained GBM model and feature columns."""
    model_path = models_dir / f"gbm_lightgbm_{target}.pkl"
    if not model_path.exists():
        return None, None
    import pickle
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    # Model is stored in a dict with 'model' key
    if isinstance(data, dict):
        return data.get("model"), data.get("feature_cols", [])
    return data, None


def _gbm_predict(models_dir: Path, df: pd.DataFrame, target: str) -> Optional[np.ndarray]:
    """Generate GBM predictions for a target."""
    model, feature_cols = _load_model(models_dir, target)
    if model is None:
        return None
    
    # Use saved feature columns if available, else infer
    if feature_cols:
        # Filter to only columns present in df
        cols = [c for c in feature_cols if c in df.columns]
    else:
        # Identify feature columns (exclude target and timestamp)
        exclude = ["timestamp", "load_mw", "wind_mw", "solar_mw", "price_eur_mwh"]
        cols = [c for c in df.columns if c not in exclude]
    
    X = df[cols].values
    return model.predict(X)


def compute_coverage(
    test_df: pd.DataFrame,
    models_dir: Path,
    output_dir: Path,
    targets: Optional[List[str]] = None,
):
    """Compute conformal interval coverage for all targets."""
    if targets is None:
        targets = ["load_mw", "wind_mw", "solar_mw"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    interval_rows = []
    
    for target in targets:
        if target not in test_df.columns:
            print(f"  ⚠ {target} not in test data, skipping")
            continue
        
        # Get true values
        y_true = test_df[target].values
        
        # Get predictions
        y_pred = _gbm_predict(models_dir, test_df, target)
        if y_pred is None:
            print(f"  ⚠ No model for {target}, skipping")
            continue
        
        # Mask invalid values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_valid = y_true[mask]
        y_pred_valid = y_pred[mask]
        
        if len(y_true_valid) < 100:
            print(f"  ⚠ Not enough valid data for {target}, skipping")
            continue
        
        # Split into calibration (1/3) and test (2/3)
        n_cal = len(y_true_valid) // 3
        y_true_cal, y_true_test = y_true_valid[:n_cal], y_true_valid[n_cal:]
        y_pred_cal, y_pred_test = y_pred_valid[:n_cal], y_pred_valid[n_cal:]
        
        # Fit conformal interval
        ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=False))
        ci.fit_calibration(y_true_cal, y_pred_cal)
        
        # Compute coverage on test set
        coverage = ci.coverage(y_true_test, y_pred_test)
        mean_width = ci.mean_width(y_pred_test)
        
        print(f"  ✓ {target}: PICP={coverage:.4f}, MPIW={mean_width:.2f}")
        
        interval_rows.append({
            "target": target,
            "alpha": 0.10,
            "picp": round(coverage, 4),
            "mpiw": round(mean_width, 4),
            "n_test": len(y_true_test),
        })
        
        # Save conformal model
        conformal_dir = ROOT / "artifacts" / "uncertainty" / "eia930"
        conformal_dir.mkdir(parents=True, exist_ok=True)
        save_conformal(
            conformal_dir / f"{target}_conformal.json",
            ci,
            {"target": target, "region": "US"},
        )
    
    # Save coverage CSV
    if interval_rows:
        df = pd.DataFrame(interval_rows)
        output_path = metrics_dir / "forecast_intervals.csv"
        df.to_csv(output_path, index=False)
        print(f"\n  ✓ Saved: {output_path.relative_to(ROOT)}")
    
    return interval_rows


def main():
    print("=" * 60)
    print("  Computing US EIA-930 Conformal Coverage")
    print("=" * 60)
    
    # Paths
    test_parquet = ROOT / "data" / "processed" / "us_eia930" / "splits" / "test.parquet"
    models_dir = ROOT / "artifacts" / "models_eia930"
    output_dir = ROOT / "reports" / "eia930"
    
    if not test_parquet.exists():
        print(f"❌ Test data not found: {test_parquet}")
        print("   Run training first: make train-us")
        return 1
    
    if not models_dir.exists():
        print(f"❌ Models not found: {models_dir}")
        print("   Run training first: make train-us")
        return 1
    
    print(f"\n📂 Loading test data: {test_parquet}")
    test_df = pd.read_parquet(test_parquet)
    print(f"   {len(test_df)} samples, {len(test_df.columns)} features")
    
    print(f"\n📊 Computing coverage...")
    results = compute_coverage(test_df, models_dir, output_dir)
    
    if results:
        print("\n" + "=" * 60)
        print("  ✅ US Coverage Complete")
        print("=" * 60)
        print("\nCoverage Summary:")
        print("-" * 40)
        for r in results:
            print(f"  {r['target']:12s} | PICP: {r['picp']:.2%} | MPIW: {r['mpiw']:.1f} MW")
    else:
        print("\n❌ No coverage computed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
