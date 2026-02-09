#!/usr/bin/env python3
"""
Statistical Significance Tests for Model Comparison
====================================================
Computes:
  1. R² scores for all models (already in metrics)
  2. Diebold-Mariano (DM) tests: GBM vs LSTM, GBM vs TCN, LSTM vs TCN
  3. Forecast error auto-correlation (Ljung-Box)

Outputs:
  - reports/{region}/significance_tests.json
  - reports/{region}/tables/significance_matrix.csv

Usage:
  python scripts/statistical_tests.py
  python scripts/statistical_tests.py --region us
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from gridpulse.utils.metrics import rmse, mae, r2_score
from gridpulse.forecasting.predict import load_model_bundle
from gridpulse.forecasting.dl_lstm import LSTMForecaster
from gridpulse.forecasting.dl_tcn import TCNForecaster
from gridpulse.forecasting.datasets import SeqConfig, TimeSeriesWindowDataset
from gridpulse.utils.scaler import StandardScaler

import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

TARGETS = ["load_mw", "wind_mw", "solar_mw"]


@dataclass
class RegionConfig:
    name: str
    label: str
    features_path: Path
    models_dir: Path
    reports_dir: Path


def _get_regions() -> dict[str, RegionConfig]:
    return {
        "de": RegionConfig(
            name="de", label="Germany (OPSD)",
            features_path=REPO / "data" / "processed" / "features.parquet",
            models_dir=REPO / "artifacts" / "models",
            reports_dir=REPO / "reports",
        ),
        "us": RegionConfig(
            name="us", label="USA (EIA-930)",
            features_path=REPO / "data" / "processed" / "us_eia930" / "features.parquet",
            models_dir=REPO / "artifacts" / "models_eia930",
            reports_dir=REPO / "reports" / "eia930",
        ),
    }


def diebold_mariano_test(
    e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2
) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: E[d_t] = 0  (equal accuracy)
    H1: E[d_t] ≠ 0  (different accuracy)

    Parameters
    ----------
    e1, e2 : forecast errors from model 1 and model 2
    h      : forecast horizon (for Newey-West bandwidth)
    power  : 1 for MAE-type, 2 for MSE-type loss

    Returns
    -------
    dict with dm_stat, p_value, conclusion
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC variance estimator.
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "conclusion": "indeterminate", "d_bar": float(d_bar)}

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    if p_value < 0.01:
        conclusion = "significant (p<0.01)"
    elif p_value < 0.05:
        conclusion = "significant (p<0.05)"
    elif p_value < 0.10:
        conclusion = "marginal (p<0.10)"
    else:
        conclusion = "not significant"

    return {
        "dm_stat": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 6),
        "conclusion": conclusion,
        "d_bar": round(float(d_bar), 4),
        "better_model": "model1" if d_bar < 0 else "model2",
    }


def get_predictions(region: RegionConfig, target: str, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Get predictions from all available models for a target."""
    preds = {}
    y_true = test_df[target].to_numpy()
    preds["y_true"] = y_true

    # GBM
    gbm_path = None
    for p in region.models_dir.glob(f"gbm_*_{target}.pkl"):
        gbm_path = p
        break
    if gbm_path and gbm_path.exists():
        with open(gbm_path, "rb") as f:
            bundle = pickle.load(f)
        feat_cols = bundle.get("feature_cols", [])
        available = [c for c in feat_cols if c in test_df.columns]
        if available:
            X_test = test_df[available].to_numpy()
            preds["gbm"] = bundle["model"].predict(X_test)

    # LSTM & TCN
    for kind in ["lstm", "tcn"]:
        path = region.models_dir / f"{kind}_{target}.pt"
        if not path.exists():
            continue
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        feat_cols = bundle.get("feature_cols", [])
        available = [c for c in feat_cols if c in test_df.columns]
        if not available:
            continue

        drop = {"timestamp", *TARGETS, "price_eur_mwh"}
        X_test = test_df[available].to_numpy()
        y_test = test_df[target].to_numpy()

        lookback = int(bundle.get("lookback", 168))
        horizon = int(bundle.get("horizon", 24))
        params = bundle.get("model_params", {})

        xs = StandardScaler.from_dict(bundle["x_scaler"])
        ys = StandardScaler.from_dict(bundle["y_scaler"])
        X_test_s = xs.transform(X_test)
        y_test_s = ys.transform(y_test.reshape(-1, 1)).reshape(-1)

        if kind == "lstm":
            model = LSTMForecaster(
                n_features=X_test.shape[1],
                hidden_size=params.get("hidden_size", 128),
                num_layers=params.get("num_layers", 2),
                dropout=params.get("dropout", 0.1),
                horizon=horizon,
            )
        else:
            model = TCNForecaster(
                n_features=X_test.shape[1],
                num_channels=params.get("num_channels", [64, 64, 64]),
                kernel_size=params.get("kernel_size", 3),
                dropout=params.get("dropout", 0.2),
            )
        model.load_state_dict(bundle["state_dict"])
        model.eval()

        scfg = SeqConfig(lookback=lookback, horizon=horizon)
        ds = TimeSeriesWindowDataset(X_test_s, y_test_s, scfg)
        if len(ds) == 0:
            continue
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        pred_parts = []
        true_parts = []
        with torch.no_grad():
            for xb, yb in dl:
                out = model(xb).numpy()[:, -horizon:]
                pred_parts.append(out.reshape(-1))
                true_parts.append(yb.numpy().reshape(-1))

        y_pred_s = np.concatenate(pred_parts)
        y_true_s = np.concatenate(true_parts)
        preds[f"{kind}_pred"] = ys.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(-1)
        preds[f"{kind}_true"] = ys.inverse_transform(y_true_s.reshape(-1, 1)).reshape(-1)

    return preds


def run_region(region: RegionConfig) -> dict:
    """Run statistical tests for one region."""
    print(f"\n{'='*60}")
    print(f"Statistical Significance Tests — {region.label}")
    print(f"{'='*60}")

    if not region.features_path.exists():
        print(f"  [skip] Features not found")
        return {}

    df = pd.read_parquet(region.features_path).sort_values("timestamp")
    n = len(df)
    test_df = df.iloc[int(n * 0.85):]

    results = {"region": region.label, "targets": {}}
    rows = []

    for target in TARGETS:
        print(f"\n  Target: {target}")
        preds = get_predictions(region, target, test_df)

        target_result = {"metrics": {}, "dm_tests": []}

        # R² and metrics for each model
        if "gbm" in preds:
            y = preds["y_true"][:len(preds["gbm"])]
            p = preds["gbm"][:len(y)]
            target_result["metrics"]["gbm"] = {
                "r2": round(r2_score(y, p), 6),
                "rmse": round(rmse(y, p), 2),
                "mae": round(mae(y, p), 2),
            }
            print(f"    GBM   R²={target_result['metrics']['gbm']['r2']:.4f}  RMSE={target_result['metrics']['gbm']['rmse']:.1f}")

        for kind in ["lstm", "tcn"]:
            if f"{kind}_pred" in preds and f"{kind}_true" in preds:
                yt = preds[f"{kind}_true"]
                yp = preds[f"{kind}_pred"]
                target_result["metrics"][kind] = {
                    "r2": round(r2_score(yt, yp), 6),
                    "rmse": round(rmse(yt, yp), 2),
                    "mae": round(mae(yt, yp), 2),
                }
                print(f"    {kind.upper():5s} R²={target_result['metrics'][kind]['r2']:.4f}  RMSE={target_result['metrics'][kind]['rmse']:.1f}")

        # DM tests for all model pairs
        model_errors = {}
        if "gbm" in preds:
            y = preds["y_true"][:len(preds["gbm"])]
            model_errors["gbm"] = y - preds["gbm"][:len(y)]

        for kind in ["lstm", "tcn"]:
            if f"{kind}_pred" in preds:
                yt = preds[f"{kind}_true"]
                yp = preds[f"{kind}_pred"]
                model_errors[kind] = yt - yp

        model_names = list(model_errors.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                e1, e2 = model_errors[m1], model_errors[m2]
                # Align lengths (seq models may differ from GBM)
                min_len = min(len(e1), len(e2))
                dm = diebold_mariano_test(e1[:min_len], e2[:min_len], h=24, power=2)
                dm["model1"] = m1
                dm["model2"] = m2
                target_result["dm_tests"].append(dm)

                winner = m1 if dm["better_model"] == "model1" else m2
                print(f"    DM {m1} vs {m2}: stat={dm['dm_stat']:.3f}  p={dm['p_value']:.4f}  → {dm['conclusion']} (better: {winner})")

                rows.append({
                    "target": target,
                    "model_1": m1,
                    "model_2": m2,
                    "dm_statistic": dm["dm_stat"],
                    "p_value": dm["p_value"],
                    "conclusion": dm["conclusion"],
                    "better_model": winner,
                })

        results["targets"][target] = target_result

    # Save outputs
    region.reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = region.reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = region.reports_dir / "significance_tests.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Saved {json_path}")

    if rows:
        csv_path = tables_dir / "significance_matrix.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Tests")
    parser.add_argument("--region", choices=["de", "us", "all"], default="all")
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
