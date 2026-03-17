#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orius.forecasting.ml_gbm import train_gbm, predict_gbm
from orius.forecasting.uncertainty.cqr import RegimeCQR, RegimeCQRConfig
from orius.dc3s.rac_cert import RACCertConfig, RACCertModel


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _feature_cols(df: pd.DataFrame, target: str, user_cols: str | None) -> list[str]:
    if user_cols:
        cols = [c.strip() for c in user_cols.split(",") if c.strip()]
        if not cols:
            raise ValueError("--features provided but no columns parsed")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return cols

    cols: list[str] = []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns found. Pass --features explicitly.")
    return cols


def _train_quantile_predictions(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    x_test: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params_base: dict[str, Any] = {
        "backend": backend,
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "random_state": 42,
    }

    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for q in (0.1, 0.5, 0.9):
        params = dict(params_base)
        params["objective"] = "quantile"
        params["alpha"] = float(q)
        _kind, model = train_gbm(x_train, y_train, params=params, use_pipeline=False)
        cal_pred = np.asarray(predict_gbm(model, x_cal), dtype=float)
        test_pred = np.asarray(predict_gbm(model, x_test), dtype=float)
        preds[str(q)] = (cal_pred, test_pred)

    q10_cal, q10_test = preds["0.1"]
    _q50_cal, q50_test = preds["0.5"]
    q90_cal, q90_test = preds["0.9"]
    return q10_cal, q90_cal, q10_test, q90_test


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train regime-aware CQR artifacts")
    p.add_argument("--train", required=True, help="Path to train split parquet/csv")
    p.add_argument("--cal", required=True, help="Path to calibration split parquet/csv")
    p.add_argument("--test", required=True, help="Path to test split parquet/csv")
    p.add_argument("--target", default="load_mw")
    p.add_argument("--features", default=None, help="Comma-separated feature list")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--bins", type=int, default=3)
    p.add_argument("--vol-window", type=int, default=24)
    p.add_argument("--backend-policy", choices=("strict", "fallback"), default="strict")
    p.add_argument("--quantile-backend", default="lightgbm")
    p.add_argument("--out", default="reports/publication")
    return p


def train_regime_cqr_artifacts(
    *,
    train_path: Path,
    cal_path: Path,
    test_path: Path,
    out_dir: Path,
    target: str = "load_mw",
    features: str | None = None,
    alpha: float = 0.10,
    bins: int = 3,
    vol_window: int = 24,
    backend_policy: str = "strict",
    quantile_backend: str = "lightgbm",
) -> dict[str, str | float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("artifacts/uncertainty")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_table(train_path)
    cal_df = _load_table(cal_path)
    test_df = _load_table(test_path)

    if target not in train_df.columns or target not in cal_df.columns or target not in test_df.columns:
        raise ValueError(f"target '{target}' must exist in train/cal/test tables")

    feats = _feature_cols(train_df, target, features)
    x_train = train_df[feats].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    x_cal = cal_df[feats].to_numpy(dtype=float)
    y_cal = cal_df[target].to_numpy(dtype=float)
    x_test = test_df[feats].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    mode = "quantile"
    fallback_meta: dict[str, Any] = {}
    try:
        q10_cal, q90_cal, q10_test, q90_test = _train_quantile_predictions(
            x_train=x_train,
            y_train=y_train,
            x_cal=x_cal,
            x_test=x_test,
            backend=str(quantile_backend),
        )
    except Exception as exc:
        if backend_policy == "strict":
            raise RuntimeError(
                "Quantile backend unavailable or failed. Re-run with --backend-policy fallback for dev mode."
            ) from exc
        mode = "fallback"
        # Residual fallback around point model prediction.
        params = {
            "backend": "sklearn_hgbrt",
            "max_depth": 6,
            "learning_rate": 0.05,
            "max_iter": 500,
            "random_state": 42,
        }
        _kind, model = train_gbm(x_train, y_train, params=params, use_pipeline=False)
        yhat_cal = np.asarray(predict_gbm(model, x_cal), dtype=float)
        yhat_test = np.asarray(predict_gbm(model, x_test), dtype=float)
        resid_cal = np.abs(y_cal - yhat_cal)
        qhat = float(np.quantile(resid_cal, max(1e-6, 1.0 - float(alpha))))
        q10_cal = yhat_cal - qhat
        q90_cal = yhat_cal + qhat
        q10_test = yhat_test - qhat
        q90_test = yhat_test + qhat
        fallback_meta = {"fallback_qhat": qhat}

    regime = RegimeCQR(
        cfg=RegimeCQRConfig(
            alpha=float(alpha),
            n_bins=int(bins),
            vol_window=int(vol_window),
            eps=1e-9,
            fail_fast_quantile_backend=(backend_policy == "strict"),
        )
    )
    fit_meta = regime.fit(y_cal=y_cal, q_lo_cal=q10_cal, q_hi_cal=q90_cal)

    lower, upper, bin_ids = regime.predict_interval(y_context=y_test, q_lo=q10_test, q_hi=q90_test)
    width = np.asarray(upper - lower, dtype=float)
    covered = np.asarray((y_test >= lower) & (y_test <= upper), dtype=float)

    label_map = {0: "low", 1: "mid", 2: "high"}
    rows: list[dict[str, Any]] = []
    for b in range(int(bins)):
        mask = bin_ids == b
        group_name = label_map.get(b, f"bin_{b}")
        if not np.any(mask):
            rows.append(
                {
                    "target": target,
                    "group": group_name,
                    "bin": int(b),
                    "picp_90": np.nan,
                    "mean_width": np.nan,
                    "sample_count": 0,
                }
            )
            continue
        rows.append(
            {
                "target": target,
                "group": group_name,
                "bin": int(b),
                "picp_90": float(np.mean(covered[mask])),
                "mean_width": float(np.mean(width[mask])),
                "sample_count": int(np.sum(mask)),
            }
        )

    coverage_df = pd.DataFrame(rows)
    coverage_path = out_dir / "cqr_group_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False, float_format="%.6f")

    summary_payload: dict[str, Any] = {
        "mode": mode,
        "target": target,
        "features": feats,
        "fit": fit_meta,
        "overall_picp_90": float(np.mean(covered)) if covered.size else 0.0,
        "overall_mean_width": float(np.mean(width)) if width.size else 0.0,
        "backend_policy": str(backend_policy),
        "quantile_backend": str(quantile_backend),
        **fallback_meta,
    }
    summary_path = out_dir / "cqr_calibration_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    regime_path = artifact_dir / f"{target}_regime_cqr.json"
    regime_path.write_text(regime.to_json(), encoding="utf-8")

    rac_model = RACCertModel(
        cfg=RACCertConfig(
            alpha=float(alpha),
            n_vol_bins=int(bins),
            vol_window=int(vol_window),
        )
    )
    rac_model.fit(y_cal=y_cal, q_lo_cal=q10_cal, q_hi_cal=q90_cal)
    rac_path = artifact_dir / f"{target}_rac_cert.json"
    rac_path.write_text(rac_model.to_json(), encoding="utf-8")

    fig_path = out_dir / "fig_cqr_group_coverage.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if coverage_df.empty:
        ax.text(0.5, 0.5, "No group coverage rows", ha="center", va="center")
    else:
        for _, row in coverage_df.iterrows():
            if not np.isfinite(row["mean_width"]) or not np.isfinite(row["picp_90"]):
                continue
            ax.scatter(float(row["mean_width"]), float(row["picp_90"]), s=80)
            ax.text(float(row["mean_width"]), float(row["picp_90"]), f" {row['group']}")
        ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean Interval Width")
    ax.set_ylabel("PICP@90")
    ax.set_title("Regime-Aware CQR Coverage vs Width")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    return {
        "coverage_path": str(coverage_path),
        "summary_path": str(summary_path),
        "regime_path": str(regime_path),
        "rac_path": str(rac_path),
        "figure_path": str(fig_path),
    }


def main() -> None:
    args = _build_parser().parse_args()
    payload = train_regime_cqr_artifacts(
        train_path=Path(args.train),
        cal_path=Path(args.cal),
        test_path=Path(args.test),
        out_dir=Path(args.out),
        target=str(args.target),
        features=args.features,
        alpha=float(args.alpha),
        bins=int(args.bins),
        vol_window=int(args.vol_window),
        backend_policy=str(args.backend_policy),
        quantile_backend=str(args.quantile_backend),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
