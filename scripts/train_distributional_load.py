#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orius.forecasting.uncertainty.distributional import (
    NGBoostConfig,
    predict_ngboost_quantiles,
    summarize_interval_quality,
    train_ngboost_distribution,
)


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _feature_cols(df: pd.DataFrame, target: str, user_cols: str | None) -> list[str]:
    if user_cols:
        cols = [c.strip() for c in user_cols.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return cols
    cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric feature columns found; pass --features explicitly")
    return cols


def _group_rows(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, target: str) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lower = np.asarray(lo, dtype=float).reshape(-1)
    upper = np.asarray(hi, dtype=float).reshape(-1)
    vol = pd.Series(y).rolling(window=24, min_periods=6).std().fillna(0.0).to_numpy()
    q1, q2 = np.quantile(vol, [1.0 / 3.0, 2.0 / 3.0]) if len(vol) > 2 else (0.0, 0.0)
    labels = np.where(vol <= q1, "low", np.where(vol <= q2, "med", "high"))
    covered = ((y >= lower) & (y <= upper)).astype(float)
    width = upper - lower

    rows: list[dict[str, float | str | int]] = []
    for label in ("low", "med", "high"):
        mask = labels == label
        rows.append(
            {
                "target": target,
                "group": label,
                "picp_90": float(np.mean(covered[mask])) if np.any(mask) else np.nan,
                "mean_width": float(np.mean(width[mask])) if np.any(mask) else np.nan,
                "sample_count": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


def train_distributional_load(
    *,
    train_path: Path,
    cal_path: Path,
    test_path: Path,
    out_dir: Path,
    target: str = "load_mw",
    features: str | None = None,
) -> dict[str, str | float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("artifacts/uncertainty")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_table(train_path)
    cal_df = _load_table(cal_path)
    test_df = _load_table(test_path)

    if target not in train_df.columns or target not in cal_df.columns or target not in test_df.columns:
        raise ValueError(f"target '{target}' missing in train/cal/test")

    feat_cols = _feature_cols(train_df, target, features)
    train_aug = pd.concat([train_df, cal_df], axis=0, ignore_index=True)
    x_train = train_aug[feat_cols].to_numpy(dtype=float)
    y_train = train_aug[target].to_numpy(dtype=float)
    x_test = test_df[feat_cols].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    model = train_ngboost_distribution(x_train=x_train, y_train=y_train, cfg=NGBoostConfig())
    q = predict_ngboost_quantiles(model, x_test, quantiles=(0.1, 0.5, 0.9))
    q10, q50, q90 = q[0.1], q[0.5], q[0.9]

    npz_path = artifact_dir / f"{target}_ngboost_quantiles.npz"
    np.savez(npz_path, y_true=y_test, q10=q10, q50=q50, q90=q90)

    overall = summarize_interval_quality(y_test, q10, q90)
    dist_rows = _group_rows(y_true=y_test, lo=q10, hi=q90, target=target)
    dist_rows["method"] = "ngboost"

    cqr_path = out_dir / "cqr_group_coverage.csv"
    if cqr_path.exists():
        cqr = pd.read_csv(cqr_path)
        cqr = cqr.rename(columns={"picp_90": "picp_90", "mean_width": "mean_width"})
        cqr = cqr[["target", "group", "picp_90", "mean_width", "sample_count"]].copy()
        cqr["method"] = "regime_cqr"
        compare = pd.concat([cqr, dist_rows], ignore_index=True, sort=False)
    else:
        compare = dist_rows.copy()

    table_path = out_dir / "table_cqr_distributional_compare.csv"
    compare.to_csv(table_path, index=False, float_format="%.6f")

    fig_path = out_dir / "fig_distributional_vs_cqr.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, sub in compare.groupby("method", sort=True):
        ax.plot(sub["group"], sub["picp_90"], marker="o", linestyle="-", label=f"{method} coverage")
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Volatility Group")
    ax.set_ylabel("PICP@90")
    ax.set_title("Distributional vs Regime-CQR Coverage")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    summary_path = out_dir / "distributional_summary.json"
    summary_payload = {
        "target": target,
        "features": feat_cols,
        "overall_picp_90": overall["picp_90"],
        "overall_mean_width": overall["mean_width"],
        "artifact_npz": str(npz_path),
        "table": str(table_path),
        "figure": str(fig_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_payload


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NGBoost distributional load model")
    p.add_argument("--train", required=True)
    p.add_argument("--cal", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--target", default="load_mw")
    p.add_argument("--features", default=None)
    p.add_argument("--out", default="reports/publication")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    payload = train_distributional_load(
        train_path=Path(args.train),
        cal_path=Path(args.cal),
        test_path=Path(args.test),
        out_dir=Path(args.out),
        target=str(args.target),
        features=args.features,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
