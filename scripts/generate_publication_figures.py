#!/usr/bin/env python3
"""
GridPulse Publication Pack — Figure & Table Generator
=====================================================
Generates all 19 publication-level figures (PNG >=300 DPI) and
9 CSV tables listed in the Publication Pack checklist.

Usage:  python scripts/generate_publication_figures.py
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from gridpulse.forecasting.baselines import persistence_24h, moving_average
from gridpulse.forecasting.predict import load_model_bundle
from gridpulse.forecasting.dl_lstm import LSTMForecaster
from gridpulse.forecasting.dl_tcn import TCNForecaster
from gridpulse.forecasting.datasets import SeqConfig, TimeSeriesWindowDataset
from gridpulse.forecasting.uncertainty.conformal import (
    ConformalConfig, ConformalInterval, save_conformal,
)
from gridpulse.optimizer.lp_dispatch import optimize_dispatch
from gridpulse.optimizer.baselines import grid_only_dispatch, naive_battery_dispatch
from gridpulse.optimizer.impact import impact_summary
from gridpulse.anomaly.detect import detect_anomalies
from gridpulse.monitoring.retraining import load_monitoring_config, compute_data_drift
from gridpulse.utils.metrics import rmse, mae, mape, smape, daylight_mape
from gridpulse.utils.scaler import StandardScaler

import torch
from torch.utils.data import DataLoader

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
DPI = 300
C = {
    "actual": "#1f77b4", "forecast": "#ff7f0e", "baseline": "#7f7f7f",
    "optimized": "#2ca02c", "anomaly": "#d62728", "band": "#aec7e8",
    "solar": "#ffbb78", "wind": "#98df8a", "load": "#1f77b4",
}


@dataclass
class Ctx:
    root: Path
    features: Path
    splits: Path
    models: Path
    reports: Path
    figures: Path
    tables: Path
    metrics_dir: Path


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_ctx() -> Ctx:
    r = REPO
    ctx = Ctx(
        root=r, features=r / "data" / "processed" / "features.parquet",
        splits=r / "data" / "processed" / "splits", models=r / "artifacts" / "models",
        reports=r / "reports", figures=r / "reports" / "figures",
        tables=r / "reports" / "tables", metrics_dir=r / "reports" / "metrics",
    )
    _ensure(ctx.figures)
    _ensure(ctx.tables)
    _ensure(ctx.metrics_dir)
    return ctx


def load_splits(ctx):
    return (
        pd.read_parquet(ctx.splits / "train.parquet"),
        pd.read_parquet(ctx.splits / "val.parquet"),
        pd.read_parquet(ctx.splits / "test.parquet"),
    )


def _clean(arr):
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr)
    if not np.isfinite(m):
        m = 0.0
    return np.where(np.isfinite(arr), arr, m)


def _safe(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return float(v)


def _load_yaml(name):
    p = REPO / "configs" / name
    if p.exists():
        return yaml.safe_load(p.read_text()) or {}
    return {}


def _metrics(yt, yp, target=""):
    m = {
        "rmse": rmse(yt, yp), "mae": mae(yt, yp),
        "mape": mape(yt, yp), "smape": smape(yt, yp),
    }
    if target == "solar_mw":
        m["daylight_mape"] = daylight_mape(yt, yp)
    return m


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _gbm_predict(ctx, df, target):
    for p in ctx.models.glob(f"gbm_*_{target}.pkl"):
        b = load_model_bundle(p)
        fc = b.get("feature_cols", [])
        if fc:
            return b["model"].predict(df[fc].to_numpy())
    return None


def _build_torch(bundle):
    mt = bundle.get("model_type")
    fc = bundle.get("feature_cols", [])
    params = bundle.get("model_params", {})
    h = int(bundle.get("horizon", params.get("horizon", 24)))
    if mt == "lstm":
        model = LSTMForecaster(
            n_features=len(fc),
            hidden_size=int(params.get("hidden_size", 128)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
            horizon=h,
        )
    elif mt == "tcn":
        model = TCNForecaster(
            n_features=len(fc),
            num_channels=list(params.get("num_channels", [32, 32, 32])),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unknown model type: {mt}")
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def _seq_predict(bundle, df):
    fc = bundle.get("feature_cols", [])
    tgt = bundle.get("target")
    lb = int(bundle.get("lookback", 168))
    hz = int(bundle.get("horizon", 24))
    if not fc or tgt is None:
        return None, None
    X = df[fc].to_numpy()
    y = df[tgt].to_numpy()
    xs = StandardScaler.from_dict(bundle.get("x_scaler"))
    ys = StandardScaler.from_dict(bundle.get("y_scaler"))
    if xs is not None:
        X = xs.transform(X)
    if ys is not None:
        y = ys.transform(y.reshape(-1, 1)).reshape(-1)
    ds = TimeSeriesWindowDataset(X, y, SeqConfig(lookback=lb, horizon=hz))
    if len(ds) == 0:
        return None, None
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    model = _build_torch(bundle)
    preds_l, trues_l = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds_l.append(model(xb).numpy()[:, -hz:].reshape(-1))
            trues_l.append(yb.numpy().reshape(-1))
    yt = np.concatenate(trues_l)
    yp = np.concatenate(preds_l)
    if ys is not None:
        yt = ys.inverse_transform(yt.reshape(-1, 1)).reshape(-1)
        yp = ys.inverse_transform(yp.reshape(-1, 1)).reshape(-1)
    return yt, yp


def _best_predict(ctx, df, target):
    g = _gbm_predict(ctx, df, target)
    if g is not None:
        return df[target].to_numpy(), g
    for kind in ["lstm", "tcn"]:
        p = ctx.models / f"{kind}_{target}.pt"
        if p.exists():
            b = load_model_bundle(p)
            yt, yp = _seq_predict(b, df)
            if yt is not None:
                return yt, yp
    pers = persistence_24h(df, target)
    return df[target].to_numpy(), pers


# ====================================================================
# TABLES (9 total)
# ====================================================================
def gen_tables(ctx, df_full, train_df, test_df):
    print("\n--- Generating publication tables ---")
    cfg = _load_yaml("optimization.yaml")

    # 1. data_coverage
    ts = pd.to_datetime(df_full["timestamp"])
    rows = []
    for t in ["load_mw", "wind_mw", "solar_mw"]:
        if t not in df_full.columns:
            continue
        nn = int(df_full[t].notna().sum())
        rows.append({
            "target": t, "start": str(ts.iloc[0]), "end": str(ts.iloc[-1]),
            "total_rows": len(df_full), "non_null": nn,
            "coverage_pct": round(nn / len(df_full) * 100, 2),
        })
    pd.DataFrame(rows).to_csv(ctx.tables / "data_coverage.csv", index=False)
    print("  [OK] data_coverage.csv")

    # 2. missingness
    miss = df_full.isnull().sum().reset_index()
    miss.columns = ["column", "missing_count"]
    miss["missing_pct"] = round(miss["missing_count"] / len(df_full) * 100, 4)
    miss.sort_values("missing_pct", ascending=False).to_csv(
        ctx.tables / "missingness.csv", index=False
    )
    print("  [OK] missingness.csv")

    # 3. forecast_point_metrics
    rows = []
    for target in ["load_mw", "wind_mw", "solar_mw"]:
        g = _gbm_predict(ctx, test_df, target)
        if g is not None:
            y = test_df[target].to_numpy()
            mask = np.isfinite(y) & np.isfinite(g)
            m = _metrics(y[mask], g[mask], target)
            m["target"] = target
            m["model"] = "gbm"
            rows.append(m)
        for kind in ["lstm", "tcn"]:
            p = ctx.models / f"{kind}_{target}.pt"
            if p.exists():
                b = load_model_bundle(p)
                yt, yp = _seq_predict(b, test_df)
                if yt is not None:
                    m = _metrics(yt, yp, target)
                    m["target"] = target
                    m["model"] = kind
                    rows.append(m)
        pers = persistence_24h(test_df, target)
        y = test_df[target].to_numpy()
        mask = np.isfinite(y) & np.isfinite(pers)
        m = _metrics(y[mask], pers[mask], target)
        m["target"] = target
        m["model"] = "persistence"
        rows.append(m)
    pd.DataFrame(rows).to_csv(ctx.metrics_dir / "forecast_point_metrics.csv", index=False)
    print("  [OK] forecast_point_metrics.csv")

    # 4. forecast_window_metrics
    df_t = test_df.sort_values("timestamp").reset_index(drop=True)
    window = 7 * 24
    stride = 7 * 24
    rows = []
    for target in ["load_mw", "wind_mw", "solar_mw"]:
        yt = df_t[target].to_numpy()
        gbm = _gbm_predict(ctx, df_t, target)
        pers = persistence_24h(df_t, target)
        mm = {"persistence": pers}
        if gbm is not None:
            mm["gbm"] = gbm
        for mn, preds in mm.items():
            if preds is None:
                continue
            wi = 0
            for s in range(0, len(df_t) - window + 1, stride):
                e = s + window
                y1 = yt[s:e]
                y2 = np.asarray(preds[s:e], dtype=float)
                mask = np.isfinite(y1) & np.isfinite(y2)
                if mask.sum() < window * 0.5:
                    continue
                m = _metrics(y1[mask], y2[mask], target)
                m.update({"target": target, "model": mn, "window_idx": wi})
                rows.append(m)
                wi += 1
    pd.DataFrame(rows).to_csv(ctx.metrics_dir / "forecast_window_metrics.csv", index=False)
    print("  [OK] forecast_window_metrics.csv")

    # 5. forecast_intervals
    int_rows = []
    for target in ["load_mw", "wind_mw", "solar_mw"]:
        yt, yp = _best_predict(ctx, test_df, target)
        if yt is None or yp is None:
            continue
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt2, yp2 = yt[mask], yp[mask]
        ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=False))
        cn = len(yt2) // 3
        ci.fit_calibration(yt2[:cn], yp2[:cn])
        cov = ci.coverage(yt2[cn:], yp2[cn:])
        wid = ci.mean_width(yp2[cn:])
        int_rows.append({
            "target": target, "alpha": 0.10, "picp": round(cov, 4),
            "mpiw": round(wid, 4), "n_test": len(yt2) - cn,
        })
        ad = REPO / "artifacts" / "uncertainty"
        _ensure(ad)
        save_conformal(ad / f"{target}_conformal.json", ci, {"target": target})
    interval_df = pd.DataFrame(int_rows)
    interval_df.to_csv(ctx.metrics_dir / "forecast_intervals.csv", index=False)
    print("  [OK] forecast_intervals.csv")

    # 6. dispatch_kpis_by_day
    df2 = test_df.sort_values("timestamp").reset_index(drop=True)
    n_days = min(len(df2) // 24, 30)
    rows = []
    for d in range(n_days):
        s, e = d * 24, (d + 1) * 24
        if e > len(df2):
            break
        day = df2.iloc[s:e]
        load = _clean(day["load_mw"].to_numpy())
        w = day["wind_mw"].to_numpy() if "wind_mw" in day.columns else np.zeros(24)
        so = day["solar_mw"].to_numpy() if "solar_mw" in day.columns else np.zeros(24)
        renew = _clean(w) + _clean(so)
        price = day["price_eur_mwh"].to_numpy() if "price_eur_mwh" in day.columns else None
        carb = day["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in day.columns else None
        bl = grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carb)
        op = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carb)
        bl_g = bl.get("grid_mw")
        op_g = op.get("grid_mw")
        rows.append({
            "day": d, "date": str(pd.to_datetime(day["timestamp"].iloc[0]).date()),
            "baseline_cost": _safe(bl.get("expected_cost_usd")),
            "optimized_cost": _safe(op.get("expected_cost_usd")),
            "baseline_carbon": _safe(bl.get("carbon_kg")),
            "optimized_carbon": _safe(op.get("carbon_kg")),
            "baseline_peak": _safe(float(np.max(bl_g))) if bl_g is not None else None,
            "optimized_peak": _safe(float(np.max(op_g))) if op_g is not None else None,
        })
    pd.DataFrame(rows).to_csv(ctx.tables / "dispatch_kpis_by_day.csv", index=False)
    print("  [OK] dispatch_kpis_by_day.csv")

    # 7. impact_summary
    df_l7 = df2.tail(7 * 24)
    load7 = _clean(df_l7["load_mw"].to_numpy())
    w7 = df_l7["wind_mw"].to_numpy() if "wind_mw" in df_l7.columns else np.zeros_like(load7)
    s7 = df_l7["solar_mw"].to_numpy() if "solar_mw" in df_l7.columns else np.zeros_like(load7)
    renew7 = _clean(w7) + _clean(s7)
    pr7 = df_l7["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df_l7.columns else None
    cb7 = df_l7["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df_l7.columns else None
    bl7 = grid_only_dispatch(load7, renew7, cfg, price_series=pr7, carbon_series=cb7)
    op7 = optimize_dispatch(load7, renew7, cfg, forecast_price=pr7, forecast_carbon_kg=cb7)
    imp = impact_summary(bl7, op7)
    pd.DataFrame([imp]).to_csv(ctx.reports / "impact_summary.csv", index=False)
    print("  [OK] impact_summary.csv")

    # 8. runtime
    rows = []
    for p in sorted(ctx.models.glob("*")):
        if p.suffix in (".pkl", ".pt"):
            st = os.stat(p)
            rows.append({
                "model_file": p.name, "size_mb": round(st.st_size / 1e6, 2),
                "modified": time.ctime(st.st_mtime),
            })
    pd.DataFrame(rows).to_csv(ctx.metrics_dir / "runtime.csv", index=False)
    print("  [OK] runtime.csv")

    # 9. robustness_summary
    day_r = test_df.sort_values("timestamp").tail(7 * 24).head(24)
    lb = _clean(day_r["load_mw"].to_numpy())
    w_r = day_r["wind_mw"].to_numpy() if "wind_mw" in day_r.columns else np.zeros(24)
    s_r = day_r["solar_mw"].to_numpy() if "solar_mw" in day_r.columns else np.zeros(24)
    rb = _clean(w_r) + _clean(s_r)
    price_r = day_r["price_eur_mwh"].to_numpy() if "price_eur_mwh" in day_r.columns else None
    carb_r = day_r["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in day_r.columns else None
    oracle = optimize_dispatch(lb, rb, cfg, forecast_price=price_r, forecast_carbon_kg=carb_r)
    oc = oracle.get("expected_cost_usd", 0.0) or 0.0
    perts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    nt = 20
    rng = np.random.RandomState(42)
    rows = []
    for pct in perts:
        costs = []
        inf_c = 0
        for _ in range(nt):
            nl = lb * (1 + rng.uniform(-pct, pct, size=len(lb)))
            nr = np.maximum(0, rb * (1 + rng.uniform(-pct, pct, size=len(rb))))
            try:
                plan = optimize_dispatch(nl, nr, cfg, forecast_price=price_r, forecast_carbon_kg=carb_r)
                c = plan.get("expected_cost_usd")
                if c is not None:
                    costs.append(float(c))
                else:
                    inf_c += 1
            except Exception:
                inf_c += 1
        reg = [(c - oc) for c in costs] if oc else costs
        rows.append({
            "perturbation_pct": pct * 100, "n_trials": nt, "infeasible_count": inf_c,
            "infeasible_rate": inf_c / nt,
            "mean_cost": _safe(float(np.mean(costs))) if costs else None,
            "mean_regret": _safe(float(np.mean(reg))) if reg else None,
            "p95_regret": _safe(float(np.percentile(reg, 95))) if len(reg) > 1 else None,
        })
    rob_df = pd.DataFrame(rows)
    rob_df.to_csv(ctx.metrics_dir / "robustness_summary.csv", index=False)
    print("  [OK] robustness_summary.csv")

    return interval_df, rob_df


# ====================================================================
# FIGURES (19 total)
# ====================================================================
def _fig_fva(ctx, test_df, target, num, label, color):
    yt, yp = _best_predict(ctx, test_df, target)
    if yt is None or yp is None:
        print(f"  [SKIP] Fig {num}")
        return
    n = min(7 * 24, len(yt))
    yt, yp = yt[-n:], yp[-n:]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(yt, color=C["actual"], label="Actual")
    ax.plot(yp, color=C["forecast"], label="Forecast", alpha=0.85)
    ax.fill_between(range(n), yt, yp, alpha=0.15, color=C["forecast"])
    ax.set(xlabel="Hour", ylabel=f"{label} (MW)",
           title=f"Fig {num}. Forecast vs Actual - {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fname = f"forecast_vs_actual_{target.replace('_mw', '')}.png"
    fig.savefig(ctx.figures / fname, dpi=DPI)
    plt.close(fig)
    print(f"  [OK] Fig {num}: {fname}")


def fig01(ctx, t):
    _fig_fva(ctx, t, "load_mw", 1, "Load", C["load"])


def fig02(ctx, t):
    _fig_fva(ctx, t, "wind_mw", 2, "Wind", C["wind"])


def fig03(ctx, t):
    _fig_fva(ctx, t, "solar_mw", 3, "Solar", C["solar"])


def fig04(ctx, test_df):
    targets = ["load_mw", "wind_mw", "solar_mw"]
    window = 7 * 24
    stride = 7 * 24
    df = test_df.sort_values("timestamp").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, target in enumerate(targets):
        yt = df[target].to_numpy()
        gbm = _gbm_predict(ctx, df, target)
        pers = persistence_24h(df, target)
        mm = {"Persistence": pers}
        if gbm is not None:
            mm["GBM"] = gbm
        for mn, preds in mm.items():
            if preds is None:
                continue
            rmses = []
            weeks = []
            w = 0
            for s in range(0, len(df) - window + 1, stride):
                e = s + window
                y1 = yt[s:e]
                y2 = np.asarray(preds[s:e], dtype=float)
                mask = np.isfinite(y1) & np.isfinite(y2)
                if mask.sum() < window * 0.3:
                    continue
                rmses.append(rmse(y1[mask], y2[mask]))
                weeks.append(w)
                w += 1
            axes[i].plot(weeks, rmses, marker="o", markersize=3, label=mn)
        axes[i].set(title=target.replace("_mw", "").title(), xlabel="Week", ylabel="RMSE (MW)")
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)
    fig.suptitle("Fig 4. Rolling Backtest RMSE by Week", fontsize=13)
    plt.tight_layout()
    fig.savefig(ctx.figures / "rolling_backtest_rmse_by_week.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 4: rolling_backtest_rmse_by_week.png")


def fig05(ctx, test_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, target in enumerate(["load_mw", "wind_mw", "solar_mw"]):
        yt, yp = _best_predict(ctx, test_df, target)
        if yt is None or yp is None:
            continue
        mask = np.isfinite(yt) & np.isfinite(yp)
        res = yt[mask] - yp[mask]
        axes[i].hist(res, bins=80, color=C["actual"], alpha=0.7, edgecolor="white")
        axes[i].axvline(0, color="red", linestyle="--")
        axes[i].set(title=target.replace("_mw", "").title(), xlabel="Residual (MW)", ylabel="Freq")
        mu, sig = float(np.mean(res)), float(np.std(res))
        axes[i].text(
            0.95, 0.95, f"mu={mu:.1f}\nsig={sig:.1f}",
            transform=axes[i].transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.5),
        )
    fig.suptitle("Fig 5. Residual Distribution", fontsize=13)
    plt.tight_layout()
    fig.savefig(ctx.figures / "error_distribution_residuals.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 5: error_distribution_residuals.png")


def fig06(ctx, test_df):
    yt, yp = _best_predict(ctx, test_df, "load_mw")
    if yt is None or yp is None:
        return
    n = min(len(yt), len(test_df))
    df = test_df.tail(n).copy()
    df["residual"] = np.abs(yt[-n:] - yp[-n:])
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["month"] = ts.dt.month
    pivot = df.pivot_table(values="residual", index="month", columns="hour", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Month")
    ax.set_xticks(range(24))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="Mean |Residual| (MW)")
    ax.set_title("Fig 6. Seasonality Error Heatmap - Load (Month x Hour)")
    fig.savefig(ctx.figures / "seasonality_error_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 6: seasonality_error_heatmap.png")


def fig07(ctx, test_df):
    yt, yp = _best_predict(ctx, test_df, "load_mw")
    if yt is None or yp is None:
        return
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt2, yp2 = yt[mask], yp[mask]
    ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=False))
    cn = len(yt2) // 3
    ci.fit_calibration(yt2[:cn], yp2[:cn])
    lo, hi = ci.predict_interval(yp2[cn:])
    yt3, yp3 = yt2[cn:], yp2[cn:]
    n = min(168, len(yt3))
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, yt3[-n:], color=C["actual"], label="Actual")
    ax.plot(x, yp3[-n:], color=C["forecast"], label="Forecast")
    ax.fill_between(x, lo[-n:], hi[-n:], alpha=0.25, color=C["band"], label="90% PI (Conformal)")
    ax.set(xlabel="Hour", ylabel="Load (MW)",
           title="Fig 7. Prediction Intervals - Load (90% Conformal)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(ctx.figures / "prediction_intervals_load.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 7: prediction_intervals_load.png")


def fig08(ctx, test_df):
    horizons = [1, 3, 6, 12, 24]
    fig, ax = plt.subplots(figsize=(8, 5))
    for target in ["load_mw", "wind_mw", "solar_mw"]:
        yt, yp = _best_predict(ctx, test_df, target)
        if yt is None or yp is None:
            continue
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt2, yp2 = yt[mask], yp[mask]
        cn = len(yt2) // 3
        ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=False))
        ci.fit_calibration(yt2[:cn], yp2[:cn])
        yt3, yp3 = yt2[cn:], yp2[cn:]
        covs = []
        for h in horizons:
            cs = [ci.coverage(yt3[s:s + h], yp3[s:s + h]) for s in range(0, len(yt3) - h + 1, h)]
            covs.append(float(np.mean(cs)) if cs else float("nan"))
        ax.plot(horizons, covs, marker="o", label=target.replace("_mw", "").title())
    ax.axhline(0.90, color="red", linestyle="--", label="Target (90%)")
    ax.set(xlabel="Horizon (hours)", ylabel="Coverage (PICP)", title="Fig 8. Coverage by Horizon")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    fig.savefig(ctx.figures / "coverage_by_horizon.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 8: coverage_by_horizon.png")


def fig09(ctx, test_df):
    horizons = [1, 3, 6, 12, 24]
    fig, ax = plt.subplots(figsize=(8, 5))
    for target in ["load_mw", "wind_mw", "solar_mw"]:
        yt, yp = _best_predict(ctx, test_df, target)
        if yt is None or yp is None:
            continue
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt2, yp2 = yt[mask], yp[mask]
        cn = len(yt2) // 3
        ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=False))
        ci.fit_calibration(yt2[:cn], yp2[:cn])
        yp3 = yp2[cn:]
        wids = []
        for h in horizons:
            ws = [ci.mean_width(yp3[s:s + h]) for s in range(0, len(yp3) - h + 1, h)]
            wids.append(float(np.mean(ws)) if ws else float("nan"))
        ax.plot(horizons, wids, marker="s", label=target.replace("_mw", "").title())
    ax.set(xlabel="Horizon (hours)", ylabel="MPIW (MW)", title="Fig 9. Interval Width by Horizon")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(ctx.figures / "interval_width_by_horizon.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 9: interval_width_by_horizon.png")


def fig10(ctx, test_df):
    yt, yp = _best_predict(ctx, test_df, "load_mw")
    if yt is None or yp is None:
        return
    n = min(len(yt), len(test_df))
    df = test_df.tail(n).copy().reset_index(drop=True)
    fc = [c for c in ["wind_mw", "solar_mw", "hour", "dayofweek"] if c in df.columns]
    feat = df[fc] if fc else None
    anom = detect_anomalies(yt[-n:], yp[-n:], feat)
    combined = anom["combined"]
    pn = min(30 * 24, n)
    ts = pd.to_datetime(df["timestamp"].iloc[-pn:]).values
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts, yt[-pn:], color=C["actual"], label="Actual", linewidth=0.8)
    ax.plot(ts, yp[-pn:], color=C["forecast"], label="Forecast", linewidth=0.8, alpha=0.7)
    am = combined[-pn:]
    if np.any(am):
        ax.scatter(ts[am], yt[-pn:][am], color=C["anomaly"], s=12, zorder=5, label="Anomaly")
    ax.set(xlabel="Time", ylabel="Load (MW)", title="Fig 10. Anomaly Timeline - Load (30 Days)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(ctx.figures / "anomaly_timeline.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 10: anomaly_timeline.png")


def fig11(ctx, test_df):
    yt, yp = _best_predict(ctx, test_df, "load_mw")
    if yt is None or yp is None:
        return
    n = min(len(yt), len(test_df))
    df = test_df.tail(n).copy().reset_index(drop=True)
    anom = detect_anomalies(yt[-n:], yp[-n:])
    zs = anom.get("z_scores", np.zeros(n))
    pn = min(30 * 24, n)
    ts = pd.to_datetime(df["timestamp"].iloc[-pn:]).values
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(ts, zs[-pn:], color=C["actual"], linewidth=0.6)
    ax.axhline(3, color="red", linestyle="--", label="+/-3 sigma")
    ax.axhline(-3, color="red", linestyle="--")
    ax.fill_between(ts, -3, 3, alpha=0.05, color="green")
    ax.set(xlabel="Time", ylabel="Z-Score", title="Fig 11. Residual Z-Score Timeline - Load")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(ctx.figures / "residual_zscore_timeline.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 11: residual_zscore_timeline.png")


def fig12(ctx, test_df):
    cfg = _load_yaml("optimization.yaml")
    df = test_df.sort_values("timestamp").reset_index(drop=True)
    hz = min(7 * 24, len(df))
    win = df.tail(hz)
    load = _clean(win["load_mw"].to_numpy())
    w = win["wind_mw"].to_numpy() if "wind_mw" in win.columns else np.zeros_like(load)
    s = win["solar_mw"].to_numpy() if "solar_mw" in win.columns else np.zeros_like(load)
    renew = _clean(w) + _clean(s)
    price = win["price_eur_mwh"].to_numpy() if "price_eur_mwh" in win.columns else None
    carb = win["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in win.columns else None
    bl = grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carb)
    op = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carb)
    x = np.arange(hz)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(x, bl["grid_mw"], color=C["baseline"], label="Baseline")
    axes[0].plot(x, op["grid_mw"], color=C["optimized"], label="Optimized")
    axes[0].set(ylabel="Grid (MW)", title="Fig 12. Dispatch Comparison (7 Days)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, op["battery_charge_mw"], color=C["actual"], label="Charge")
    axes[1].plot(x, op["battery_discharge_mw"], color=C["forecast"], label="Discharge")
    axes[1].set(ylabel="Battery (MW)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    bl_soc = bl.get("soc_mwh")
    if bl_soc is None:
        bl_soc = np.zeros(hz)
    axes[2].plot(x, bl_soc, color=C["baseline"], linestyle="--", label="Baseline SOC")
    axes[2].plot(x, op["soc_mwh"], color=C["optimized"], label="Optimized SOC")
    axes[2].set(ylabel="SOC (MWh)", xlabel="Hour")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(ctx.figures / "dispatch_compare.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 12: dispatch_compare.png")


def fig13(ctx, test_df):
    cfg = _load_yaml("optimization.yaml")
    df = test_df.sort_values("timestamp").tail(7 * 24)
    load = _clean(df["load_mw"].to_numpy())
    w = df["wind_mw"].to_numpy() if "wind_mw" in df.columns else np.zeros_like(load)
    s = df["solar_mw"].to_numpy() if "solar_mw" in df.columns else np.zeros_like(load)
    renew = _clean(w) + _clean(s)
    price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
    carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
    op = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carb)
    nv = naive_battery_dispatch(load, renew, cfg, price_series=price, carbon_series=carb)
    cap = float(cfg.get("battery", {}).get("capacity_mwh", 20000))
    msoc = float(cfg.get("battery", {}).get("min_soc_mwh", 1000))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(op["soc_mwh"], color=C["optimized"], label="Optimized")
    ax.plot(nv["soc_mwh"], color=C["baseline"], linestyle="--", label="Naive")
    ax.axhline(cap, color="red", linestyle=":", alpha=0.5, label=f"Max ({cap:.0f})")
    ax.axhline(msoc, color="orange", linestyle=":", alpha=0.5, label=f"Min ({msoc:.0f})")
    ax.set(xlabel="Hour", ylabel="SOC (MWh)", title="Fig 13. Battery SOC Trajectory")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(ctx.figures / "soc_trajectory.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 13: soc_trajectory.png")


def fig14(ctx, test_df):
    cfg = _load_yaml("optimization.yaml")
    df = test_df.sort_values("timestamp").tail(7 * 24)
    load = _clean(df["load_mw"].to_numpy())
    w = df["wind_mw"].to_numpy() if "wind_mw" in df.columns else np.zeros_like(load)
    s = df["solar_mw"].to_numpy() if "solar_mw" in df.columns else np.zeros_like(load)
    renew = _clean(w) + _clean(s)
    price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
    carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
    bl = grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carb)
    op = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carb)
    imp = impact_summary(bl, op)
    bl_g = bl.get("grid_mw")
    op_g = op.get("grid_mw")
    bp = float(np.max(bl_g)) if bl_g is not None else 0.0
    opp = float(np.max(op_g)) if op_g is not None else 0.0
    ppct = (bp - opp) / bp * 100 if bp > 0 else 0
    labels = ["Cost\nSavings %", "Carbon\nReduction %", "Peak\nShaving %"]
    vals = [imp.get("cost_savings_pct", 0) or 0, imp.get("carbon_reduction_pct", 0) or 0, ppct]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set(ylabel="Improvement (%)", title="Fig 14. Impact Savings - GridPulse vs Baseline")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(ctx.figures / "impact_savings.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 14: impact_savings.png")


def fig15(ctx, test_df):
    cfg = _load_yaml("optimization.yaml")
    df = test_df.sort_values("timestamp").tail(24)
    load = _clean(df["load_mw"].to_numpy())
    w = df["wind_mw"].to_numpy() if "wind_mw" in df.columns else np.zeros_like(load)
    s = df["solar_mw"].to_numpy() if "solar_mw" in df.columns else np.zeros_like(load)
    renew = _clean(w) + _clean(s)
    price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
    carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
    cws = [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    costs = []
    carbons = []
    for cw in cws:
        sc = copy.deepcopy(cfg)
        sc.setdefault("objective", {})["carbon_weight"] = cw
        try:
            plan = optimize_dispatch(load, renew, sc, forecast_price=price, forecast_carbon_kg=carb)
            costs.append(plan.get("expected_cost_usd", 0) or 0)
            carbons.append(plan.get("carbon_kg", 0) or 0)
        except Exception:
            costs.append(float("nan"))
            carbons.append(float("nan"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(costs, carbons, "o-", color=C["actual"], markersize=8, linewidth=2)
    for i, cw in enumerate(cws):
        if np.isfinite(costs[i]):
            ax.annotate(
                f"w={cw}", (costs[i], carbons[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8,
            )
    ax.set(xlabel="Cost (USD)", ylabel="Carbon (kg CO2)",
           title="Fig 15. Cost vs Carbon Trade-Off Sweep")
    ax.grid(alpha=0.3)
    fig.savefig(ctx.figures / "cost_vs_carbon_tradeoff.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 15: cost_vs_carbon_tradeoff.png")


def fig16(ctx, rob_df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        rob_df["perturbation_pct"].astype(str),
        rob_df["infeasible_rate"] * 100, color=C["anomaly"],
    )
    ax.set(xlabel="Perturbation (%)", ylabel="Infeasible Rate (%)",
           title="Fig 16. Robustness - Infeasible Rate")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(ctx.figures / "robustness_infeasible_rate.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 16: robustness_infeasible_rate.png")


def fig17(ctx, test_df):
    cfg = _load_yaml("optimization.yaml")
    day = test_df.sort_values("timestamp").tail(7 * 24).head(24)
    lb = _clean(day["load_mw"].to_numpy())
    w = day["wind_mw"].to_numpy() if "wind_mw" in day.columns else np.zeros(24)
    s = day["solar_mw"].to_numpy() if "solar_mw" in day.columns else np.zeros(24)
    rb = _clean(w) + _clean(s)
    price = day["price_eur_mwh"].to_numpy() if "price_eur_mwh" in day.columns else None
    carb = day["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in day.columns else None
    oracle = optimize_dispatch(lb, rb, cfg, forecast_price=price, forecast_carbon_kg=carb)
    oc = oracle.get("expected_cost_usd", 0.0) or 0.0
    perts = [0.05, 0.10, 0.15, 0.20, 0.30]
    nt = 30
    rng = np.random.RandomState(42)
    all_reg = {}
    for pct in perts:
        regs = []
        for _ in range(nt):
            nl = lb * (1 + rng.uniform(-pct, pct, size=len(lb)))
            nr = np.maximum(0, rb * (1 + rng.uniform(-pct, pct, size=len(rb))))
            try:
                plan = optimize_dispatch(nl, nr, cfg, forecast_price=price, forecast_carbon_kg=carb)
                c = plan.get("expected_cost_usd")
                if c is not None:
                    regs.append(float(c) - oc)
            except Exception:
                pass
        all_reg[f"{int(pct * 100)}%"] = regs
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [v for v in all_reg.values()]
    tick_labels = list(all_reg.keys())
    ax.boxplot(
        data, patch_artist=True,
        boxprops=dict(facecolor=C["band"], alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    ax.set_xticklabels(tick_labels)
    ax.set(xlabel="Perturbation", ylabel="Regret vs Oracle (USD)",
           title="Fig 17. Robustness - Regret Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(ctx.figures / "robustness_regret_boxplot.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 17: robustness_regret_boxplot.png")


def fig18(ctx, train_df, test_df):
    fcols = [c for c in train_df.columns if c not in {"timestamp", "load_mw", "wind_mw", "solar_mw"}]
    if not fcols:
        print("  [SKIP] Fig 18: no feature columns")
        return
    mcfg = load_monitoring_config()
    thr = float(mcfg.get("data_drift", {}).get("p_value_threshold", 0.01))
    window = 7 * 24
    stride = 7 * 24
    df = test_df.sort_values("timestamp").reset_index(drop=True)
    weeks = []
    drift_fracs = []
    for start in range(0, len(df) - window + 1, stride):
        current = df.iloc[start:start + window]
        drift = compute_data_drift(train_df, current, fcols, thr)
        nd = sum(1 for c in drift.get("columns", {}).values() if c.get("p_value", 1.0) < thr)
        drift_fracs.append(nd / max(len(fcols), 1) * 100)
        weeks.append(len(weeks))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weeks, drift_fracs, marker="o", color=C["actual"])
    ax.axhline(50, color="red", linestyle="--", label="50% threshold")
    ax.set(xlabel="Week", ylabel="Features Drifted (%)",
           title="Fig 18. Data Drift (KS Test) Over Time")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(ctx.figures / "data_drift_ks_over_time.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 18: data_drift_ks_over_time.png")


def fig19(ctx, test_df):
    window = 7 * 24
    stride = 7 * 24
    df = test_df.sort_values("timestamp").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for target in ["load_mw", "wind_mw", "solar_mw"]:
        gbm = _gbm_predict(ctx, df, target)
        if gbm is None:
            continue
        yt = df[target].to_numpy()
        weeks = []
        rmses = []
        for s in range(0, len(df) - window + 1, stride):
            e = s + window
            y1 = yt[s:e]
            y2 = gbm[s:e]
            mask = np.isfinite(y1) & np.isfinite(y2)
            if mask.sum() < window * 0.3:
                continue
            rmses.append(rmse(y1[mask], y2[mask]))
            weeks.append(len(weeks))
        ax.plot(weeks, rmses, marker="o", markersize=4, label=target.replace("_mw", "").title())
    ax.set(xlabel="Week", ylabel="RMSE (MW)", title="Fig 19. Model Drift Over Time (GBM)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(ctx.figures / "model_drift_metric_over_time.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 19: model_drift_metric_over_time.png")


# ====================================================================
# MARKDOWN REPORTS
# ====================================================================
def build_reports(ctx, test_df, interval_df, rob_df):
    print("\n--- Generating markdown reports ---")
    yt, yp = _best_predict(ctx, test_df, "load_mw")
    if yt is not None and yp is not None:
        n = min(len(yt), len(test_df))
        fc = [c for c in ["wind_mw", "solar_mw", "hour", "dayofweek"] if c in test_df.columns]
        feat = test_df.tail(n).reset_index(drop=True)[fc] if fc else None
        anom = detect_anomalies(yt[-n:], yp[-n:], feat)
        na = int(np.sum(anom["combined"]))
        lines = [
            "# Anomaly Detection Report\n\n",
            f"- Total points: {n}\n",
            f"- Anomalies: {na} ({na / n * 100:.2f}%)\n\n",
            "![](figures/anomaly_timeline.png)\n\n",
            "![](figures/residual_zscore_timeline.png)\n",
        ]
        (ctx.reports / "anomaly_report.md").write_text("".join(lines))
        print("  [OK] anomaly_report.md")

    try:
        rob_md = rob_df.to_markdown(index=False)
    except ImportError:
        rob_md = rob_df.to_string(index=False)
    lines = [
        "# Decision Robustness Report\n\n", rob_md,
        "\n\n![](figures/robustness_infeasible_rate.png)\n\n",
        "![](figures/robustness_regret_boxplot.png)\n",
    ]
    (ctx.reports / "decision_robustness.md").write_text("".join(lines))
    print("  [OK] decision_robustness.md")

    try:
        int_md = interval_df.to_markdown(index=False)
    except ImportError:
        int_md = interval_df.to_string(index=False)
    lines = [
        "# Forecast Interval Report\n\n", int_md,
        "\n\n![](figures/prediction_intervals_load.png)\n\n",
        "![](figures/coverage_by_horizon.png)\n\n",
        "![](figures/interval_width_by_horizon.png)\n",
    ]
    (ctx.reports / "forecast_intervals.md").write_text("".join(lines))
    print("  [OK] forecast_intervals.md")


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("GridPulse Publication Pack - Figure & Table Generator")
    print("=" * 60)
    ctx = make_ctx()
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_splits(ctx)
    df_full = pd.read_parquet(ctx.features)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    interval_df, rob_df = gen_tables(ctx, df_full, train_df, test_df)

    print("\n--- Generating 19 publication figures ---")
    fig01(ctx, test_df)
    fig02(ctx, test_df)
    fig03(ctx, test_df)
    fig04(ctx, test_df)
    fig05(ctx, test_df)
    fig06(ctx, test_df)
    fig07(ctx, test_df)
    fig08(ctx, test_df)
    fig09(ctx, test_df)
    fig10(ctx, test_df)
    fig11(ctx, test_df)
    fig12(ctx, test_df)
    fig13(ctx, test_df)
    fig14(ctx, test_df)
    fig15(ctx, test_df)
    fig16(ctx, rob_df)
    fig17(ctx, test_df)
    fig18(ctx, train_df, test_df)
    fig19(ctx, test_df)

    build_reports(ctx, test_df, interval_df, rob_df)

    print("\n" + "=" * 60)
    print("CHECKLIST")
    print("=" * 60)
    figs = [
        "forecast_vs_actual_load.png", "forecast_vs_actual_wind.png",
        "forecast_vs_actual_solar.png", "rolling_backtest_rmse_by_week.png",
        "error_distribution_residuals.png", "seasonality_error_heatmap.png",
        "prediction_intervals_load.png", "coverage_by_horizon.png",
        "interval_width_by_horizon.png", "anomaly_timeline.png",
        "residual_zscore_timeline.png", "dispatch_compare.png", "soc_trajectory.png",
        "impact_savings.png", "cost_vs_carbon_tradeoff.png",
        "robustness_infeasible_rate.png", "robustness_regret_boxplot.png",
        "data_drift_ks_over_time.png", "model_drift_metric_over_time.png",
    ]
    tbls = [
        "tables/data_coverage.csv", "tables/missingness.csv",
        "metrics/forecast_point_metrics.csv", "metrics/forecast_window_metrics.csv",
        "metrics/forecast_intervals.csv", "impact_summary.csv",
        "tables/dispatch_kpis_by_day.csv", "metrics/runtime.csv",
        "metrics/robustness_summary.csv",
    ]
    ok = True
    print("\nFigures:")
    for f in figs:
        exists = (ctx.figures / f).exists()
        ok = ok and exists
        print(f"  {'OK' if exists else 'MISSING'}  {f}")
    print("\nTables:")
    for t in tbls:
        exists = (ctx.reports / t).exists()
        ok = ok and exists
        print(f"  {'OK' if exists else 'MISSING'}  {t}")
    status = "ALL 19 FIGURES + 9 TABLES GENERATED" if ok else "SOME ARTIFACTS MISSING"
    print(f"\n{status}")
    print("=" * 60)


if __name__ == "__main__":
    main()