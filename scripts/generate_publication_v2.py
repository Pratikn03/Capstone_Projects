#!/usr/bin/env python3
"""
GridPulse Publication Pack v2 — 2-Country Figures & Tables
==========================================================
Generates 16 publication figures and 8 tables comparing
Germany (OPSD) and USA (EIA-930) results.

Usage:  python scripts/generate_publication_v2.py
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from gridpulse.optimizer.baselines import (
    grid_only_dispatch, naive_battery_dispatch,
    peak_shaving_dispatch, greedy_price_dispatch,
)
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
PAL = {
    "de": "#1f77b4", "us": "#ff7f0e", "actual": "#1f77b4",
    "forecast": "#ff7f0e", "baseline": "#7f7f7f", "optimized": "#2ca02c",
    "anomaly": "#d62728", "band": "#aec7e8", "solar": "#ffbb78",
    "wind": "#98df8a", "load": "#1f77b4",
}


# ── Dataset abstraction ────────────────────────────────────────
@dataclass
class Dataset:
    label: str           # "Germany (OPSD)" or "USA (EIA-930)"
    short: str           # "DE" or "US"
    color: str
    features_path: Path
    splits_path: Path
    models_path: Path
    reports_path: Path
    targets: List[str] = field(default_factory=lambda: ["load_mw", "wind_mw", "solar_mw"])
    has_price: bool = False
    has_carbon: bool = False
    df_full: Optional[pd.DataFrame] = None
    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None


def load_dataset(ds: Dataset):
    ds.df_full = pd.read_parquet(ds.features_path)
    ds.train_df = pd.read_parquet(ds.splits_path / "train.parquet")
    ds.val_df = pd.read_parquet(ds.splits_path / "val.parquet")
    ds.test_df = pd.read_parquet(ds.splits_path / "test.parquet")
    ds.has_price = "price_eur_mwh" in ds.df_full.columns
    ds.has_carbon = "carbon_kg_per_mwh" in ds.df_full.columns


OUT = REPO / "reports" / "publication"
FIG_DIR = OUT / "figures"
TBL_DIR = OUT / "tables"


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ── Helpers (same as v1) ──────────────────────────────────────
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
    m = {"rmse": rmse(yt, yp), "mae": mae(yt, yp),
         "mape": mape(yt, yp), "smape": smape(yt, yp)}
    if target == "solar_mw":
        try:
            m["daylight_mape"] = daylight_mape(yt, yp)
        except Exception:
            pass
    return m


def _gbm_predict(ds: Dataset, df, target):
    for p in ds.models_path.glob(f"gbm_*_{target}.pkl"):
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
            n_features=len(fc), hidden_size=int(params.get("hidden_size", 128)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)), horizon=h,
        )
    elif mt == "tcn":
        model = TCNForecaster(
            n_features=len(fc),
            num_channels=list(params.get("num_channels", [32, 32, 32])),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unknown: {mt}")
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
    ds_w = TimeSeriesWindowDataset(X, y, SeqConfig(lookback=lb, horizon=hz))
    if len(ds_w) == 0:
        return None, None
    dl = DataLoader(ds_w, batch_size=256, shuffle=False)
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


def _best_predict(ds: Dataset, df, target):
    g = _gbm_predict(ds, df, target)
    if g is not None:
        return df[target].to_numpy(), g
    for kind in ["lstm", "tcn"]:
        p = ds.models_path / f"{kind}_{target}.pt"
        if p.exists():
            b = load_model_bundle(p)
            yt, yp = _seq_predict(b, df)
            if yt is not None:
                return yt, yp
    pers = persistence_24h(df, target)
    return df[target].to_numpy(), pers


# ====================================================================
# TABLES
# ====================================================================
def table1_dataset_summary(datasets: List[Dataset]):
    """Table 1: Dataset Summary (Germany vs USA)"""
    rows = []
    for ds in datasets:
        ts = pd.to_datetime(ds.df_full["timestamp"])
        for t in ds.targets:
            if t not in ds.df_full.columns:
                continue
            nn = int(ds.df_full[t].notna().sum())
            rows.append({
                "Dataset": ds.label, "Country": ds.short,
                "Start": str(ts.iloc[0].date()), "End": str(ts.iloc[-1].date()),
                "Rows": len(ds.df_full), "Signal": t,
                "Non-Null": nn, "Coverage%": round(nn / len(ds.df_full) * 100, 2),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "table1_dataset_summary.csv", index=False)
    print("  [OK] Table 1: dataset summary")
    return df


def table2_forecast_metrics(ds: Dataset, table_num: int):
    """Table 2/3: Forecast Metrics per country"""
    rows = []
    for target in ds.targets:
        if target not in ds.test_df.columns:
            continue
        # GBM
        g = _gbm_predict(ds, ds.test_df, target)
        if g is not None:
            y = ds.test_df[target].to_numpy()
            mask = np.isfinite(y) & np.isfinite(g)
            m = _metrics(y[mask], g[mask], target)
            m.update({"target": target, "model": "GBM"})
            rows.append(m)
        # LSTM / TCN
        for kind in ["lstm", "tcn"]:
            p = ds.models_path / f"{kind}_{target}.pt"
            if p.exists():
                b = load_model_bundle(p)
                yt, yp = _seq_predict(b, ds.test_df)
                if yt is not None:
                    m = _metrics(yt, yp, target)
                    m.update({"target": target, "model": kind.upper()})
                    rows.append(m)
        # Persistence
        pers = persistence_24h(ds.test_df, target)
        y = ds.test_df[target].to_numpy()
        mask = np.isfinite(y) & np.isfinite(pers)
        m = _metrics(y[mask], pers[mask], target)
        m.update({"target": target, "model": "Persistence"})
        rows.append(m)
    df = pd.DataFrame(rows)
    fname = f"table{table_num}_forecast_metrics_{ds.short.lower()}.csv"
    df.to_csv(TBL_DIR / fname, index=False)
    print(f"  [OK] Table {table_num}: forecast metrics ({ds.short})")
    return df


def table4_uncertainty(datasets: List[Dataset]):
    """Table 4: Uncertainty Calibration"""
    rows = []
    for ds in datasets:
        for target in ds.targets:
            if target not in ds.test_df.columns:
                continue
            yt, yp = _best_predict(ds, ds.test_df, target)
            if yt is None or yp is None:
                continue
            mask = np.isfinite(yt) & np.isfinite(yp)
            yt2, yp2 = yt[mask], yp[mask]
            ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=False))
            cn = len(yt2) // 3
            ci.fit_calibration(yt2[:cn], yp2[:cn])
            cov = ci.coverage(yt2[cn:], yp2[cn:])
            wid = ci.mean_width(yp2[cn:])
            rows.append({
                "Dataset": ds.short, "Target": target, "Alpha": 0.10,
                "PICP": round(cov, 4), "MPIW": round(wid, 4),
                "N_test": len(yt2) - cn,
            })
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "table4_uncertainty.csv", index=False)
    print("  [OK] Table 4: uncertainty calibration")
    return df


def table5_optimization_impact(ds_de: Dataset):
    """Table 5: Optimization Impact (Germany)"""
    cfg = _load_yaml("optimization.yaml")
    df = ds_de.test_df.sort_values("timestamp").tail(7 * 24)
    load = _clean(df["load_mw"].to_numpy())
    w = _clean(df["wind_mw"].to_numpy()) if "wind_mw" in df.columns else np.zeros_like(load)
    s = _clean(df["solar_mw"].to_numpy()) if "solar_mw" in df.columns else np.zeros_like(load)
    renew = w + s
    price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
    carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
    bl = grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carb)
    op = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carb)
    imp = impact_summary(bl, op)
    rows = [
        {"Metric": "Cost (USD)", "Baseline": _safe(bl.get("expected_cost_usd")),
         "GridPulse": _safe(op.get("expected_cost_usd")),
         "Delta": f"{imp.get('cost_savings_pct', 0) or 0:.2f}%"},
        {"Metric": "Carbon (kg)", "Baseline": _safe(bl.get("carbon_kg")),
         "GridPulse": _safe(op.get("carbon_kg")),
         "Delta": f"{imp.get('carbon_reduction_pct', 0) or 0:.2f}%"},
        {"Metric": "Peak (MW)",
         "Baseline": _safe(float(np.max(bl["grid_mw"]))) if bl.get("grid_mw") is not None else None,
         "GridPulse": _safe(float(np.max(op["grid_mw"]))) if op.get("grid_mw") is not None else None,
         "Delta": ""},
    ]
    df_out = pd.DataFrame(rows)
    df_out.to_csv(TBL_DIR / "table5_optimization_impact.csv", index=False)
    print("  [OK] Table 5: optimization impact")
    return df_out, bl, op, cfg, imp


def table6_robustness(datasets: List[Dataset]):
    """Table 6: Robustness Summary"""
    cfg = _load_yaml("optimization.yaml")
    all_rows = []
    for ds in datasets:
        df = ds.test_df.sort_values("timestamp").tail(7 * 24).head(24)
        lb = _clean(df["load_mw"].to_numpy())
        w = _clean(df["wind_mw"].to_numpy()) if "wind_mw" in df.columns else np.zeros(24)
        s = _clean(df["solar_mw"].to_numpy()) if "solar_mw" in df.columns else np.zeros(24)
        rb = w + s
        price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
        carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
        oracle = optimize_dispatch(lb, rb, cfg, forecast_price=price, forecast_carbon_kg=carb)
        oc = oracle.get("expected_cost_usd", 0.0) or 0.0
        perts = [0.0, 0.05, 0.10, 0.20, 0.30]
        nt = 20
        rng = np.random.RandomState(42)
        for pct in perts:
            costs = []
            inf_c = 0
            for _ in range(nt):
                nl = lb * (1 + rng.uniform(-pct, pct, size=len(lb)))
                nr = np.maximum(0, rb * (1 + rng.uniform(-pct, pct, size=len(rb))))
                try:
                    plan = optimize_dispatch(nl, nr, cfg, forecast_price=price, forecast_carbon_kg=carb)
                    c = plan.get("expected_cost_usd")
                    if c is not None:
                        costs.append(float(c))
                    else:
                        inf_c += 1
                except Exception:
                    inf_c += 1
            reg = [(c - oc) for c in costs] if oc else costs
            all_rows.append({
                "Dataset": ds.short, "Perturbation%": pct * 100,
                "Infeasible%": round(inf_c / nt * 100, 1),
                "Mean_Regret": _safe(float(np.mean(reg))) if reg else None,
            })
    df = pd.DataFrame(all_rows)
    df.to_csv(TBL_DIR / "table6_robustness.csv", index=False)
    print("  [OK] Table 6: robustness")
    return df


def table7_baseline_comparison(ds_de: Dataset):
    """Table 7: Baseline Comparison (Grid-only, Heuristic, Oracle, GridPulse)"""
    cfg = _load_yaml("optimization.yaml")
    df = ds_de.test_df.sort_values("timestamp").tail(7 * 24)
    load = _clean(df["load_mw"].to_numpy())
    w = _clean(df["wind_mw"].to_numpy()) if "wind_mw" in df.columns else np.zeros_like(load)
    s = _clean(df["solar_mw"].to_numpy()) if "solar_mw" in df.columns else np.zeros_like(load)
    renew = w + s
    price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
    carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
    policies = {
        "Grid-Only": grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carb),
        "Naive Battery": naive_battery_dispatch(load, renew, cfg, price_series=price, carbon_series=carb),
        "GridPulse (LP)": optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carb),
    }
    rows = []
    for name, plan in policies.items():
        gm = plan.get("grid_mw")
        rows.append({
            "Policy": name,
            "Cost_USD": _safe(plan.get("expected_cost_usd")),
            "Carbon_kg": _safe(plan.get("carbon_kg")),
            "Peak_MW": _safe(float(np.max(gm))) if gm is not None else None,
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(TBL_DIR / "table7_baseline_comparison.csv", index=False)
    print("  [OK] Table 7: baseline comparison")
    return df_out


def table8_runtime(datasets: List[Dataset]):
    """Table 8: Runtime & Efficiency"""
    rows = []
    for ds in datasets:
        n_models = 0
        total_size = 0
        for p in ds.models_path.glob("*"):
            if p.suffix in (".pkl", ".pt"):
                n_models += 1
                total_size += p.stat().st_size
        rows.append({
            "Dataset": ds.short, "Train_rows": len(ds.train_df),
            "Test_rows": len(ds.test_df), "N_models": n_models,
            "Total_model_MB": round(total_size / 1e6, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "table8_runtime.csv", index=False)
    print("  [OK] Table 8: runtime")
    return df


# ====================================================================
# FIGURES
# ====================================================================

def fig01_geographic_scope(datasets: List[Dataset]):
    """Fig 1. Data Coverage Timeline"""
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, ds in enumerate(datasets):
        ts = pd.to_datetime(ds.df_full["timestamp"])
        start, end = ts.iloc[0], ts.iloc[-1]
        ax.barh(i, (end - start).days, left=start, height=0.5, color=ds.color, alpha=0.8,
                label=f"{ds.label} ({len(ds.df_full):,} rows)")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([d.label for d in datasets])
    ax.set_xlabel("Date")
    ax.set_title("Fig 1. Geographic Scope & Data Coverage")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(FIG_DIR / "fig01_geographic_scope.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 1: geographic scope")


def fig02_load_renewable_profiles(datasets: List[Dataset]):
    """Fig 2. Load & Renewable Profiles (one-week typical)"""
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        df = ds.test_df.sort_values("timestamp").tail(7 * 24)
        x = np.arange(len(df))
        ax.plot(x, df["load_mw"].to_numpy(), label="Load", color=PAL["load"])
        if "wind_mw" in df.columns:
            ax.plot(x, df["wind_mw"].to_numpy(), label="Wind", color=PAL["wind"], alpha=0.8)
        if "solar_mw" in df.columns:
            ax.plot(x, df["solar_mw"].to_numpy(), label="Solar", color=PAL["solar"], alpha=0.8)
        ax.set(xlabel="Hour", ylabel="MW", title=ds.label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Fig 2. Load & Renewable Profiles (Typical Week)", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig02_load_renewable_profiles.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 2: load & renewable profiles")


def fig03_05_forecast_vs_actual(datasets: List[Dataset]):
    """Fig 3-5. Forecast vs Actual (7-Day Window) per target per country"""
    targets = ["load_mw", "wind_mw", "solar_mw"]
    labels = ["Load", "Wind", "Solar"]
    fig, axes = plt.subplots(len(targets), len(datasets), figsize=(7 * len(datasets), 3.5 * len(targets)))
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)
    for j, ds in enumerate(datasets):
        for i, (target, lbl) in enumerate(zip(targets, labels)):
            ax = axes[i, j]
            if target not in ds.test_df.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{lbl} — {ds.short}")
                continue
            yt, yp = _best_predict(ds, ds.test_df, target)
            if yt is None:
                continue
            n = min(7 * 24, len(yt))
            yt, yp = yt[-n:], yp[-n:]
            ax.plot(yt, color=PAL["actual"], label="Actual")
            ax.plot(yp, color=PAL["forecast"], label="Forecast", alpha=0.85)
            ax.fill_between(range(n), yt, yp, alpha=0.12, color=PAL["forecast"])
            mask = np.isfinite(yt) & np.isfinite(yp)
            r = rmse(yt[mask], yp[mask])
            ax.set_title(f"{lbl} — {ds.short} (RMSE={r:.1f})", fontsize=11)
            ax.set(xlabel="Hour", ylabel="MW")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("Fig 3-5. Forecast vs Actual (7-Day Window)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig03_05_forecast_vs_actual.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 3-5: forecast vs actual")


def fig06_rolling_backtest(datasets: List[Dataset]):
    """Fig 6. Rolling Backtest RMSE (Germany vs USA)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 7 * 24
    stride = 7 * 24
    for ds in datasets:
        target = "load_mw"
        df = ds.test_df.sort_values("timestamp").reset_index(drop=True)
        gbm = _gbm_predict(ds, df, target)
        if gbm is None:
            gbm = persistence_24h(df, target)
        yt = df[target].to_numpy()
        weeks = []
        rmses = []
        for s in range(0, len(df) - window + 1, stride):
            e = s + window
            y1 = yt[s:e]
            y2 = np.asarray(gbm[s:e], dtype=float)
            mask = np.isfinite(y1) & np.isfinite(y2)
            if mask.sum() < window * 0.3:
                continue
            rmses.append(rmse(y1[mask], y2[mask]))
            weeks.append(len(weeks))
        ax.plot(weeks, rmses, marker="o", markersize=4, color=ds.color, label=f"{ds.label}")
    ax.set(xlabel="Week", ylabel="RMSE (MW)", title="Fig 6. Rolling Backtest RMSE — Load (GBM)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig06_rolling_backtest_rmse.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 6: rolling backtest")


def fig07_error_seasonality(datasets: List[Dataset]):
    """Fig 7. Error Seasonality Heatmap (Month x Hour)"""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        yt, yp = _best_predict(ds, ds.test_df, "load_mw")
        if yt is None:
            continue
        n = min(len(yt), len(ds.test_df))
        df = ds.test_df.tail(n).copy()
        df["residual"] = np.abs(yt[-n:] - yp[-n:])
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["month"] = ts.dt.month
        pivot = df.pivot_table(values="residual", index="month", columns="hour", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Month")
        ax.set_xticks(range(24))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="|Residual| MW")
        ax.set_title(f"{ds.label}")
    fig.suptitle("Fig 7. Error Seasonality Heatmap — Load (Month x Hour)", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig07_error_seasonality.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 7: error seasonality")


def fig08_conformal_intervals(datasets: List[Dataset]):
    """Fig 8. Conformal Prediction Intervals (Germany vs USA)"""
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        yt, yp = _best_predict(ds, ds.test_df, "load_mw")
        if yt is None:
            continue
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt2, yp2 = yt[mask], yp[mask]
        ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=False))
        cn = len(yt2) // 3
        ci.fit_calibration(yt2[:cn], yp2[:cn])
        lo, hi = ci.predict_interval(yp2[cn:])
        yt3, yp3 = yt2[cn:], yp2[cn:]
        n = min(168, len(yt3))
        x = np.arange(n)
        ax.plot(x, yt3[-n:], color=PAL["actual"], label="Actual")
        ax.plot(x, yp3[-n:], color=PAL["forecast"], label="Forecast")
        ax.fill_between(x, lo[-n:], hi[-n:], alpha=0.25, color=PAL["band"], label="90% PI")
        cov = ci.coverage(yt3, yp3)
        ax.set(xlabel="Hour", ylabel="Load (MW)", title=f"{ds.label} (PICP={cov:.3f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Fig 8. Conformal Prediction Intervals — Load (90%)", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig08_conformal_intervals.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 8: conformal intervals")


def fig09_coverage_vs_horizon(datasets: List[Dataset]):
    """Fig 9. Coverage vs Horizon"""
    horizons = [1, 3, 6, 12, 24]
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds in datasets:
        yt, yp = _best_predict(ds, ds.test_df, "load_mw")
        if yt is None:
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
        ax.plot(horizons, covs, marker="o", color=ds.color, label=ds.label)
    ax.axhline(0.90, color="red", linestyle="--", label="Target 90%")
    ax.set(xlabel="Horizon (hours)", ylabel="PICP", title="Fig 9. Coverage vs Horizon — Load")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    fig.savefig(FIG_DIR / "fig09_coverage_vs_horizon.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 9: coverage vs horizon")


def fig10_anomaly_timeline(datasets: List[Dataset]):
    """Fig 10. Anomaly Timeline (30-day)"""
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 4 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        yt, yp = _best_predict(ds, ds.test_df, "load_mw")
        if yt is None:
            continue
        n = min(len(yt), len(ds.test_df))
        df = ds.test_df.tail(n).copy().reset_index(drop=True)
        anom = detect_anomalies(yt[-n:], yp[-n:])
        combined = anom["combined"]
        pn = min(30 * 24, n)
        ts = pd.to_datetime(df["timestamp"].iloc[-pn:]).values
        ax.plot(ts, yt[-pn:], color=PAL["actual"], label="Actual", linewidth=0.8)
        ax.plot(ts, yp[-pn:], color=PAL["forecast"], label="Forecast", linewidth=0.8, alpha=0.7)
        am = combined[-pn:]
        if np.any(am):
            ax.scatter(ts[am], yt[-pn:][am], color=PAL["anomaly"], s=12, zorder=5, label="Anomaly")
        ax.set(xlabel="Time", ylabel="Load (MW)", title=f"{ds.label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Fig 10. Anomaly Timeline — Load (30 Days)", fontsize=13)
    plt.tight_layout()
    fig.autofmt_xdate()
    fig.savefig(FIG_DIR / "fig10_anomaly_timeline.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 10: anomaly timeline")


def fig11_dispatch_comparison(ds_de: Dataset):
    """Fig 11. Dispatch Comparison (Germany)"""
    cfg = _load_yaml("optimization.yaml")
    df = ds_de.test_df.sort_values("timestamp").tail(7 * 24)
    load = _clean(df["load_mw"].to_numpy())
    w = _clean(df["wind_mw"].to_numpy()) if "wind_mw" in df.columns else np.zeros_like(load)
    s = _clean(df["solar_mw"].to_numpy()) if "solar_mw" in df.columns else np.zeros_like(load)
    renew = w + s
    price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
    carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
    bl = grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carb)
    op = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carb)
    x = np.arange(len(load))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(x, bl["grid_mw"], color=PAL["baseline"], label="Grid-Only Baseline")
    axes[0].plot(x, op["grid_mw"], color=PAL["optimized"], label="GridPulse Optimized")
    axes[0].set(ylabel="Grid Import (MW)", title="Fig 11. Dispatch Comparison — Germany (7 Days)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, op["battery_charge_mw"], color=PAL["actual"], label="Charge")
    axes[1].plot(x, op["battery_discharge_mw"], color=PAL["forecast"], label="Discharge")
    axes[1].set(ylabel="Battery (MW)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    bl_soc = bl.get("soc_mwh")
    if bl_soc is None:
        bl_soc = np.zeros(len(load))
    axes[2].plot(x, bl_soc, color=PAL["baseline"], linestyle="--", label="Baseline SOC")
    axes[2].plot(x, op["soc_mwh"], color=PAL["optimized"], label="Optimized SOC")
    axes[2].set(ylabel="SOC (MWh)", xlabel="Hour")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig11_dispatch_comparison.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 11: dispatch comparison")


def fig12_soc_trajectory(ds_de: Dataset):
    """Fig 12. Battery SOC Trajectory"""
    cfg = _load_yaml("optimization.yaml")
    df = ds_de.test_df.sort_values("timestamp").tail(7 * 24)
    load = _clean(df["load_mw"].to_numpy())
    w = _clean(df["wind_mw"].to_numpy()) if "wind_mw" in df.columns else np.zeros_like(load)
    s = _clean(df["solar_mw"].to_numpy()) if "solar_mw" in df.columns else np.zeros_like(load)
    renew = w + s
    price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
    carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
    op = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carb)
    nv = naive_battery_dispatch(load, renew, cfg, price_series=price, carbon_series=carb)
    cap = float(cfg.get("battery", {}).get("capacity_mwh", 20000))
    msoc = float(cfg.get("battery", {}).get("min_soc_mwh", 1000))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(op["soc_mwh"], color=PAL["optimized"], label="GridPulse (LP)")
    ax.plot(nv["soc_mwh"], color=PAL["baseline"], linestyle="--", label="Naive Heuristic")
    ax.axhline(cap, color="red", linestyle=":", alpha=0.5, label=f"Max ({cap:,.0f})")
    ax.axhline(msoc, color="orange", linestyle=":", alpha=0.5, label=f"Min ({msoc:,.0f})")
    ax.set(xlabel="Hour", ylabel="SOC (MWh)", title="Fig 12. Battery SOC Trajectory — Germany")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig12_soc_trajectory.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 12: SOC trajectory")


def fig13_cost_carbon_tradeoff(ds_de: Dataset):
    """Fig 13. Cost vs Carbon Trade-off Curve"""
    cfg = _load_yaml("optimization.yaml")
    df = ds_de.test_df.sort_values("timestamp").tail(24)
    load = _clean(df["load_mw"].to_numpy())
    w = _clean(df["wind_mw"].to_numpy()) if "wind_mw" in df.columns else np.zeros_like(load)
    s = _clean(df["solar_mw"].to_numpy()) if "solar_mw" in df.columns else np.zeros_like(load)
    renew = w + s
    price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
    carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
    cws = [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
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
    ax.plot(costs, carbons, "o-", color=PAL["de"], markersize=8, linewidth=2)
    for i, cw in enumerate(cws):
        if np.isfinite(costs[i]):
            ax.annotate(f"w={cw}", (costs[i], carbons[i]), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)
    ax.set(xlabel="Cost (USD)", ylabel="Carbon (kg CO₂)",
           title="Fig 13. Cost vs Carbon Trade-Off — Germany")
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig13_cost_carbon_tradeoff.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 13: cost-carbon tradeoff")


def fig14_savings_sensitivity(ds_de: Dataset):
    """Fig 14. Savings Sensitivity (Battery Size Sweep)"""
    cfg = _load_yaml("optimization.yaml")
    df = ds_de.test_df.sort_values("timestamp").tail(24)
    load = _clean(df["load_mw"].to_numpy())
    w = _clean(df["wind_mw"].to_numpy()) if "wind_mw" in df.columns else np.zeros_like(load)
    s = _clean(df["solar_mw"].to_numpy()) if "solar_mw" in df.columns else np.zeros_like(load)
    renew = w + s
    price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
    carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
    bl = grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carb)
    bl_cost = bl.get("expected_cost_usd", 0) or 0
    caps = [1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000]
    savings_pct = []
    for cap in caps:
        sc = copy.deepcopy(cfg)
        sc.setdefault("battery", {})["capacity_mwh"] = cap
        sc["battery"]["max_power_mw"] = cap // 4
        sc["battery"]["initial_soc_mwh"] = cap // 2
        sc["battery"]["min_soc_mwh"] = int(cap * 0.05)
        try:
            plan = optimize_dispatch(load, renew, sc, forecast_price=price, forecast_carbon_kg=carb)
            oc = plan.get("expected_cost_usd", 0) or 0
            sv = (bl_cost - oc) / bl_cost * 100 if bl_cost else 0
            savings_pct.append(sv)
        except Exception:
            savings_pct.append(float("nan"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([c / 1000 for c in caps], savings_pct, "s-", color=PAL["optimized"], markersize=8, linewidth=2)
    ax.set(xlabel="Battery Capacity (GWh)", ylabel="Cost Savings (%)",
           title="Fig 14. Savings Sensitivity — Battery Size Sweep")
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig14_savings_sensitivity.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 14: savings sensitivity")


def fig15_regret_perturbation(datasets: List[Dataset]):
    """Fig 15. Regret Under Forecast Perturbation"""
    cfg = _load_yaml("optimization.yaml")
    perts = [0.05, 0.10, 0.15, 0.20, 0.30]
    nt = 30
    fig, ax = plt.subplots(figsize=(9, 5))
    all_data = {}
    for ds in datasets:
        df = ds.test_df.sort_values("timestamp").tail(7 * 24).head(24)
        lb = _clean(df["load_mw"].to_numpy())
        w = _clean(df["wind_mw"].to_numpy()) if "wind_mw" in df.columns else np.zeros(24)
        s = _clean(df["solar_mw"].to_numpy()) if "solar_mw" in df.columns else np.zeros(24)
        rb = w + s
        price = df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in df.columns else None
        carb = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None
        oracle = optimize_dispatch(lb, rb, cfg, forecast_price=price, forecast_carbon_kg=carb)
        oc = oracle.get("expected_cost_usd", 0.0) or 0.0
        rng = np.random.RandomState(42)
        mean_regs = []
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
            mean_regs.append(float(np.mean(regs)) if regs else float("nan"))
        ax.plot([p * 100 for p in perts], mean_regs, marker="o", color=ds.color, label=ds.label)
    ax.set(xlabel="Forecast Perturbation (%)", ylabel="Mean Regret (USD)",
           title="Fig 15. Regret Under Forecast Perturbation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig15_regret_perturbation.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 15: regret perturbation")


def fig16_data_drift(datasets: List[Dataset]):
    """Fig 16. Data Drift Detection Over Time"""
    mcfg = load_monitoring_config()
    thr = float(mcfg.get("data_drift", {}).get("p_value_threshold", 0.01))
    window = 7 * 24
    stride = 7 * 24
    fig, ax = plt.subplots(figsize=(10, 4))
    for ds in datasets:
        fcols = [c for c in ds.train_df.columns if c not in {"timestamp", "load_mw", "wind_mw", "solar_mw"}]
        if not fcols:
            continue
        df = ds.test_df.sort_values("timestamp").reset_index(drop=True)
        weeks = []
        drift_fracs = []
        for start in range(0, len(df) - window + 1, stride):
            current = df.iloc[start:start + window]
            drift = compute_data_drift(ds.train_df, current, fcols, thr)
            nd = sum(1 for c in drift.get("columns", {}).values() if c.get("p_value", 1.0) < thr)
            drift_fracs.append(nd / max(len(fcols), 1) * 100)
            weeks.append(len(weeks))
        ax.plot(weeks, drift_fracs, marker="o", markersize=4, color=ds.color, label=ds.label)
    ax.axhline(50, color="red", linestyle="--", label="50% threshold")
    ax.set(xlabel="Week", ylabel="Features Drifted (%)",
           title="Fig 16. Data Drift (KS Test) Over Time")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig16_data_drift.png", dpi=DPI)
    plt.close(fig)
    print("  [OK] Fig 16: data drift")


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("GridPulse Publication Pack v2 — 2-Country")
    print("=" * 60)

    _ensure(FIG_DIR)
    _ensure(TBL_DIR)

    # Define datasets
    ds_de = Dataset(
        label="Germany (OPSD)", short="DE", color=PAL["de"],
        features_path=REPO / "data" / "processed" / "features.parquet",
        splits_path=REPO / "data" / "processed" / "splits",
        models_path=REPO / "artifacts" / "models",
        reports_path=REPO / "reports",
    )
    ds_us = Dataset(
        label="USA (EIA-930)", short="US", color=PAL["us"],
        features_path=REPO / "data" / "processed" / "us_eia930" / "features.parquet",
        splits_path=REPO / "data" / "processed" / "us_eia930" / "splits",
        models_path=REPO / "artifacts" / "models_eia930",
        reports_path=REPO / "reports" / "eia930",
    )

    datasets = [ds_de, ds_us]

    print("\nLoading datasets...")
    for ds in datasets:
        load_dataset(ds)
        print(f"  {ds.label}: {len(ds.df_full):,} rows | Train: {len(ds.train_df):,} | Test: {len(ds.test_df):,}")

    # ── TABLES ──
    print("\n--- Generating 8 publication tables ---")
    t1 = table1_dataset_summary(datasets)
    t2_de = table2_forecast_metrics(ds_de, 2)
    t3_us = table2_forecast_metrics(ds_us, 3)
    t4 = table4_uncertainty(datasets)
    t5_out, bl, op, cfg, imp = table5_optimization_impact(ds_de)
    t6 = table6_robustness(datasets)
    t7 = table7_baseline_comparison(ds_de)
    t8 = table8_runtime(datasets)

    # ── FIGURES ──
    print("\n--- Generating 16 publication figures ---")
    fig01_geographic_scope(datasets)
    fig02_load_renewable_profiles(datasets)
    fig03_05_forecast_vs_actual(datasets)
    fig06_rolling_backtest(datasets)
    fig07_error_seasonality(datasets)
    fig08_conformal_intervals(datasets)
    fig09_coverage_vs_horizon(datasets)
    fig10_anomaly_timeline(datasets)
    fig11_dispatch_comparison(ds_de)
    fig12_soc_trajectory(ds_de)
    fig13_cost_carbon_tradeoff(ds_de)
    fig14_savings_sensitivity(ds_de)
    fig15_regret_perturbation(datasets)
    fig16_data_drift(datasets)

    # ── CHECKLIST ──
    print("\n" + "=" * 60)
    print("PUBLICATION PACK v2 CHECKLIST")
    print("=" * 60)
    fig_files = [
        "fig01_geographic_scope.png", "fig02_load_renewable_profiles.png",
        "fig03_05_forecast_vs_actual.png", "fig06_rolling_backtest_rmse.png",
        "fig07_error_seasonality.png", "fig08_conformal_intervals.png",
        "fig09_coverage_vs_horizon.png", "fig10_anomaly_timeline.png",
        "fig11_dispatch_comparison.png", "fig12_soc_trajectory.png",
        "fig13_cost_carbon_tradeoff.png", "fig14_savings_sensitivity.png",
        "fig15_regret_perturbation.png", "fig16_data_drift.png",
    ]
    tbl_files = [
        "table1_dataset_summary.csv", "table2_forecast_metrics_de.csv",
        "table3_forecast_metrics_us.csv", "table4_uncertainty.csv",
        "table5_optimization_impact.csv", "table6_robustness.csv",
        "table7_baseline_comparison.csv", "table8_runtime.csv",
    ]
    ok = True
    print("\nFigures:")
    for f in fig_files:
        exists = (FIG_DIR / f).exists()
        ok = ok and exists
        print(f"  {'OK' if exists else 'MISSING'}  {f}")
    print("\nTables:")
    for t in tbl_files:
        exists = (TBL_DIR / t).exists()
        ok = ok and exists
        print(f"  {'OK' if exists else 'MISSING'}  {t}")
    status = "ALL 16 FIGURES + 8 TABLES GENERATED" if ok else "SOME ARTIFACTS MISSING"
    print(f"\n{status}")
    print("=" * 60)


if __name__ == "__main__":
    main()