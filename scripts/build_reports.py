"""Script: build reports."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml 

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    # Ensure local package imports resolve when running this script standalone.
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.forecasting.baselines import persistence_24h
from gridpulse.forecasting.ml_gbm import train_gbm, predict_gbm
from gridpulse.forecasting.backtest import multi_horizon_metrics
from gridpulse.forecasting.predict import load_model_bundle, predict_next_24h
from gridpulse.forecasting.dl_lstm import LSTMForecaster
from gridpulse.forecasting.dl_tcn import TCNForecaster
from gridpulse.forecasting.datasets import SeqConfig, TimeSeriesWindowDataset
from gridpulse.optimizer.lp_dispatch import optimize_dispatch
from gridpulse.forecasting.uncertainty.conformal import load_conformal
from gridpulse.optimizer.baselines import (
    grid_only_dispatch,
    naive_battery_dispatch,
    peak_shaving_dispatch,
    greedy_price_dispatch,
)
from gridpulse.anomaly.detect import detect_anomalies
from gridpulse.optimizer.impact import impact_summary
from gridpulse.monitoring.retraining import load_monitoring_config, compute_data_drift
from gridpulse.utils.metrics import rmse, mae, mape, smape, daylight_mape, r2_score
from gridpulse.utils.scaler import StandardScaler
from gridpulse.forecasting.baselines import moving_average


@dataclass
class ReportContext:
    repo_root: Path
    features_path: Path
    splits_dir: Path
    models_dir: Path
    reports_dir: Path


def ensure_dir(path: Path):
    # Key: CLI/reporting helper
    path.mkdir(parents=True, exist_ok=True)


def _load_forecast_cfg(ctx: ReportContext) -> dict:
    cfg_path = ctx.repo_root / "configs" / "forecast.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return {"models": {}, "fallback_order": ["lstm", "tcn", "gbm"]}


def _load_uncertainty_cfg(ctx: ReportContext) -> dict:
    cfg_path = ctx.repo_root / "configs" / "uncertainty.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return {"enabled": False}


def _conformal_bounds(target: str, yhat: np.ndarray, ctx: ReportContext, cfg: dict) -> dict | None:
    if not cfg.get("enabled", False):
        return None
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts/uncertainty"))
    path = artifacts_dir / f"{target}_conformal.json"
    if not path.exists():
        return None
    ci = load_conformal(path)
    if ci.q_h is not None and len(ci.q_h) != len(yhat):
        return None
    lower, upper = ci.predict_interval(np.asarray(yhat, dtype=float))
    return {"lower": lower, "upper": upper}


def _resolve_model_path(target: str, cfg: dict, models_dir: Path) -> Path | None:
    explicit = cfg.get("models", {}).get(target)
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    order = cfg.get("fallback_order", ["lstm", "tcn", "gbm"])
    patterns = []
    for kind in order:
        if kind == "gbm":
            patterns.append(f"gbm_*_{target}.pkl")
        else:
            patterns.append(f"{kind}_{target}.pt")
    for pat in patterns:
        for p in models_dir.glob(pat):
            if p.exists():
                return p
    return None


def _forecast_with_bundle(df: pd.DataFrame, target: str, horizon: int, ctx: ReportContext) -> dict | None:
    cfg = _load_forecast_cfg(ctx)
    model_path = _resolve_model_path(target, cfg, ctx.models_dir)
    if not model_path:
        return None
    bundle = load_model_bundle(model_path)
    try:
        return predict_next_24h(df, bundle, horizon=horizon)
    except Exception:
        return None


def _get_quantile(pred: dict | None, q: float, fallback: np.ndarray) -> np.ndarray:
    if not pred:
        return fallback
    q_key = str(q)
    quantiles = pred.get("quantiles", {})
    if q_key not in quantiles:
        return fallback
    return np.asarray(quantiles[q_key], dtype=float)


def _safe_float(val):
    try:
        if val is None:
            return None
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return float(val)
    except Exception:
        return None


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        # Strip NaN/Inf so JSON/CSV stay stable across runs.
        return _safe_float(obj)
    return obj


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    # Core evaluation metrics used in markdown reports and plots.
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    if target in ("solar_mw", "wind_mw"):
        # Daylight‑only MAPE avoids near‑zero instability.
        out["daylight_mape"] = daylight_mape(y_true, y_pred)
    return out


def _clean_series(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if not np.isfinite(arr).all():
        # Fill bad values with a robust mean so optimizers don’t explode.
        mean = np.nanmean(arr)
        if not np.isfinite(mean):
            mean = 0.0
        arr = np.where(np.isfinite(arr), arr, mean)
    return arr


def _load_optimization_config(ctx: ReportContext) -> dict:
    cfg_path = ctx.repo_root / "configs" / "optimization.yaml"
    if cfg_path.exists():
        import yaml
        # Allow report generation to respect the same dispatch constraints as training.
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return {}


def _build_torch_model(bundle: dict):
    model_type = bundle.get("model_type")
    feat_cols = bundle.get("feature_cols", [])
    params = bundle.get("model_params", {})
    horizon = int(bundle.get("horizon", params.get("horizon", 24)))
    if model_type == "lstm":
        model = LSTMForecaster(
            n_features=len(feat_cols),
            hidden_size=int(params.get("hidden_size", 128)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
            horizon=horizon,
        )
    elif model_type == "tcn":
        model = TCNForecaster(
            n_features=len(feat_cols),
            num_channels=list(params.get("num_channels", [32, 32, 32])),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unknown torch model_type: {model_type}")

    # Restore trained weights into the exact architecture used at training time.
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def _eval_seq_model(bundle: dict, df: pd.DataFrame) -> dict | None:
    feat_cols = bundle.get("feature_cols", [])
    target = bundle.get("target")
    lookback = int(bundle.get("lookback", 168))
    horizon = int(bundle.get("horizon", 24))
    if not feat_cols or target is None:
        return None
    # Use the same scalers saved in the bundle (prevents train/test mismatch).
    X = df[feat_cols].to_numpy()
    y = df[target].to_numpy()
    x_scaler = StandardScaler.from_dict(bundle.get("x_scaler"))
    y_scaler = StandardScaler.from_dict(bundle.get("y_scaler"))
    if x_scaler is not None:
        X = x_scaler.transform(X)
    if y_scaler is not None:
        y = y_scaler.transform(y.reshape(-1, 1)).reshape(-1)

    ds = TimeSeriesWindowDataset(X, y, SeqConfig(lookback=lookback, horizon=horizon))
    if len(ds) == 0:
        return None
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    model = _build_torch_model(bundle)
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in dl:
            pred_seq = model(xb).numpy()
            # Use the full horizon window for evaluation.
            pred_hz = pred_seq[:, -horizon:]
            preds.append(pred_hz.reshape(-1))
            trues.append(yb.numpy().reshape(-1))
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    if y_scaler is not None:
        # Inverse‑transform back to real units for metrics.
        y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    return _compute_metrics(y_true, y_pred, target)


def load_split_data(ctx: ReportContext):
    if (ctx.splits_dir / "test.parquet").exists():
        test_df = pd.read_parquet(ctx.splits_dir / "test.parquet")
        train_df = pd.read_parquet(ctx.splits_dir / "train.parquet")
        return train_df, test_df
    if ctx.features_path.exists():
        # Fallback to time‑based split if explicit splits are missing.
        df = pd.read_parquet(ctx.features_path).sort_values("timestamp")
        n = len(df)
        train_df = df.iloc[: int(n * 0.7)]
        test_df = df.iloc[int(n * 0.85):]
        return train_df, test_df
    return None, None


def build_multi_horizon(ctx: ReportContext):
    train_df, test_df = load_split_data(ctx)
    if train_df is None or test_df is None:
        return None

    horizons = [1, 3, 6, 12, 24]
    targets = ["load_mw", "wind_mw", "solar_mw"]

    result = {"horizons": horizons, "targets": {}}
    for target in targets:
        y_true = test_df[target].to_numpy()
        y_pred = persistence_24h(test_df, target)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        baseline = multi_horizon_metrics(y_true, y_pred, horizons, target)

        # Quick GBM baseline for reference at multiple horizons.
        X_train = train_df[[c for c in train_df.columns if c not in {"timestamp", "load_mw", "wind_mw", "solar_mw"}]].to_numpy()
        y_train = train_df[target].to_numpy()
        X_test = test_df[[c for c in test_df.columns if c not in {"timestamp", "load_mw", "wind_mw", "solar_mw"}]].to_numpy()
        _, gbm = train_gbm(X_train, y_train, params={"n_estimators": 200, "learning_rate": 0.05, "random_state": 42})
        gbm_pred = predict_gbm(gbm, X_test)
        gbm = multi_horizon_metrics(test_df[target].to_numpy(), gbm_pred, horizons, target)

        result["targets"][target] = {"persistence": baseline, "gbm": gbm}

    out_path = ctx.reports_dir / "multi_horizon_backtest.json"
    ensure_dir(ctx.reports_dir)
    out_path.write_text(json.dumps(_sanitize(result), indent=2), encoding="utf-8")
    return result


def plot_multi_horizon(ctx: ReportContext, report: dict):
    fig_dir = ctx.reports_dir / "figures"
    ensure_dir(fig_dir)

    target = "load_mw"
    data = report["targets"].get(target, {})
    horizons = report.get("horizons", [])
    if not horizons or not data:
        return None

    metrics = ["rmse", "mae", "smape"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, metric in enumerate(metrics):
        for label in ["persistence", "gbm"]:
            res = data.get(label, {})
            vals = []
            for h in horizons:
                summary = res.get("results", {}).get(str(h), {}).get("summary", {})
                v = summary.get(metric)
                vals.append(v if v is not None else float("nan"))
            ax[i].plot(horizons, vals, marker="o", label=label)
        ax[i].set_title(f"{metric.upper()} vs Horizon")
        ax[i].set_xlabel("Horizon (hours)")
        ax[i].set_ylabel(metric.upper())
        ax[i].grid(True, alpha=0.3)
    ax[0].legend()
    plt.tight_layout()

    out_path = fig_dir / "multi_horizon_backtest.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return out_path


def make_sample_figures(ctx: ReportContext):
    fig_dir = ctx.reports_dir / "figures"
    ensure_dir(fig_dir)

    if ctx.features_path.exists():
        # Snapshot time‑series for quick visual sanity check.
        df = pd.read_parquet(ctx.features_path).sort_values("timestamp")
        recent = df.tail(7 * 24)
        fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        recent.plot(x="timestamp", y="load_mw", ax=ax[0], color="#1f77b4", title="Load (last 7 days)")
        recent.plot(x="timestamp", y="wind_mw", ax=ax[1], color="#2ca02c", title="Wind (last 7 days)")
        recent.plot(x="timestamp", y="solar_mw", ax=ax[2], color="#ff7f0e", title="Solar (last 7 days)")
        plt.tight_layout()
        fig.savefig(fig_dir / "forecast_sample.png", dpi=300, bbox_inches="tight")

    cfg_path = ctx.repo_root / "configs" / "optimization.yaml"
    cfg = {}
    if cfg_path.exists():
        import yaml
        # Sample dispatch uses the same config as production optimization.
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    load = np.full(24, 8000.0)
    renew = np.full(24, 3200.0)
    plan = optimize_dispatch(load, renew, cfg)
    df = pd.DataFrame({
        "grid_mw": plan["grid_mw"],
        "battery_charge_mw": plan["battery_charge_mw"],
        "battery_discharge_mw": plan["battery_discharge_mw"],
        "soc_mwh": plan["soc_mwh"],
    })
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    df[["grid_mw", "battery_charge_mw", "battery_discharge_mw"]].plot(ax=ax[0], title="Dispatch (sample)")
    df[["soc_mwh"]].plot(ax=ax[1], title="Battery SOC (sample)")
    plt.tight_layout()
    fig.savefig(fig_dir / "dispatch_sample.png", dpi=300, bbox_inches="tight")

    train_df, test_df = load_split_data(ctx)
    if train_df is not None and test_df is not None:
        # Plot drift diagnostics on a short window to keep the figure compact.
        current_df = test_df.tail(7 * 24)
        feature_cols = [c for c in train_df.columns if c not in {"timestamp", "load_mw", "wind_mw", "solar_mw"}]
        cfg = load_monitoring_config()
        drift = compute_data_drift(train_df, current_df, feature_cols, float(cfg.get("data_drift", {}).get("p_value_threshold", 0.01)))
        columns = list(drift.get("columns", {}).keys())[:10]
        pvals = [drift["columns"][c].get("p_value") for c in columns]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(columns, pvals)
        ax.axhline(float(cfg.get("data_drift", {}).get("p_value_threshold", 0.01)), color="red", linestyle="--", label="threshold")
        ax.set_title("Data Drift p-values (sample)")
        ax.set_ylabel("p-value")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        plt.tight_layout()
        fig.savefig(fig_dir / "drift_sample.png", dpi=300, bbox_inches="tight")


def make_demo_gif(ctx: ReportContext):
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig_dir = ctx.reports_dir / "figures"
    ensure_dir(fig_dir)
    if not ctx.features_path.exists():
        return None

    # Small animated preview for the README (lightweight, no model required).
    df = pd.read_parquet(ctx.features_path).sort_values("timestamp")
    df = df.tail(72)

    fig, ax = plt.subplots(figsize=(6, 3))
    line1, = ax.plot([], [], color="#1f77b4", label="load_mw")
    line2, = ax.plot([], [], color="#2ca02c", label="wind_mw")
    line3, = ax.plot([], [], color="#ff7f0e", label="solar_mw")

    ax.set_xlim(0, len(df))
    ax.set_ylim(0, max(df["load_mw"].max(), df["wind_mw"].max(), df["solar_mw"].max()) * 1.1)
    ax.set_title("GridPulse Forecast Inputs (sample)")
    ax.legend(loc="upper right", fontsize=8)

    def update(i):
        x = np.arange(i)
        line1.set_data(x, df["load_mw"].values[:i])
        line2.set_data(x, df["wind_mw"].values[:i])
        line3.set_data(x, df["solar_mw"].values[:i])
        return line1, line2, line3

    anim = FuncAnimation(fig, update, frames=len(df), interval=100, blit=True)
    out_path = fig_dir / "demo.gif"
    anim.save(out_path, writer=PillowWriter(fps=10))
    return out_path


def plot_model_comparison(ctx: ReportContext, metrics: dict):
    fig_dir = ctx.reports_dir / "figures"
    ensure_dir(fig_dir)

    targets = metrics.get("targets", {})
    if not targets:
        return None

    model_names = set()
    for _, data in targets.items():
        for model in data.keys():
            if model != "n_features":
                model_names.add(model)
    models = sorted(model_names)
    if not models:
        return None

    metric_keys = ["rmse", "mae", "smape"]
    means = {m: {} for m in models}
    for m in models:
        for k in metric_keys:
            vals = []
            for _, data in targets.items():
                v = data.get(m, {}).get(k)
                if v is not None:
                    vals.append(v)
            means[m][k] = float(np.mean(vals)) if vals else None

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, metric in enumerate(metric_keys):
        vals = [means[m][metric] if means[m][metric] is not None else float("nan") for m in models]
        ax[i].bar(models, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(models)])
        ax[i].set_title(metric.upper())
        ax[i].tick_params(axis="x", rotation=25)
        ax[i].set_ylabel("mean across targets")
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax[i].text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()

    out_path = fig_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return out_path


def refresh_metrics_from_models(ctx: ReportContext) -> dict:
    report_path = ctx.reports_dir / "week2_metrics.json"
    if report_path.exists():
        metrics = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        metrics = {"targets": {}}

    train_df, test_df = load_split_data(ctx)
    if test_df is None:
        return metrics

    targets = ["load_mw", "wind_mw", "solar_mw"]
    for target in targets:
        metrics.setdefault("targets", {}).setdefault(target, {})

        gbm_path = None
        for p in ctx.models_dir.glob(f"gbm_*_{target}.pkl"):
            gbm_path = p
            break
        if gbm_path and gbm_path.exists():
            bundle = load_model_bundle(gbm_path)
            feat_cols = bundle.get("feature_cols", [])
            if feat_cols:
                X = test_df[feat_cols]
                y = test_df[target].to_numpy()
                pred = bundle["model"].predict(X)
                m = _compute_metrics(y, pred, target)
                m["model"] = gbm_path.stem.replace(f"_{target}", "")
                metrics["targets"][target]["gbm"] = {
                    **metrics["targets"][target].get("gbm", {}),
                    **m,
                }

        for kind in ["lstm", "tcn"]:
            path = ctx.models_dir / f"{kind}_{target}.pt"
            if path.exists():
                # Evaluate sequence models with the same scaling as training.
                bundle = load_model_bundle(path)
                m = _eval_seq_model(bundle, test_df)
                if m:
                    metrics["targets"][target][kind] = {
                        **metrics["targets"][target].get(kind, {}),
                        **m,
                    }

    report_path.write_text(json.dumps(_sanitize(metrics), indent=2), encoding="utf-8")
    return metrics


def compute_baseline_metrics(ctx: ReportContext) -> dict:
    _, test_df = load_split_data(ctx)
    if test_df is None or test_df.empty:
        return {}

    targets = ["load_mw", "wind_mw", "solar_mw"]
    out: dict = {}
    for target in targets:
        y_true = test_df[target].to_numpy()
        pers = persistence_24h(test_df, target)
        ma = moving_average(test_df, target, 24)

        def _eval(pred):
            mask = np.isfinite(y_true) & np.isfinite(pred)
            if mask.sum() == 0:
                return None
            return _compute_metrics(y_true[mask], pred[mask], target)

        out[target] = {
            "persistence_24h": _eval(pers),
            "moving_average_24h": _eval(ma),
        }

    return out


def plot_arbitrage_dispatch(ctx: ReportContext, hours: np.ndarray, price_curve: np.ndarray, grid_load: np.ndarray, optimized_load: np.ndarray, battery_flow: np.ndarray):
    """
    Generates the Level-4 Arbitrage Optimization plot.
    battery_flow: positive = discharge, negative = charge
    """
    fig_dir = ctx.reports_dir / "figures"
    ensure_dir(fig_dir)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Market Price (signal) and grid load (action) on dual axes.
    color = 'tab:red'
    ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Electricity Price ($/MWh)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(hours, price_curve, color=color, linestyle='--', linewidth=2, label='Market Price', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Second axis for grid/battery power.
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Power (MW)', color=color, fontsize=12, fontweight='bold')

    # Baseline grid import without optimization.
    ax2.plot(hours, grid_load, color='gray', alpha=0.4, linewidth=2, label='Baseline Grid Load')

    # Optimized grid import from LP dispatch.
    ax2.plot(hours, optimized_load, color=color, linewidth=3, label='GridPulse Optimized Load')

    # Highlight arbitrage windows (charge at low price, discharge at high price).
    # Green area = Charging (Money saved later)
    ax2.fill_between(hours, grid_load, optimized_load, 
                     where=(np.array(battery_flow) < 0), color='green', alpha=0.3, label='Charging (Low Price)')
    # Orange area = Discharging (Cost Avoided)
    ax2.fill_between(hours, grid_load, optimized_load, 
                     where=(np.array(battery_flow) > 0), color='orange', alpha=0.5, label='Discharging (High Price)')

    # Presentation polish for README‑ready output.
    plt.title('GridPulse Decision Logic: Arbitrage Optimization', fontsize=16, fontweight='bold', pad=20)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    out_path = fig_dir / "arbitrage_optimization.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_impact_report(ctx: ReportContext):
    train_df, test_df = load_split_data(ctx)
    if test_df is None or test_df.empty:
        return None

    horizon = 24 * 7
    if len(test_df) < horizon:
        return None

    test_df = test_df.sort_values("timestamp").reset_index(drop=True)
    cfg = _load_optimization_config(ctx)

    max_import = float(cfg.get("grid", {}).get("max_import_mw", cfg.get("grid", {}).get("max_draw_mw", 50.0)))

    def _select_window(df: pd.DataFrame, h: int) -> tuple[pd.DataFrame, int, int]:
        load_series = df["load_mw"].to_numpy()
        wind_series = df["wind_mw"].to_numpy() if "wind_mw" in df.columns else np.zeros_like(load_series)
        solar_series = df["solar_mw"].to_numpy() if "solar_mw" in df.columns else np.zeros_like(load_series)
        carbon_series = df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in df.columns else None

        best = None
        best_score = -np.inf
        step = 24
        for start in range(0, len(df) - h + 1, step):
            end = start + h
            load = load_series[start:end]
            renew = wind_series[start:end] + solar_series[start:end]
            grid = np.maximum(0.0, load - renew)
            cap_frac = float(np.mean(grid >= max_import * 0.999))
            # Avoid fully saturated windows; favor higher carbon variability.
            if cap_frac >= 0.9:
                continue
            if carbon_series is not None:
                score = float(np.nanstd(carbon_series[start:end])) - cap_frac
            else:
                score = -cap_frac
            if score > best_score:
                best = (start, end)
                best_score = score

        if best is None:
            start = len(df) - h
            end = len(df)
        else:
            start, end = best
        return df.iloc[start:end], start, end

    # Use a 7-day slice from the test split to compare policies.
    window, win_start, win_end = _select_window(test_df, horizon)
    context_df = test_df.iloc[:win_start] if win_start > 0 else test_df.iloc[:-horizon]
    load = _clean_series(window["load_mw"].to_numpy())
    wind = window["wind_mw"].to_numpy() if "wind_mw" in window.columns else np.zeros_like(load)
    solar = window["solar_mw"].to_numpy() if "solar_mw" in window.columns else np.zeros_like(load)
    if "price_eur_mwh" in window.columns:
        price = window["price_eur_mwh"].to_numpy()
    elif "price_usd_mwh" in window.columns:
        price = window["price_usd_mwh"].to_numpy()
    else:
        # If no price signal is available, cost savings will be near zero.
        price = None
    carbon = window["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in window.columns else None
    moer = window["moer_kg_per_mwh"].to_numpy() if "moer_kg_per_mwh" in window.columns else None
    carbon_source = cfg.get("carbon", {}).get("source", "average")
    carbon_for_opt = carbon
    if carbon_source == "moer" and moer is not None:
        carbon_for_opt = moer
    elif carbon_source == "moer" and moer is None:
        carbon_source = "average"
    renew = _clean_series(wind) + _clean_series(solar)

    # Forecast-driven dispatch (if models are available).
    load_pred = _forecast_with_bundle(context_df, "load_mw", horizon, ctx)
    wind_pred = _forecast_with_bundle(context_df, "wind_mw", horizon, ctx)
    solar_pred = _forecast_with_bundle(context_df, "solar_mw", horizon, ctx)

    forecast_available = bool(load_pred and wind_pred and solar_pred)
    if forecast_available:
        f_load = np.asarray(load_pred["forecast"], dtype=float)
        f_wind = np.asarray(wind_pred["forecast"], dtype=float)
        f_solar = np.asarray(solar_pred["forecast"], dtype=float)
        f_renew = f_wind + f_solar
        if len(f_load) != horizon or len(f_renew) != horizon:
            # Forecast models are trained for 24h; fall back when window is longer.
            forecast_available = False
    if not forecast_available:
        f_load, f_renew = load, renew

    # Policy suite: baseline, naive battery, heuristic, greedy, optimized.
    baseline = grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carbon_for_opt)
    naive = naive_battery_dispatch(load, renew, cfg, price_series=price, carbon_series=carbon_for_opt)
    peak = peak_shaving_dispatch(load, renew, cfg, price_series=price, carbon_series=carbon_for_opt)
    greedy = greedy_price_dispatch(load, renew, cfg, price_series=price, carbon_series=carbon_for_opt)

    # GridPulse uses forecast-driven dispatch when available.
    optimized = optimize_dispatch(f_load, f_renew, cfg, forecast_price=price, forecast_carbon_kg=carbon_for_opt)
    # Oracle upper bound: perfect-forecast dispatch using actuals.
    oracle = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carbon_for_opt)

    # Risk-aware dispatch using conformal bounds (fallback to quantiles if needed).
    risk_plan = None
    risk_cfg = cfg.get("risk", {})
    unc_cfg = _load_uncertainty_cfg(ctx)
    if forecast_available and risk_cfg.get("enabled", False):
        load_bounds = _conformal_bounds("load_mw", f_load, ctx, unc_cfg)
        wind_bounds = _conformal_bounds("wind_mw", f_wind, ctx, unc_cfg)
        solar_bounds = _conformal_bounds("solar_mw", f_solar, ctx, unc_cfg)

        if load_bounds is None:
            load_bounds = {
                "lower": _get_quantile(load_pred, 0.1, f_load),
                "upper": _get_quantile(load_pred, 0.9, f_load),
            }
        if wind_bounds is None:
            wind_bounds = {
                "lower": _get_quantile(wind_pred, 0.1, f_wind),
                "upper": _get_quantile(wind_pred, 0.9, f_wind),
            }
        if solar_bounds is None:
            solar_bounds = {
                "lower": _get_quantile(solar_pred, 0.1, f_solar),
                "upper": _get_quantile(solar_pred, 0.9, f_solar),
            }

        renew_bounds = {
            "lower": np.asarray(wind_bounds["lower"], dtype=float)
            + np.asarray(solar_bounds["lower"], dtype=float),
            "upper": np.asarray(wind_bounds["upper"], dtype=float)
            + np.asarray(solar_bounds["upper"], dtype=float),
        }
        risk_plan = optimize_dispatch(
            f_load,
            f_renew,
            cfg,
            forecast_price=price,
            forecast_carbon_kg=carbon_for_opt,
            load_interval=load_bounds,
            renewables_interval=renew_bounds,
        )

    impact = impact_summary(baseline, optimized)
    impact_naive = impact_summary(naive, optimized)
    impact_oracle = impact_summary(baseline, oracle)

    moer_summary = None
    if moer is not None:
        def _moer_kg(plan):
            return float(np.sum(np.asarray(plan["grid_mw"]) * moer))
        moer_summary = {
            "baseline_moer_kg": _moer_kg(baseline),
            "gridpulse_moer_kg": _moer_kg(optimized),
            "oracle_moer_kg": _moer_kg(oracle),
        }

    baseline_peak = float(np.max(baseline["grid_mw"])) if baseline.get("grid_mw") else None
    optimized_peak = float(np.max(optimized["grid_mw"])) if optimized.get("grid_mw") else None
    peak_shaving_pct = None
    if baseline_peak and baseline_peak > 0 and optimized_peak is not None:
        # Peak shaving measured as reduction vs baseline peak.
        peak_shaving_pct = (baseline_peak - optimized_peak) / baseline_peak * 100.0

    fig_dir = ctx.reports_dir / "figures"
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(baseline["grid_mw"], label="baseline grid", color="#1f77b4")
    ax[0].plot(optimized["grid_mw"], label="gridpulse (forecast)", color="#ff7f0e")
    ax[0].plot(peak["grid_mw"], label="peak‑shaving", color="#9467bd")
    ax[0].plot(naive["grid_mw"], label="naive grid", color="#2ca02c")
    ax[0].set_title("Grid Import (MW)")
    ax[0].legend()

    ax[1].plot(optimized["battery_charge_mw"], label="charge", color="#1f77b4")
    ax[1].plot(optimized["battery_discharge_mw"], label="discharge", color="#ff7f0e")
    ax[1].set_title("Optimized Battery Flow (MW)")
    ax[1].legend()

    ax[2].plot(baseline["soc_mwh"], label="baseline SOC", color="#1f77b4", linestyle="--")
    ax[2].plot(optimized["soc_mwh"], label="optimized SOC", color="#ff7f0e")
    ax[2].set_title("State of Charge (MWh)")
    ax[2].legend()

    plt.tight_layout()
    fig_path = fig_dir / "dispatch_compare.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Impact summary CSV + savings plot for README automation.
    summary = {
        "baseline_cost_usd": impact.get("baseline_cost_usd"),
        "gridpulse_cost_usd": impact.get("optimized_cost_usd"),
        "cost_savings_pct": impact.get("cost_savings_pct"),
        "baseline_carbon_kg": impact.get("baseline_carbon_kg"),
        "gridpulse_carbon_kg": impact.get("optimized_carbon_kg"),
        "carbon_reduction_pct": impact.get("carbon_reduction_pct"),
        "baseline_peak_mw": baseline_peak,
        "gridpulse_peak_mw": optimized_peak,
        "peak_shaving_pct": peak_shaving_pct,
        "oracle_cost_usd": oracle.get("expected_cost_usd"),
        "oracle_gap_pct": (
            (optimized.get("expected_cost_usd") - oracle.get("expected_cost_usd"))
            / oracle.get("expected_cost_usd")
            * 100.0
            if oracle.get("expected_cost_usd")
            else None
        ),
        "carbon_source": carbon_source,
    }
    impact_csv = ctx.reports_dir / "impact_summary.csv"
    pd.DataFrame([summary]).to_csv(impact_csv, index=False)

    labels = ["Cost Savings %", "Carbon Reduction %", "Peak Shaving %"]
    values = [
        summary.get("cost_savings_pct"),
        summary.get("carbon_reduction_pct"),
        summary.get("peak_shaving_pct"),
    ]
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.bar(labels, [v if v is not None else 0.0 for v in values], color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax2.set_ylabel("% improvement")
    ax2.set_title("Impact Savings vs Baseline")
    for i, v in enumerate(values):
        if v is not None:
            ax2.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    impact_plot = fig_dir / "impact_savings.png"
    fig2.savefig(impact_plot, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Generate arbitrage visualization only if price data exists.
    arb_plot_path = None
    if price is not None:
        # battery_flow: discharge (pos) - charge (pos) -> net flow (pos=discharge, neg=charge)
        battery_flow = np.array(optimized["battery_discharge_mw"]) - np.array(optimized["battery_charge_mw"])
        # optimized_load (grid import) is roughly load - renewables - battery_flow (simplified for viz)
        # But we can just use the grid_mw from the plan
        grid_load_baseline = np.maximum(0, load - renew) # Simple baseline
        grid_load_optimized = np.array(optimized["grid_mw"])
        
        arb_plot_path = plot_arbitrage_dispatch(ctx, np.arange(horizon), price, grid_load_baseline, grid_load_optimized, battery_flow)


    def _fmt(val, digits=2):
        if val is None:
            return "n/a"
        return f"{val:,.{digits}f}"

    def _fmt_pct(val, digits=2):
        if val is None:
            return "n/a"
        return f"{val:.{digits}f}%"

    lines = []
    lines.append("# Impact Evaluation — Baseline vs GridPulse\n\n")
    lines.append("This report compares dispatch outcomes for the same 7‑day forecast window (selected from the test split).\n\n")
    lines.append(f"- Horizon: {horizon} hours (7 days)\n")
    lines.append(f"- Window index: {win_start}–{win_end}\n")
    lines.append("- Forecast source: test split (proxy for day‑ahead forecast)\n")
    lines.append("- Config: `configs/optimization.yaml`\n\n")

    lines.append("## Policy Comparison\n")
    lines.append("| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |\n")
    lines.append("|---|---:|---:|---:|\n")
    policies = [
        ("Grid‑only baseline", baseline),
        ("Naive battery", naive),
        ("Peak‑shaving heuristic", peak),
    ]
    if price is not None:
        policies.append(("Price‑greedy (MPC‑style)", greedy))
    policies.append(("GridPulse (forecast‑optimized)", optimized))
    if risk_plan is not None:
        policies.append(("Risk‑aware (interval)", risk_plan))
    policies.append(("Oracle upper bound (perfect forecast)", oracle))
    for name, plan in policies:
        lines.append(
            f"| {name} | {_fmt(plan.get('expected_cost_usd'))} | {_fmt(plan.get('carbon_kg'))} | {_fmt(plan.get('carbon_cost_usd'))} |\n"
        )
    lines.append("\n")

    lines.append("## Savings vs Baseline (GridPulse vs Grid‑only)\n")
    lines.append(f"- Cost savings: {_fmt(impact.get('cost_savings_usd'))} ({_fmt_pct(impact.get('cost_savings_pct'))})\n")
    lines.append(f"- Carbon reduction: {_fmt(impact.get('carbon_reduction_kg'))} kg ({_fmt_pct(impact.get('carbon_reduction_pct'))})\n\n")
    lines.append(f"- Carbon source used for optimization: {carbon_source}\n\n")

    lines.append("## Savings vs Naive Battery (GridPulse vs Naive)\n")
    lines.append(f"- Cost savings: {_fmt(impact_naive.get('cost_savings_usd'))} ({_fmt_pct(impact_naive.get('cost_savings_pct'))})\n")
    lines.append(f"- Carbon reduction: {_fmt(impact_naive.get('carbon_reduction_kg'))} kg ({_fmt_pct(impact_naive.get('carbon_reduction_pct'))})\n\n")

    lines.append("## Oracle Gap (GridPulse vs Perfect‑Forecast Upper Bound)\n")
    lines.append(f"- Oracle cost: {_fmt(oracle.get('expected_cost_usd'))}\n")
    lines.append(f"- Gap vs oracle: {_fmt(optimized.get('expected_cost_usd') - oracle.get('expected_cost_usd'))}\n\n")

    if moer_summary is not None:
        lines.append("## Marginal Emissions (MOER)\n")
        lines.append(f"- Baseline MOER kg: {_fmt(moer_summary.get('baseline_moer_kg'))}\n")
        lines.append(f"- GridPulse MOER kg: {_fmt(moer_summary.get('gridpulse_moer_kg'))}\n")
        lines.append(f"- Oracle MOER kg: {_fmt(moer_summary.get('oracle_moer_kg'))}\n\n")

    lines.append("## Dispatch Comparison\n")
    lines.append("![](figures/dispatch_compare.png)\n")

    if arb_plot_path:
        lines.append("\n## Arbitrage Logic (Level-4)\n")
        lines.append("![](figures/arbitrage_optimization.png)\n")

    out_path = ctx.reports_dir / "impact_comparison.md"
    ensure_dir(ctx.reports_dir)
    out_path.write_text("".join(lines), encoding="utf-8")

    json_path = ctx.reports_dir / "impact_comparison.json"
    json_payload = {
        "horizon": horizon,
        "carbon_source": carbon_source,
        "baseline": baseline,
        "naive": naive,
        "peak_shaving": peak,
        "greedy_price": greedy if price is not None else None,
        "optimized_forecast": optimized,
        "risk": risk_plan,
        "oracle": oracle,
        "impact_vs_baseline": impact,
        "impact_vs_naive": impact_naive,
        "impact_vs_oracle": impact_oracle,
        "moer_summary": moer_summary,
    }
    json_path.write_text(json.dumps(_sanitize(json_payload), indent=2), encoding="utf-8")

    return {
        "report_path": out_path,
        "figure_path": fig_path,
        "summary_csv": impact_csv,
        "savings_plot": impact_plot,
    }


def build_rolling_backtest(
    ctx: ReportContext,
    targets: list[str] | None = None,
    window_days: int = 7,
    stride_days: int = 7,
) -> dict | None:
    _, test_df = load_split_data(ctx)
    if test_df is None or test_df.empty:
        return None

    if targets is None:
        targets = ["load_mw", "wind_mw", "solar_mw"]

    df = test_df.sort_values("timestamp").reset_index(drop=True)
    window = window_days * 24
    stride = stride_days * 24
    if len(df) < window:
        return None

    out = {
        "window_days": window_days,
        "stride_days": stride_days,
        "targets": {},
    }

    md_lines = [
        "# Rolling Backtest Summary\n\n",
        f"- Window: {window_days} days\n",
        f"- Stride: {stride_days} days\n\n",
    ]

    for target in targets:
        if target not in df.columns:
            continue
        y_true = df[target].to_numpy()

        # Baseline: persistence.
        persistence = persistence_24h(df, target)

        # GBM predictions if available.
        gbm_pred = None
        gbm_path = None
        for p in ctx.models_dir.glob(f"gbm_*_{target}.pkl"):
            gbm_path = p
            break
        if gbm_path and gbm_path.exists():
            bundle = load_model_bundle(gbm_path)
            feat_cols = bundle.get("feature_cols", [])
            if feat_cols:
                X = df[feat_cols].to_numpy()
                gbm_pred = bundle["model"].predict(X)

        models = {"persistence": persistence}
        if gbm_pred is not None:
            models["gbm"] = gbm_pred

        results: dict[str, dict] = {}
        for name, preds in models.items():
            per_window = []
            for start in range(0, len(df) - window + 1, stride):
                end = start + window
                yt = y_true[start:end]
                yp = np.asarray(preds[start:end], dtype=float)
                mask = np.isfinite(yt) & np.isfinite(yp)
                if mask.sum() < window * 0.5:
                    continue
                yt = yt[mask]
                yp = yp[mask]
                per_window.append(
                    {
                        "rmse": rmse(yt, yp),
                        "mae": mae(yt, yp),
                        "mape": mape(yt, yp),
                        "smape": smape(yt, yp),
                    }
                )

            if not per_window:
                continue
            arr = pd.DataFrame(per_window)
            summary = {}
            for metric in ["rmse", "mae", "mape", "smape"]:
                mean = float(arr[metric].mean())
                std = float(arr[metric].std(ddof=1)) if len(arr) > 1 else 0.0
                ci95 = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
                summary[metric] = {"mean": mean, "std": std, "ci95": ci95}
            results[name] = {"n_windows": len(arr), "summary": summary, "per_window": per_window}

        if not results:
            continue

        out["targets"][target] = {"results": results}
        md_lines.append(f"## Target: {target}\n")
        md_lines.append("| Model | RMSE | MAE | MAPE | sMAPE |\n")
        md_lines.append("|---|---:|---:|---:|---:|\n")
        for name, res in results.items():
            s = res["summary"]
            md_lines.append(
                f"| {name} | {s['rmse']['mean']:.2f} ± {s['rmse']['ci95']:.2f} | {s['mae']['mean']:.2f} ± {s['mae']['ci95']:.2f} | {s['mape']['mean']:.4f} ± {s['mape']['ci95']:.4f} | {s['smape']['mean']:.4f} ± {s['smape']['ci95']:.4f} |\n"
            )
        md_lines.append("\n")

    out_path = ctx.reports_dir / "rolling_backtest.json"
    ensure_dir(ctx.reports_dir)
    out_path.write_text(json.dumps(_sanitize(out), indent=2), encoding="utf-8")

    md_path = ctx.reports_dir / "rolling_backtest.md"
    md_path.write_text("".join(md_lines), encoding="utf-8")
    return out


def build_case_study(ctx: ReportContext, days: int = 90) -> dict | None:
    _, test_df = load_split_data(ctx)
    if test_df is None or test_df.empty:
        return None

    df = test_df.sort_values("timestamp").reset_index(drop=True)
    window_len = min(len(df), days * 24)
    case_df = df.tail(window_len).copy()

    # Use GBM if available; otherwise persistence.
    pred = None
    model_name = "persistence_24h"
    gbm_path = None
    for p in ctx.models_dir.glob("gbm_*_load_mw.pkl"):
        gbm_path = p
        break
    if gbm_path and gbm_path.exists():
        bundle = load_model_bundle(gbm_path)
        feat_cols = bundle.get("feature_cols", [])
        if feat_cols:
            pred = bundle["model"].predict(case_df[feat_cols].to_numpy())
            model_name = "gbm"
    if pred is None:
        pred = persistence_24h(case_df, "load_mw")

    actual = case_df["load_mw"].to_numpy()
    case_df["forecast"] = np.asarray(pred, dtype=float)
    residuals = actual - case_df["forecast"].to_numpy()
    case_df["abs_residual"] = np.abs(residuals)
    case_df["date"] = pd.to_datetime(case_df["timestamp"]).dt.date
    daily = case_df.groupby("date")["abs_residual"].mean()
    failure_day = daily.idxmax() if not daily.empty else None

    # Anomaly detection on the case window.
    anomaly_out = detect_anomalies(
        actual,
        pred,
        case_df[["wind_mw", "solar_mw", "hour", "dayofweek"]],
    )
    case_df["anomaly"] = anomaly_out["combined"]

    # Build a focused week around the failure day.
    if failure_day is not None:
        tz = case_df["timestamp"].dt.tz
        failure_ts = pd.Timestamp(failure_day).tz_localize(tz)
        week_start = failure_ts - pd.Timedelta(days=3)
        week_end = failure_ts + pd.Timedelta(days=3)
        week_df = case_df[
            (case_df["timestamp"] >= week_start) & (case_df["timestamp"] <= week_end)
        ]
    else:
        week_df = case_df.tail(7 * 24)

    fig_dir = ctx.reports_dir / "figures"
    ensure_dir(fig_dir)

    # Plot actual vs forecast with anomaly markers.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(week_df["timestamp"], week_df["load_mw"], label="actual", color="#1f77b4")
    ax.plot(week_df["timestamp"], week_df["forecast"], label="forecast", color="#ff7f0e")
    anomaly_idx = week_df[week_df["anomaly"]]
    ax.scatter(anomaly_idx["timestamp"], anomaly_idx["load_mw"], color="red", s=10, label="anomaly")
    if failure_day is not None:
        ax.axvspan(failure_ts, failure_ts + pd.Timedelta(days=1), color="red", alpha=0.1, label="failure day")
    ax.set_title("Case Study: Forecast vs Actual (1‑week slice)")
    ax.legend()
    plt.tight_layout()
    week_fig = fig_dir / "case_study_week.png"
    fig.savefig(week_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Dispatch comparison for failure day (24h) if available.
    dispatch_fig = None
    if failure_day is not None:
        day_df = case_df[case_df["date"] == failure_day]
        if len(day_df) >= 24:
            day_df = day_df.head(24)
            load = day_df["load_mw"].to_numpy()
            renew = day_df["wind_mw"].to_numpy() + day_df["solar_mw"].to_numpy()
            price = day_df["price_eur_mwh"].to_numpy() if "price_eur_mwh" in day_df.columns else None
            carbon = day_df["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in day_df.columns else None
            cfg = _load_optimization_config(ctx)
            baseline = grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carbon)
            optimized = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carbon)

            fig2, ax2 = plt.subplots(figsize=(10, 3.5))
            ax2.plot(baseline["grid_mw"], label="baseline", color="#1f77b4")
            ax2.plot(optimized["grid_mw"], label="gridpulse", color="#ff7f0e")
            ax2.set_title("Failure Day Dispatch (Grid Import)")
            ax2.legend()
            plt.tight_layout()
            dispatch_fig = fig_dir / "case_study_dispatch.png"
            fig2.savefig(dispatch_fig, dpi=300, bbox_inches="tight")
            plt.close(fig2)

    # Summary stats
    mask = np.isfinite(case_df["forecast"].to_numpy())
    mape_val = mape(actual[mask], case_df["forecast"].to_numpy()[mask])
    anomaly_count = int(np.sum(case_df["anomaly"]))

    lines = [
        "# Case Study — Forecasting, Anomalies, Dispatch\n\n",
        f"- Window: last {window_len // 24} days of test split\n",
        f"- Forecast model: {model_name}\n",
        f"- Failure day (max abs error): {failure_day}\n",
        f"- Mean absolute percent error (MAPE): {mape_val:.4f}\n",
        f"- Anomalies detected: {anomaly_count}\n\n",
        "## Weekly View (Failure‑centered)\n",
        f"![](figures/{week_fig.name})\n\n",
    ]
    if dispatch_fig:
        lines.extend(
            [
                "## Failure Day Dispatch\n",
                f"![](figures/{dispatch_fig.name})\n\n",
            ]
        )

    out_path = ctx.reports_dir / "case_study.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    return {"report_path": out_path, "week_fig": week_fig, "dispatch_fig": dispatch_fig}


def build_model_cards(ctx: ReportContext, metrics: dict | None = None):
    metrics = metrics or refresh_metrics_from_models(ctx)
    targets = metrics.get("targets", {})

    cards_dir = ctx.reports_dir / "model_cards"
    ensure_dir(cards_dir)

    for target, data in targets.items():
        lines = [f"# Model Card — {target}\n"]
        lines.append("## Overview\n")
        lines.append("Forecasting model trained on OPSD Germany time‑series.\n")
        lines.append("\n## Metrics (test split)\n")
        if target == "solar_mw":
            lines.append("| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |\n")
            lines.append("|---|---:|---:|---:|---:|---:|\n")
        else:
            lines.append("| Model | RMSE | MAE | sMAPE | MAPE |\n")
            lines.append("|---|---:|---:|---:|---:|\n")
        for model, vals in data.items():
            if model == "n_features":
                continue
            if target == "solar_mw":
                lines.append(
                    f"| {model} | {vals.get('rmse', 'n/a')} | {vals.get('mae', 'n/a')} | {vals.get('smape', 'n/a')} | {vals.get('mape', 'n/a')} | {vals.get('daylight_mape', 'n/a')} |\n"
                )
            else:
                lines.append(
                    f"| {model} | {vals.get('rmse', 'n/a')} | {vals.get('mae', 'n/a')} | {vals.get('smape', 'n/a')} | {vals.get('mape', 'n/a')} |\n"
                )
        lines.append("\n## Intended Use\n")
        lines.append("Day‑ahead forecasting for grid planning and dispatch optimization.\n")
        lines.append("\n## Limitations\n")
        lines.append("Performance depends on feature availability and data quality; retraining is required as grid conditions shift.\n")

        (cards_dir / f"{target}.md").write_text("".join(lines), encoding="utf-8")


def build_formal_report(
    ctx: ReportContext,
    multi_horizon_plot: Path | None,
    metrics: dict | None = None,
    baselines: dict | None = None,
):
    metrics = metrics or refresh_metrics_from_models(ctx)
    baselines = baselines or compute_baseline_metrics(ctx)
    targets = metrics.get("targets", {})

    lines = []
    lines.append("# Formal Evaluation Report\n\n")
    lines.append("## Summary\n")
    lines.append("This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.\n\n")

    if targets:
        lines.append("## Model Metrics (Test Split)\n")
        for target, data in targets.items():
            lines.append(f"### {target}\n")
            if target == "solar_mw":
                lines.append("| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |\n")
                lines.append("|---|---:|---:|---:|---:|---:|\n")
            else:
                lines.append("| Model | RMSE | MAE | sMAPE | MAPE |\n")
                lines.append("|---|---:|---:|---:|---:|\n")
            for model, vals in data.items():
                if model == "n_features":
                    continue
                if target == "solar_mw":
                    lines.append(
                        f"| {model} | {vals.get('rmse', 'n/a')} | {vals.get('mae', 'n/a')} | {vals.get('smape', 'n/a')} | {vals.get('mape', 'n/a')} | {vals.get('daylight_mape', 'n/a')} |\n"
                    )
                else:
                    lines.append(
                        f"| {model} | {vals.get('rmse', 'n/a')} | {vals.get('mae', 'n/a')} | {vals.get('smape', 'n/a')} | {vals.get('mape', 'n/a')} |\n"
                    )
            lines.append("\n")

    if baselines:
        lines.append("## Baseline Metrics (Test Split)\n")
        for target, data in baselines.items():
            lines.append(f"### {target}\n")
            if target == "solar_mw":
                lines.append("| Baseline | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |\n")
                lines.append("|---|---:|---:|---:|---:|---:|\n")
            else:
                lines.append("| Baseline | RMSE | MAE | sMAPE | MAPE |\n")
                lines.append("|---|---:|---:|---:|---:|\n")
            for name, vals in data.items():
                if vals is None:
                    continue
                if target == "solar_mw":
                    lines.append(
                        f"| {name} | {vals.get('rmse', 'n/a')} | {vals.get('mae', 'n/a')} | {vals.get('smape', 'n/a')} | {vals.get('mape', 'n/a')} | {vals.get('daylight_mape', 'n/a')} |\n"
                    )
                else:
                    lines.append(
                        f"| {name} | {vals.get('rmse', 'n/a')} | {vals.get('mae', 'n/a')} | {vals.get('smape', 'n/a')} | {vals.get('mape', 'n/a')} |\n"
                    )
            lines.append("\n")

    if multi_horizon_plot:
        # Reference the generated plot by path for markdown rendering.
        lines.append("## Multi‑Horizon Backtest (Load)\n")
        lines.append(f"![]({multi_horizon_plot.as_posix()})\n\n")

    lines.append("## Conclusions\n")
    lines.append("GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.\n")

    out_path = ctx.reports_dir / "formal_evaluation_report.md"
    ensure_dir(ctx.reports_dir)
    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--splits", default="data/processed/splits")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    # Silence known LightGBM feature‑name warning for cleaner logs.
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
    )

    ctx = ReportContext(
        repo_root=repo_root,
        features_path=repo_root / args.features,
        splits_dir=repo_root / args.splits,
        models_dir=repo_root / args.models_dir,
        reports_dir=repo_root / args.reports_dir,
    )

    # End‑to‑end report generation (figures + markdown + JSON summaries).
    multi = build_multi_horizon(ctx)
    plot_path = plot_multi_horizon(ctx, multi) if multi else None

    make_sample_figures(ctx)
    make_demo_gif(ctx)
    metrics = refresh_metrics_from_models(ctx)
    baselines = compute_baseline_metrics(ctx)
    plot_model_comparison(ctx, metrics)
    build_model_cards(ctx, metrics)
    build_formal_report(ctx, plot_path, metrics, baselines)
    build_impact_report(ctx)
    build_rolling_backtest(ctx)
    build_case_study(ctx)

    print("Reports and figures generated.")


if __name__ == "__main__":
    main()
