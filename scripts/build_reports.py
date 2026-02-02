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

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.forecasting.baselines import persistence_24h
from gridpulse.forecasting.ml_gbm import train_gbm, predict_gbm
from gridpulse.forecasting.backtest import multi_horizon_metrics
from gridpulse.forecasting.predict import load_model_bundle
from gridpulse.forecasting.dl_lstm import LSTMForecaster
from gridpulse.forecasting.dl_tcn import TCNForecaster
from gridpulse.forecasting.datasets import SeqConfig, TimeSeriesWindowDataset
from gridpulse.optimizer.lp_dispatch import optimize_dispatch
from gridpulse.optimizer.baselines import grid_only_dispatch, naive_battery_dispatch
from gridpulse.optimizer.impact import impact_summary
from gridpulse.monitoring.retraining import load_monitoring_config, compute_data_drift
from gridpulse.utils.metrics import rmse, mae, mape, smape, daylight_mape
from gridpulse.utils.scaler import StandardScaler


@dataclass
class ReportContext:
    repo_root: Path
    features_path: Path
    splits_dir: Path
    models_dir: Path
    reports_dir: Path


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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
        return _safe_float(obj)
    return obj


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if target == "solar_mw":
        out["daylight_mape"] = daylight_mape(y_true, y_pred)
    return out


def _clean_series(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if not np.isfinite(arr).all():
        mean = np.nanmean(arr)
        if not np.isfinite(mean):
            mean = 0.0
        arr = np.where(np.isfinite(arr), arr, mean)
    return arr


def _load_optimization_config(ctx: ReportContext) -> dict:
    cfg_path = ctx.repo_root / "configs" / "optimization.yaml"
    if cfg_path.exists():
        import yaml
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
            pred_hz = pred_seq[:, -horizon:]
            preds.append(pred_hz.reshape(-1))
            trues.append(yb.numpy().reshape(-1))
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    return _compute_metrics(y_true, y_pred, target)


def load_split_data(ctx: ReportContext):
    if (ctx.splits_dir / "test.parquet").exists():
        test_df = pd.read_parquet(ctx.splits_dir / "test.parquet")
        train_df = pd.read_parquet(ctx.splits_dir / "train.parquet")
        return train_df, test_df
    if ctx.features_path.exists():
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
                bundle = load_model_bundle(path)
                m = _eval_seq_model(bundle, test_df)
                if m:
                    metrics["targets"][target][kind] = {
                        **metrics["targets"][target].get(kind, {}),
                        **m,
                    }

    report_path.write_text(json.dumps(_sanitize(metrics), indent=2), encoding="utf-8")
    return metrics


def plot_arbitrage_dispatch(ctx: ReportContext, hours: np.ndarray, price_curve: np.ndarray, grid_load: np.ndarray, optimized_load: np.ndarray, battery_flow: np.ndarray):
    """
    Generates the Level-4 Arbitrage Optimization plot.
    battery_flow: positive = discharge, negative = charge
    """
    fig_dir = ctx.reports_dir / "figures"
    ensure_dir(fig_dir)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Market Price (The "Signal")
    color = 'tab:red'
    ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Electricity Price ($/MWh)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(hours, price_curve, color=color, linestyle='--', linewidth=2, label='Market Price', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create a second y-axis for Power (The "Action")
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Power (MW)', color=color, fontsize=12, fontweight='bold')

    # Plot 1: Baseline Load (The "Before")
    ax2.plot(hours, grid_load, color='gray', alpha=0.4, linewidth=2, label='Baseline Grid Load')

    # Plot 2: Optimized Load (The "After")
    ax2.plot(hours, optimized_load, color=color, linewidth=3, label='GridPulse Optimized Load')

    # Highlight the "Arbitrage" (The Level-4 Magic)
    # Green area = Charging (Money saved later)
    ax2.fill_between(hours, grid_load, optimized_load, 
                     where=(np.array(battery_flow) < 0), color='green', alpha=0.3, label='Charging (Low Price)')
    # Orange area = Discharging (Cost Avoided)
    ax2.fill_between(hours, grid_load, optimized_load, 
                     where=(np.array(battery_flow) > 0), color='orange', alpha=0.5, label='Discharging (High Price)')

    # --- 3. POLISH ---
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

    horizon = 24
    if len(test_df) < horizon:
        return None

    window = test_df.tail(horizon)
    load = _clean_series(window["load_mw"].to_numpy())
    wind = window["wind_mw"].to_numpy() if "wind_mw" in window.columns else np.zeros_like(load)
    solar = window["solar_mw"].to_numpy() if "solar_mw" in window.columns else np.zeros_like(load)
    if "price_eur_mwh" in window.columns:
        price = window["price_eur_mwh"].to_numpy()
    elif "price_usd_mwh" in window.columns:
        price = window["price_usd_mwh"].to_numpy()
    else:
        price = None
    carbon = window["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in window.columns else None
    renew = _clean_series(wind) + _clean_series(solar)

    cfg = _load_optimization_config(ctx)
    baseline = grid_only_dispatch(load, renew, cfg, price_series=price, carbon_series=carbon)
    naive = naive_battery_dispatch(load, renew, cfg, price_series=price, carbon_series=carbon)
    optimized = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carbon)

    impact = impact_summary(baseline, optimized)
    impact_naive = impact_summary(naive, optimized)

    baseline_peak = float(np.max(baseline["grid_mw"])) if baseline.get("grid_mw") else None
    optimized_peak = float(np.max(optimized["grid_mw"])) if optimized.get("grid_mw") else None
    peak_shaving_pct = None
    if baseline_peak and baseline_peak > 0 and optimized_peak is not None:
        peak_shaving_pct = (baseline_peak - optimized_peak) / baseline_peak * 100.0

    fig_dir = ctx.reports_dir / "figures"
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(baseline["grid_mw"], label="baseline grid", color="#1f77b4")
    ax[0].plot(optimized["grid_mw"], label="optimized grid", color="#ff7f0e")
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

    # Impact summary CSV + savings plot
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

    # Generate Arbitrage Plot if price data exists
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
    lines.append("This report compares dispatch outcomes for the same 24‑hour forecast window (last 24 hours of the test split).\n\n")
    lines.append(f"- Horizon: {horizon} hours\n")
    lines.append("- Forecast source: test split (proxy for day‑ahead forecast)\n")
    lines.append("- Config: `configs/optimization.yaml`\n\n")

    lines.append("## Policy Comparison\n")
    lines.append("| Policy | Cost (USD) | Carbon (kg) | Carbon Cost (USD) |\n")
    lines.append("|---|---:|---:|---:|\n")
    for name, plan in [
        ("Grid‑only baseline", baseline),
        ("Naive battery", naive),
        ("GridPulse (LP optimized)", optimized),
    ]:
        lines.append(
            f"| {name} | {_fmt(plan.get('expected_cost_usd'))} | {_fmt(plan.get('carbon_kg'))} | {_fmt(plan.get('carbon_cost_usd'))} |\n"
        )
    lines.append("\n")

    lines.append("## Savings vs Baseline (GridPulse vs Grid‑only)\n")
    lines.append(f"- Cost savings: {_fmt(impact.get('cost_savings_usd'))} ({_fmt_pct(impact.get('cost_savings_pct'))})\n")
    lines.append(f"- Carbon reduction: {_fmt(impact.get('carbon_reduction_kg'))} kg ({_fmt_pct(impact.get('carbon_reduction_pct'))})\n\n")

    lines.append("## Savings vs Naive Battery (GridPulse vs Naive)\n")
    lines.append(f"- Cost savings: {_fmt(impact_naive.get('cost_savings_usd'))} ({_fmt_pct(impact_naive.get('cost_savings_pct'))})\n")
    lines.append(f"- Carbon reduction: {_fmt(impact_naive.get('carbon_reduction_kg'))} kg ({_fmt_pct(impact_naive.get('carbon_reduction_pct'))})\n\n")

    lines.append("## Dispatch Comparison\n")
    lines.append("![](figures/dispatch_compare.png)\n")

    if arb_plot_path:
        lines.append("\n## Arbitrage Logic (Level-4)\n")
        lines.append("![](figures/arbitrage_optimization.png)\n")

    out_path = ctx.reports_dir / "impact_comparison.md"
    ensure_dir(ctx.reports_dir)
    out_path.write_text("".join(lines), encoding="utf-8")

    json_path = ctx.reports_dir / "impact_comparison.json"
    json_path.write_text(
        json.dumps(
            _sanitize(
                {
                    "horizon": horizon,
                    "baseline": baseline,
                    "naive": naive,
                    "optimized": optimized,
                    "impact_vs_baseline": impact,
                    "impact_vs_naive": impact_naive,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "report_path": out_path,
        "figure_path": fig_path,
        "summary_csv": impact_csv,
        "savings_plot": impact_plot,
    }


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


def build_formal_report(ctx: ReportContext, multi_horizon_plot: Path | None, metrics: dict | None = None):
    metrics = metrics or refresh_metrics_from_models(ctx)
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

    if multi_horizon_plot:
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

    multi = build_multi_horizon(ctx)
    plot_path = plot_multi_horizon(ctx, multi) if multi else None

    make_sample_figures(ctx)
    make_demo_gif(ctx)
    metrics = refresh_metrics_from_models(ctx)
    plot_model_comparison(ctx, metrics)
    build_model_cards(ctx, metrics)
    build_formal_report(ctx, plot_path, metrics)
    build_impact_report(ctx)

    print("Reports and figures generated.")


if __name__ == "__main__":
    main()
