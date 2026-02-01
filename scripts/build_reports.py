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

    print("Reports and figures generated.")


if __name__ == "__main__":
    main()
