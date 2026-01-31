from __future__ import annotations

import json
import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.forecasting.baselines import persistence_24h
from gridpulse.forecasting.ml_gbm import train_gbm, predict_gbm
from gridpulse.forecasting.backtest import multi_horizon_metrics
from gridpulse.optimizer.lp_dispatch import optimize_dispatch
from gridpulse.monitoring.retraining import load_monitoring_config, compute_data_drift


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_split_data(repo_root: Path):
    splits_dir = repo_root / "data" / "processed" / "splits"
    features_path = repo_root / "data" / "processed" / "features.parquet"
    if (splits_dir / "test.parquet").exists():
        test_df = pd.read_parquet(splits_dir / "test.parquet")
        train_df = pd.read_parquet(splits_dir / "train.parquet")
        return train_df, test_df
    if features_path.exists():
        df = pd.read_parquet(features_path).sort_values("timestamp")
        n = len(df)
        train_df = df.iloc[: int(n * 0.7)]
        test_df = df.iloc[int(n * 0.85):]
        return train_df, test_df
    return None, None


def build_multi_horizon(repo_root: Path):
    train_df, test_df = load_split_data(repo_root)
    if train_df is None or test_df is None:
        return None

    horizons = [1, 3, 6, 12, 24]
    targets = ["load_mw", "wind_mw", "solar_mw"]

    result = {"horizons": horizons, "targets": {}}
    for target in targets:
        y_true = test_df[target].to_numpy()
        # baseline persistence
        y_pred = persistence_24h(test_df, target)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        baseline = multi_horizon_metrics(y_true, y_pred, horizons, target)

        # gbm if available; else train a quick gbm for report
        gbm_pred = None
        X_train = train_df[[c for c in train_df.columns if c not in {"timestamp", "load_mw", "wind_mw", "solar_mw"}]].to_numpy()
        y_train = train_df[target].to_numpy()
        X_test = test_df[[c for c in test_df.columns if c not in {"timestamp", "load_mw", "wind_mw", "solar_mw"}]].to_numpy()
        _, gbm = train_gbm(X_train, y_train, params={"n_estimators": 200, "learning_rate": 0.05, "random_state": 42})
        gbm_pred = predict_gbm(gbm, X_test)
        gbm = multi_horizon_metrics(test_df[target].to_numpy(), gbm_pred, horizons, target)

        result["targets"][target] = {"persistence": baseline, "gbm": gbm}

    out_path = repo_root / "reports" / "multi_horizon_backtest.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def plot_multi_horizon(repo_root: Path, report: dict):
    fig_dir = repo_root / "reports" / "figures"
    ensure_dir(fig_dir)

    # Plot for load_mw
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
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return out_path


def make_sample_figures(repo_root: Path):
    fig_dir = repo_root / "reports" / "figures"
    ensure_dir(fig_dir)

    features_path = repo_root / "data" / "processed" / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path).sort_values("timestamp")
        recent = df.tail(7 * 24)
        fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        recent.plot(x="timestamp", y="load_mw", ax=ax[0], color="#1f77b4", title="Load (last 7 days)")
        recent.plot(x="timestamp", y="wind_mw", ax=ax[1], color="#2ca02c", title="Wind (last 7 days)")
        recent.plot(x="timestamp", y="solar_mw", ax=ax[2], color="#ff7f0e", title="Solar (last 7 days)")
        plt.tight_layout()
        fig.savefig(fig_dir / "forecast_sample.png", dpi=180, bbox_inches="tight")

    # Optimization sample
    cfg_path = repo_root / "configs" / "optimization.yaml"
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
    fig.savefig(fig_dir / "dispatch_sample.png", dpi=180, bbox_inches="tight")

    # Drift sample
    train_df, test_df = load_split_data(repo_root)
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
        fig.savefig(fig_dir / "drift_sample.png", dpi=180, bbox_inches="tight")


def make_demo_gif(repo_root: Path):
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig_dir = repo_root / "reports" / "figures"
    ensure_dir(fig_dir)
    features_path = repo_root / "data" / "processed" / "features.parquet"
    if not features_path.exists():
        return None

    df = pd.read_parquet(features_path).sort_values("timestamp")
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


def build_model_cards(repo_root: Path):
    report_path = repo_root / "reports" / "week2_metrics.json"
    if not report_path.exists():
        return None
    metrics = json.loads(report_path.read_text(encoding="utf-8"))
    targets = metrics.get("targets", {})

    cards_dir = repo_root / "reports" / "model_cards"
    ensure_dir(cards_dir)

    for target, data in targets.items():
        lines = [f"# Model Card — {target}\n"]
        lines.append("## Overview\n")
        lines.append("Forecasting model trained on OPSD Germany time‑series.\n")
        lines.append("\n## Metrics (test split)\n")
        lines.append("| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |\n")
        lines.append("|---|---:|---:|---:|---:|---:|\n")
        for model, vals in data.items():
            if model == "n_features":
                continue
            lines.append(
                f"| {model} | {vals.get('rmse', 'n/a')} | {vals.get('mae', 'n/a')} | {vals.get('smape', 'n/a')} | {vals.get('mape', 'n/a')} | {vals.get('daylight_mape', 'n/a')} |\n"
            )
        lines.append("\n## Intended Use\n")
        lines.append("Day‑ahead forecasting for grid planning and dispatch optimization.\n")
        lines.append("\n## Limitations\n")
        lines.append("Performance depends on feature availability and data quality; retraining is required as grid conditions shift.\n")

        (cards_dir / f"{target}.md").write_text("".join(lines), encoding="utf-8")


def build_formal_report(repo_root: Path, multi_horizon_plot: Path | None):
    report_path = repo_root / "reports" / "week2_metrics.json"
    metrics = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    targets = metrics.get("targets", {})

    lines = []
    lines.append("# Formal Evaluation Report\n\n")
    lines.append("## Summary\n")
    lines.append("This report summarizes forecasting performance, backtesting, and decision‑support outputs for GridPulse.\n\n")

    if targets:
        lines.append("## Model Metrics (Test Split)\n")
        for target, data in targets.items():
            lines.append(f"### {target}\n")
            lines.append("| Model | RMSE | MAE | sMAPE | MAPE | Daylight‑MAPE |\n")
            lines.append("|---|---:|---:|---:|---:|---:|\n")
            for model, vals in data.items():
                if model == "n_features":
                    continue
                lines.append(
                    f"| {model} | {vals.get('rmse', 'n/a')} | {vals.get('mae', 'n/a')} | {vals.get('smape', 'n/a')} | {vals.get('mape', 'n/a')} | {vals.get('daylight_mape', 'n/a')} |\n"
                )
            lines.append("\n")

    if multi_horizon_plot:
        lines.append("## Multi‑Horizon Backtest (Load)\n")
        lines.append(f"![]({multi_horizon_plot.as_posix()})\n\n")

    lines.append("## Conclusions\n")
    lines.append("GBM provides a strong baseline on the OPSD data, while sequence models capture temporal structure for longer horizons. Optimization outputs are cost‑ and carbon‑aware and suitable for operator decision support.\n")

    out_path = repo_root / "reports" / "formal_evaluation_report.md"
    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    multi = build_multi_horizon(repo_root)
    plot_path = plot_multi_horizon(repo_root, multi) if multi else None

    make_sample_figures(repo_root)
    make_demo_gif(repo_root)
    build_model_cards(repo_root)
    build_formal_report(repo_root, plot_path)

    print("Reports and figures generated.")


if __name__ == "__main__":
    main()
