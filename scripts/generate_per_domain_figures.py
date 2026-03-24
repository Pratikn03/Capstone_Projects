"""Generate four thesis-quality figures for each peer domain.

Produces (for industrial, healthcare, aerospace):
  reports/{domain}/figures/forecast_sample.png   — forecast vs actual + 90% conformal band
  reports/{domain}/figures/model_comparison.png  — persistence vs GBM metrics bar chart
  reports/{domain}/figures/drift_sample.png      — reliability score w_t under fault injection
  reports/{domain}/figures/multi_horizon_backtest.png — RMSE/MAE/sMAPE vs horizon

Requires:
  data/{domain}/processed/splits/test.parquet
  artifacts/models_{domain}/gbm_lightgbm_{target}.pkl
  artifacts/backtests/{domain}/gbm_{target}_calibration.npz
  reports/{domain}/multi_horizon_backtest.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]

DOMAINS: dict[str, dict] = {
    "av": {
        "target": "speed_mps",
        "display": "Autonomous Vehicles",
        "units": "m/s",
        "constraint_lo": 0.0,
        "constraint_hi": 11.0,
    },
    "industrial": {
        "target": "power_mw",
        "display": "Industrial Process Control",
        "units": "MW",
        "constraint_lo": 430.0,
        "constraint_hi": 500.0,
    },
    "healthcare": {
        "target": "hr_bpm",
        "display": "Medical Monitoring (ICU Vitals)",
        "units": "bpm",
        "constraint_lo": 50.0,
        "constraint_hi": 130.0,
    },
    "aerospace": {
        "target": "airspeed_kt",
        "display": "Aerospace (Flight Envelope)",
        "units": "kt",
        "constraint_lo": 60.0,
        "constraint_hi": 350.0,
    },
    # Navigation uses a closed-loop simulation surface with no trained forecasting
    # model; it appears in the multi-domain comparison table but not in the
    # per-domain training evidence profiles.
}

FAULT_COLOR = "#d62728"   # red for fault windows
BAND_COLOR  = "#aec7e8"   # light blue for conformal band
PRED_COLOR  = "#1f77b4"   # blue for predicted
ACT_COLOR   = "#ff7f0e"   # orange for actual
REL_COLOR   = "#2ca02c"   # green for reliability


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_model(domain: str, target: str):
    path = REPO_ROOT / f"artifacts/models_{domain}/gbm_lightgbm_{target}.pkl"
    return joblib.load(path)


def _load_test(domain: str) -> pd.DataFrame:
    return pd.read_parquet(REPO_ROOT / f"data/{domain}/processed/splits/test.parquet")


def _load_cal(domain: str, target: str) -> dict:
    path = REPO_ROOT / f"artifacts/backtests/{domain}/gbm_{target}_calibration.npz"
    arr = np.load(path)
    return {k: arr[k].ravel() for k in arr.keys()}


def _load_backtest(domain: str) -> dict:
    path = REPO_ROOT / f"reports/{domain}/multi_horizon_backtest.json"
    with open(path) as f:
        return json.load(f)


def _out_dir(domain: str) -> Path:
    d = REPO_ROOT / f"reports/{domain}/figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _conformal_quantile(residuals: np.ndarray, alpha: float = 0.10) -> float:
    """Return the (1-alpha) quantile of absolute residuals."""
    q = np.quantile(np.abs(residuals), 1.0 - alpha)
    return float(q)


# ---------------------------------------------------------------------------
# Figure 1: forecast_sample
# ---------------------------------------------------------------------------

def fig_forecast_sample(domain: str, cfg: dict) -> None:
    target = cfg["target"]
    units  = cfg["units"]
    display = cfg["display"]

    mdl = _load_model(domain, target)
    df  = _load_test(domain)
    cal = _load_cal(domain, target)

    # conformal quantile from calibration residuals
    cal_residuals = cal["y_true"] - 0.5 * (cal["q_lo"] + cal["q_hi"])
    q_conf = _conformal_quantile(cal_residuals, alpha=0.10)

    # predict on test
    feat_cols = mdl["feature_cols"]
    X = df[feat_cols].values
    y_pred = mdl["model"].predict(X)
    y_true = df[target].values

    # use first 120 steps for readability
    n = min(120, len(df))
    xs  = np.arange(n)
    yt  = y_true[:n]
    yp  = y_pred[:n]
    lo  = yp - q_conf
    hi  = yp + q_conf

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(xs, lo, hi, alpha=0.35, color=BAND_COLOR, label="90% conformal band")
    ax.plot(xs, yp, color=PRED_COLOR, lw=1.5, label="GBM forecast")
    ax.plot(xs, yt,  color=ACT_COLOR,  lw=1.0, alpha=0.85, label="Observed")

    # constraint lines
    lo_c = cfg.get("constraint_lo")
    hi_c = cfg.get("constraint_hi")
    if lo_c is not None:
        ax.axhline(lo_c, color="black", ls="--", lw=0.8, alpha=0.6, label=f"Safety bounds")
    if hi_c is not None:
        ax.axhline(hi_c, color="black", ls="--", lw=0.8, alpha=0.6)

    ax.set_xlabel("Step")
    ax.set_ylabel(f"{target} ({units})")
    ax.set_title(f"{display} — Forecast Sample (first {n} test steps)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out = _out_dir(domain) / "forecast_sample.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2: model_comparison  (persistence vs GBM)
# ---------------------------------------------------------------------------

def fig_model_comparison(domain: str, cfg: dict) -> None:
    target  = cfg["target"]
    display = cfg["display"]
    bt = _load_backtest(domain)
    targets_data = bt.get("targets", {})

    # gather summary metrics at max horizon from each model
    tgt_data = targets_data.get(target, {})
    max_hz = str(max(int(k) for k in (bt.get("horizons") or [24])))

    metrics_order = ["rmse", "mae", "smape"]
    models = ["persistence", "gbm"]
    colors = ["#aec7e8", PRED_COLOR]
    labels = ["Persistence baseline", "DC3S GBM"]

    results: dict[str, dict[str, float]] = {}
    for mdl in models:
        summ = tgt_data.get(mdl, {}).get("results", {}).get(max_hz, {}).get("summary", {})
        results[mdl] = summ

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    metric_labels = {"rmse": "RMSE", "mae": "MAE", "smape": "sMAPE"}
    for ax, met in zip(axes, metrics_order):
        vals = [results.get(m, {}).get(met, 0.0) for m in models]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
        # annotate values
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(metric_labels[met])
        ax.set_ylabel(f"{target} ({metric_labels[met]})")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(f"{display} — Model Comparison (horizon {max_hz})", fontsize=9)
    plt.tight_layout()
    out = _out_dir(domain) / "model_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 3: drift_sample (reliability w_t under fault injection)
# ---------------------------------------------------------------------------

def _compute_wt(signal: np.ndarray, dropout_p: float = 0.15,
                spike_p: float = 0.08, window: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple OQE proxy: inject faults into signal, compute rolling fraction of clean steps.
    Returns (faulted_signal, w_t).
    """
    rng = np.random.default_rng(42)
    n = len(signal)
    faulted = signal.copy().astype(float)
    fault_mask = np.zeros(n, dtype=bool)

    # dropout → NaN → replaced by last valid
    drop_idx = rng.choice(n, size=int(n * dropout_p), replace=False)
    faulted[drop_idx] = np.nan
    fault_mask[drop_idx] = True

    # spike → replace with out-of-range value
    std = np.nanstd(signal)
    spike_idx = rng.choice(n, size=int(n * spike_p), replace=False)
    faulted[spike_idx] = faulted[spike_idx] + rng.choice([-1, 1], size=len(spike_idx)) * 5 * std
    fault_mask[spike_idx] = True

    # forward-fill NaN
    for i in range(1, n):
        if np.isnan(faulted[i]):
            faulted[i] = faulted[i - 1]

    # w_t = rolling fraction of clean steps
    clean = (~fault_mask).astype(float)
    w_t = np.array([
        clean[max(0, i - window):i + 1].mean()
        for i in range(n)
    ])
    return faulted, w_t, fault_mask


def fig_drift_sample(domain: str, cfg: dict) -> None:
    target  = cfg["target"]
    units   = cfg["units"]
    display = cfg["display"]

    df  = _load_test(domain)
    sig = df[target].values
    n   = min(200, len(sig))
    sig = sig[:n]

    faulted, w_t, fault_mask = _compute_wt(sig)
    xs = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # top: signal with fault shading
    ax1.plot(xs, sig,    color=ACT_COLOR,  lw=1.2, label="Clean signal")
    ax1.plot(xs, faulted, color=FAULT_COLOR, lw=0.8, alpha=0.6, label="Faulted signal")
    # shade fault windows
    in_fault = False
    f_start  = 0
    for i in range(n):
        if fault_mask[i] and not in_fault:
            f_start  = i
            in_fault = True
        elif not fault_mask[i] and in_fault:
            ax1.axvspan(f_start, i, alpha=0.15, color=FAULT_COLOR)
            in_fault = False
    if in_fault:
        ax1.axvspan(f_start, n, alpha=0.15, color=FAULT_COLOR)

    ax1.set_ylabel(f"{target} ({units})")
    ax1.set_title(f"{display} — OQE Reliability Under Fault Injection")
    ax1.legend(fontsize=8, loc="upper right")

    # bottom: w_t reliability score
    ax2.plot(xs, w_t, color=REL_COLOR, lw=1.5, label="Reliability $w_t$")
    ax2.axhline(0.7, color="black", ls="--", lw=0.8, alpha=0.6, label="Renewal threshold")
    ax2.axhline(0.4, color=FAULT_COLOR, ls=":", lw=0.8, alpha=0.8, label="Fallback threshold")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("$w_t$")
    ax2.legend(fontsize=7, loc="lower right")

    fault_patch = mpatches.Patch(color=FAULT_COLOR, alpha=0.3, label="Fault window")
    ax1.add_artist(ax1.legend(handles=[
        plt.Line2D([0], [0], color=ACT_COLOR, lw=1.2, label="Clean signal"),
        plt.Line2D([0], [0], color=FAULT_COLOR, lw=0.8, alpha=0.8, label="Faulted signal"),
        fault_patch,
    ], fontsize=8, loc="upper right"))

    plt.tight_layout()
    out = _out_dir(domain) / "drift_sample.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 4: multi_horizon_backtest
# ---------------------------------------------------------------------------

def fig_multi_horizon_backtest(domain: str, cfg: dict) -> None:
    target  = cfg["target"]
    display = cfg["display"]
    bt = _load_backtest(domain)
    horizons = bt.get("horizons", [1, 3, 6, 12, 24])
    tgt_data = bt.get("targets", {}).get(target, {})

    metrics = ["rmse", "mae", "smape"]
    metric_labels = {"rmse": "RMSE", "mae": "MAE", "smape": "sMAPE"}
    models = [("persistence", "Persistence", "#aec7e8", "--"),
              ("gbm",         "DC3S GBM",    PRED_COLOR, "-")]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, met in zip(axes, metrics):
        for mdl_key, mdl_label, color, ls in models:
            ys = []
            xs = []
            for h in horizons:
                summ = tgt_data.get(mdl_key, {}).get("results", {}).get(str(h), {}).get("summary", {})
                if met in summ:
                    ys.append(summ[met])
                    xs.append(h)
            if ys:
                ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=4, lw=1.5, label=mdl_label)
        ax.set_xlabel("Horizon (steps)")
        ax.set_ylabel(metric_labels[met])
        ax.set_title(f"{metric_labels[met]} vs Horizon")
        ax.legend(fontsize=7)

    fig.suptitle(f"{display} — Multi-Horizon Backtest", fontsize=9)
    plt.tight_layout()
    out = _out_dir(domain) / "multi_horizon_backtest.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    for domain, cfg in DOMAINS.items():
        print(f"\n[{domain}] Generating 4 figures...")
        try:
            fig_forecast_sample(domain, cfg)
        except Exception as e:
            print(f"  ✗ forecast_sample: {e}")
        try:
            fig_model_comparison(domain, cfg)
        except Exception as e:
            print(f"  ✗ model_comparison: {e}")
        try:
            fig_drift_sample(domain, cfg)
        except Exception as e:
            print(f"  ✗ drift_sample: {e}")
        try:
            fig_multi_horizon_backtest(domain, cfg)
        except Exception as e:
            print(f"  ✗ multi_horizon_backtest: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
