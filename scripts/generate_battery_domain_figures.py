"""Generate four thesis-quality figures for the energy management (battery) domain.

Uses the locked DE ENTSO-E run artifacts:
  artifacts/runs/de/r1_gbm_fast_20260311/

Produces:
  reports/battery/figures/forecast_sample.png
  reports/battery/figures/model_comparison.png
  reports/battery/figures/drift_sample.png
  reports/battery/figures/multi_horizon_backtest.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.release.artifact_loader import load_joblib_artifact

RUN_DIR = REPO_ROOT / "artifacts/runs/de/r1_gbm_fast_20260311"
OUT_DIR = REPO_ROOT / "reports/battery/figures"
TARGET = "load_mw"
UNITS = "MW"
DISPLAY = "Energy Management (Battery)"

FAULT_COLOR = "#d62728"
BAND_COLOR = "#aec7e8"
PRED_COLOR = "#1f77b4"
ACT_COLOR = "#ff7f0e"
REL_COLOR = "#2ca02c"


def _load_test() -> dict[str, np.ndarray]:
    arr = np.load(RUN_DIR / f"backtests/{TARGET}_test.npz")
    return {k: arr[k] for k in arr}


def _load_cal() -> dict[str, np.ndarray]:
    arr = np.load(RUN_DIR / f"backtests/{TARGET}_calibration.npz")
    return {k: arr[k] for k in arr}


def _load_model():
    return load_joblib_artifact(RUN_DIR / f"models/gbm_lightgbm_{TARGET}.pkl")


def _conformal_quantile(residuals: np.ndarray, alpha: float = 0.10) -> float:
    return float(np.quantile(np.abs(residuals), 1.0 - alpha))


def _compute_wt(signal: np.ndarray, dropout_p: float = 0.15, spike_p: float = 0.08, window: int = 10):
    rng = np.random.default_rng(42)
    n = len(signal)
    faulted = signal.copy().astype(float)
    fault_mask = np.zeros(n, dtype=bool)
    drop_idx = rng.choice(n, size=int(n * dropout_p), replace=False)
    faulted[drop_idx] = np.nan
    fault_mask[drop_idx] = True
    std = float(np.nanstd(signal))
    spike_idx = rng.choice(n, size=int(n * spike_p), replace=False)
    faulted[spike_idx] = faulted[spike_idx] + rng.choice([-1, 1], size=len(spike_idx)) * 5 * std
    fault_mask[spike_idx] = True
    for i in range(1, n):
        if np.isnan(faulted[i]):
            faulted[i] = faulted[i - 1]
    clean = (~fault_mask).astype(float)
    w_t = np.array([clean[max(0, i - window) : i + 1].mean() for i in range(n)])
    return faulted, w_t, fault_mask


def fig_forecast_sample() -> None:
    test = _load_test()
    cal = _load_cal()

    # flatten windows → 1-D time series (column 0 = 1-step-ahead for each window)
    y_true = test["y_true"][:, 0]
    q_mid = 0.5 * (test["q_lo"][:, 0] + test["q_hi"][:, 0])

    cal_res = cal["y_true"][:, 0] - 0.5 * (cal["q_lo"][:, 0] + cal["q_hi"][:, 0])
    q_conf = _conformal_quantile(cal_res)
    lo = q_mid - q_conf
    hi = q_mid + q_conf

    n = min(120, len(y_true))
    xs = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(xs, lo[:n], hi[:n], alpha=0.35, color=BAND_COLOR, label="90% conformal band")
    ax.plot(xs, q_mid[:n], color=PRED_COLOR, lw=1.5, label="GBM forecast")
    ax.plot(xs, y_true[:n], color=ACT_COLOR, lw=1.0, alpha=0.85, label="Observed")
    ax.set_xlabel("Window step")
    ax.set_ylabel(f"{TARGET} ({UNITS})")
    ax.set_title(f"{DISPLAY} — Forecast Sample (first {n} test windows, h=1)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out = OUT_DIR / "forecast_sample.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO_ROOT)}")


def fig_model_comparison() -> None:
    test = _load_test()
    y_true = test["y_true"][:, 0]
    y_pred = 0.5 * (test["q_lo"][:, 0] + test["q_hi"][:, 0])

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def _smape(a, b):
        denom = (np.abs(a) + np.abs(b)) / 2
        denom[denom < 1e-9] = 1e-9
        return float(np.mean(np.abs(a - b) / denom))

    # persistence baseline: predict previous value
    persist = np.roll(y_true, 1)
    persist[0] = y_true[0]

    models = ["Persistence baseline", "DC3S GBM"]
    colors = [BAND_COLOR, PRED_COLOR]
    metrics = {
        "RMSE": [_rmse(y_true, persist), _rmse(y_true, y_pred)],
        "MAE": [_mae(y_true, persist), _mae(y_true, y_pred)],
        "sMAPE": [_smape(y_true, persist), _smape(y_true, y_pred)],
    }

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    for ax, (met, vals) in zip(axes, metrics.items(), strict=False):
        bars = ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, vals, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{v:.1f}" if met != "sMAPE" else f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_title(met)
        ax.set_ylabel(f"{TARGET} ({met})")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(f"{DISPLAY} — Model Comparison", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "model_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO_ROOT)}")


def fig_drift_sample() -> None:
    test = _load_test()
    sig = test["y_true"][:, 0]
    n = min(200, len(sig))
    sig = sig[:n]

    faulted, w_t, fault_mask = _compute_wt(sig)
    xs = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(xs, sig, color=ACT_COLOR, lw=1.2, label="Clean signal")
    ax1.plot(xs, faulted, color=FAULT_COLOR, lw=0.8, alpha=0.6, label="Faulted signal")
    in_fault = False
    f_start = 0
    for i in range(n):
        if fault_mask[i] and not in_fault:
            f_start = i
            in_fault = True
        elif not fault_mask[i] and in_fault:
            ax1.axvspan(f_start, i, alpha=0.15, color=FAULT_COLOR)
            in_fault = False
    if in_fault:
        ax1.axvspan(f_start, n, alpha=0.15, color=FAULT_COLOR)

    ax1.set_ylabel(f"{TARGET} ({UNITS})")
    ax1.set_title(f"{DISPLAY} — OQE Reliability Under Fault Injection")
    fault_patch = mpatches.Patch(color=FAULT_COLOR, alpha=0.3, label="Fault window")
    ax1.legend(
        handles=[
            plt.Line2D([0], [0], color=ACT_COLOR, lw=1.2, label="Clean signal"),
            plt.Line2D([0], [0], color=FAULT_COLOR, lw=0.8, alpha=0.8, label="Faulted signal"),
            fault_patch,
        ],
        fontsize=8,
        loc="upper right",
    )

    ax2.plot(xs, w_t, color=REL_COLOR, lw=1.5, label="Reliability $w_t$")
    ax2.axhline(0.7, color="black", ls="--", lw=0.8, alpha=0.6, label="Renewal threshold")
    ax2.axhline(0.4, color=FAULT_COLOR, ls=":", lw=0.8, alpha=0.8, label="Fallback threshold")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("$w_t$")
    ax2.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    out = OUT_DIR / "drift_sample.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO_ROOT)}")


def fig_multi_horizon_backtest() -> None:
    test = _load_test()
    _load_cal()
    horizons = list(range(1, test["y_true"].shape[1] + 1))  # 1..24

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def _smape(a, b):
        denom = (np.abs(a) + np.abs(b)) / 2
        denom[denom < 1e-9] = 1e-9
        return float(np.mean(np.abs(a - b) / denom))

    gbm_rmse, gbm_mae, gbm_smape = [], [], []
    pers_rmse, pers_mae, pers_smape = [], [], []

    for h_idx in range(len(horizons)):
        yt = test["y_true"][:, h_idx]
        yp = 0.5 * (test["q_lo"][:, h_idx] + test["q_hi"][:, h_idx])
        # persistence: repeat h=1 prediction
        yp0 = 0.5 * (test["q_lo"][:, 0] + test["q_hi"][:, 0])
        gbm_rmse.append(_rmse(yt, yp))
        gbm_mae.append(_mae(yt, yp))
        gbm_smape.append(_smape(yt, yp))
        pers_rmse.append(_rmse(yt, yp0))
        pers_mae.append(_mae(yt, yp0))
        pers_smape.append(_smape(yt, yp0))

    xs = horizons
    metrics = [
        ("RMSE", gbm_rmse, pers_rmse),
        ("MAE", gbm_mae, pers_mae),
        ("sMAPE", gbm_smape, pers_smape),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, (met, gbm_ys, pers_ys) in zip(axes, metrics, strict=False):
        ax.plot(xs, pers_ys, color=BAND_COLOR, ls="--", marker="o", ms=3, lw=1.5, label="Persistence")
        ax.plot(xs, gbm_ys, color=PRED_COLOR, ls="-", marker="o", ms=3, lw=1.5, label="DC3S GBM")
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel(met)
        ax.set_title(f"{met} vs Horizon")
        ax.legend(fontsize=7)

    fig.suptitle(rf"{DISPLAY} — Multi-Horizon Backtest (DE load\_mw)", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "multi_horizon_backtest.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO_ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[battery] Generating 4 figures...")
    for fn in (fig_forecast_sample, fig_model_comparison, fig_drift_sample, fig_multi_horizon_backtest):
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
