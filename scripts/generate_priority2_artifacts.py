#!/usr/bin/env python3
"""Generate Priority-2 evidence artifacts."""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _build_hyperparameter_surface() -> dict[str, str]:
    ablation = ROOT / "reports/publication/ablation_table.csv"
    if not ablation.exists():
        raise FileNotFoundError("Expected reports/publication/ablation_table.csv from ablation sweep")
    df = pd.read_csv(ablation)
    dc3s = df[df["controller"] == "dc3s_wrapped"].copy()
    surf = (
        dc3s.groupby(["k_quality", "k_drift", "infl_max"], as_index=False)
        .agg(
            violation_rate_mean=("violation_rate", "mean"),
            picp_90_mean=("picp_90", "mean"),
            mean_interval_width_mean=("mean_interval_width", "mean"),
        )
        .sort_values(["k_quality", "k_drift", "infl_max"])
    )
    csv_path = ROOT / "reports/publication/hyperparameter_surfaces.csv"
    surf.to_csv(csv_path, index=False)

    # 2D slice for infl_max=2.0
    slice_df = surf[np.isclose(surf["infl_max"], 2.0)]
    pivot = slice_df.pivot(index="k_quality", columns="k_drift", values="violation_rate_mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel("k_drift")
    ax.set_ylabel("k_quality")
    ax.set_title("Violation surface (infl_max=2.0)")
    plt.colorbar(im, ax=ax, label="Violation rate")
    fig.tight_layout()
    fig_path = ROOT / "reports/publication/hyperparameter_surface.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return {"csv": str(csv_path), "figure": str(fig_path)}


def _build_blackout_half_life() -> dict[str, str]:
    trace = pd.read_csv(ROOT / "reports/publication/48h_trace.csv")
    fault = trace["fault_active"].to_numpy(dtype=bool)
    width_col = "interval_width_mwh" if "interval_width_mwh" in trace.columns else "interval_width_mw"
    width = trace[width_col].to_numpy(dtype=float)
    if not fault.any():
        raise RuntimeError("48h trace has no fault-active rows")
    onset = int(np.where(fault)[0][0])
    baseline = float(np.median(width[max(0, onset - 5):onset])) if onset > 0 else float(width[0])
    peak = float(np.max(width[onset:onset + 6])) if onset < len(width) - 1 else float(width[onset])
    target = baseline + 0.5 * (peak - baseline)
    half_idx = None
    for i in range(onset, len(width)):
        if width[i] <= target:
            half_idx = i
            break
    half_life_steps = int(half_idx - onset) if half_idx is not None else -1
    out = pd.DataFrame(
        [
            {
                "fault_onset_step": onset,
                "baseline_width_mwh": baseline,
                "peak_width_mwh": peak,
                "half_target_width_mwh": target,
                "half_life_steps": half_life_steps,
            }
        ]
    )
    csv_path = ROOT / "reports/publication/blackout_half_life.csv"
    out.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(width, label="interval width")
    ax.axvline(onset, color="red", linestyle="--", label="fault onset")
    ax.axhline(target, color="orange", linestyle=":", label="half-life target")
    ax.set_xlabel("step")
    ax.set_ylabel("width (MWh)")
    ax.legend(loc="upper right")
    ax.set_title("Blackout half-life response")
    fig.tight_layout()
    fig_path = ROOT / "reports/publication/blackout_half_life.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return {"csv": str(csv_path), "figure": str(fig_path)}


def _build_graceful_degradation() -> dict[str, str]:
    trace = pd.read_csv(ROOT / "reports/publication/48h_trace.csv")
    rel_col = "reliability_w" if "reliability_w" in trace.columns else "w_t"
    width_col = "interval_width_mwh" if "interval_width_mwh" in trace.columns else "interval_width_mw"
    out = trace[
        [
            "timestamp",
            rel_col,
            width_col,
            "proposed_discharge_mw",
            "safe_discharge_mw",
            "intervened",
            "fault_active",
        ]
    ].copy()
    out = out.rename(columns={rel_col: "reliability_w", width_col: "interval_width_mwh"})
    out["action_gap_mw"] = (out["proposed_discharge_mw"] - out["safe_discharge_mw"]).abs()
    csv_path = ROOT / "reports/publication/graceful_degradation_trace.csv"
    out.to_csv(csv_path, index=False)

    t = pd.to_datetime(out["timestamp"], utc=True)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, out["reliability_w"], color="tab:blue", label="reliability_w")
    ax1.set_ylabel("w_t", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(t, out["action_gap_mw"], color="tab:red", label="|candidate-safe|")
    ax2.set_ylabel("Action gap (MW)", color="tab:red")
    ax1.set_title("Graceful degradation trace")
    fig.tight_layout()
    fig_path = ROOT / "reports/publication/graceful_degradation_trace.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return {"csv": str(csv_path), "figure": str(fig_path)}


def main() -> None:
    outputs = {
        "hyperparameter_surfaces": _build_hyperparameter_surface(),
        "blackout_half_life": _build_blackout_half_life(),
        "graceful_degradation": _build_graceful_degradation(),
    }
    manifest = ROOT / "reports/publication/priority2_artifacts_manifest.json"
    manifest.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()

