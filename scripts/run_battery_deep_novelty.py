#!/usr/bin/env python3
"""Battery-scoped deep-learning novelty package for ORIUS.

This script creates a bounded deep-learning artifact family without mutating the
official parity gate or the locked engineered forecasting baseline tables.

Outputs:
- learned DeepOQE checkpoint + diagnostics
- heuristic-vs-deep battery replay comparison under the existing CPSBench loop
- raw-sequence PatchTST benchmark against the engineered-track GBM baseline
- manuscript-ready tables and figures copied into paper assets
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
from textwrap import dedent
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from orius.cpsbench_iot.metrics import compute_forecast_metrics
from orius.cpsbench_iot.runner import _to_telemetry_events, run_single
from orius.cpsbench_iot.scenarios import DEFAULT_SCENARIOS, generate_episode
from orius.dc3s.certificate import recompute_certificate_hash, store_certificates_batch
from orius.dc3s.deep_oqe import (
    DeepOQEConfig,
    DeepOQEModel,
    FEATURE_NAMES,
    extract_feature_vector,
    save_model,
)
from orius.dc3s.quality import compute_reliability
from orius.forecasting.dl_patchtst import PatchTSTForecaster
from orius.orius_bench.metrics_engine import StepRecord, compute_all_metrics as compute_bench_metrics


DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "publication"
DEFAULT_PAPER_TABLE_DIR = REPO_ROOT / "paper" / "assets" / "tables" / "generated"
DEFAULT_PAPER_FIG_DIR = REPO_ROOT / "paper" / "assets" / "figures"
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "deep_oqe"
ENGINEERED_BASELINE_PATH = REPO_ROOT / "reports" / "week2_metrics.json"
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"

TRAIN_SCENARIOS = list(DEFAULT_SCENARIOS) + ["stale_sensor"]
EVAL_SCENARIOS = list(DEFAULT_SCENARIOS)
TRAIN_SEEDS = [11, 22, 33, 44]
EVAL_SEEDS = [55, 66]
REPLAY_SCENARIOS = ["nominal", "dropout", "stale_sensor"]
REPLAY_SEEDS = [55]
REPLAY_HORIZON = 96
RAW_TRACK_INPUTS = [
    "load_mw",
    "wind_mw",
    "solar_mw",
    "price_eur_mwh",
    "carbon_kg_per_mwh",
    "fault_dropout",
    "fault_stale_sensor",
    "fault_spikes",
    "fault_delay_jitter",
    "fault_out_of_order",
    "reliability_w",
]
FORECAST_TARGETS = ["load_mw", "wind_mw", "solar_mw"]


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bounded battery-scoped ORIUS deep-learning novelty package.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--paper-table-dir", type=Path, default=DEFAULT_PAPER_TABLE_DIR)
    parser.add_argument("--paper-fig-dir", type=Path, default=DEFAULT_PAPER_FIG_DIR)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--engineered-baseline", type=Path, default=ENGINEERED_BASELINE_PATH)
    parser.add_argument("--deep-oqe-epochs", type=int, default=12)
    parser.add_argument("--forecast-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=6)
    parser.add_argument("--eval-stride", type=int, default=12)
    return parser.parse_args()


def _target_reliability(
    *,
    load_true: float,
    load_obs: float,
    renew_true: float,
    renew_obs: float,
    fault_row: Mapping[str, Any],
) -> float:
    load_err = abs(float(load_true) - float(load_obs)) / max(abs(float(load_true)), 1.0)
    renew_err = abs(float(renew_true) - float(renew_obs)) / max(abs(float(renew_true)), 1.0)
    delay_steps = float(fault_row.get("delay_steps", 0.0) or 0.0)
    ooo_steps = float(fault_row.get("out_of_order_steps", 0.0) or 0.0)
    spike_sigma = float(fault_row.get("spike_sigma", 0.0) or 0.0)
    boolean_faults = mean(
        [
            float(bool(fault_row.get("dropout", 0))),
            float(bool(fault_row.get("stale_sensor", 0))),
            float(bool(fault_row.get("delay_jitter", 0))),
            float(bool(fault_row.get("out_of_order", 0))),
            float(bool(fault_row.get("spikes", 0))),
        ]
    )
    severity = (
        2.8 * load_err
        + 1.8 * renew_err
        + 0.20 * min(delay_steps, 4.0)
        + 0.15 * min(ooo_steps, 4.0)
        + 0.12 * spike_sigma
        + 0.70 * boolean_faults
    )
    return float(np.clip(1.0 - severity, 0.05, 1.0))


def _pad_sequence(rows: Sequence[np.ndarray], seq_len: int) -> np.ndarray:
    history = [np.asarray(row, dtype=np.float32) for row in rows][-seq_len:]
    if len(history) < seq_len:
        pad = [np.zeros(len(FEATURE_NAMES), dtype=np.float32) for _ in range(seq_len - len(history))]
        history = pad + history
    return np.stack(history, axis=0)


def build_deep_oqe_training_frame(seq_len: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in TRAIN_SCENARIOS:
        for seed in TRAIN_SEEDS + EVAL_SEEDS:
            x_obs, x_true, event_log = generate_episode(scenario=scenario, seed=seed, horizon=168)
            telemetry_events = _to_telemetry_events(x_obs=x_obs, event_log=event_log)
            prev_event: Mapping[str, Any] | None = None
            adaptive_state: dict[str, Any] = {}
            episode_features: list[np.ndarray] = []
            heuristic_w_values: list[float] = []
            for idx, event in enumerate(telemetry_events):
                heuristic_w, flags = compute_reliability(
                    event=event,
                    last_event=prev_event,
                    expected_cadence_s=3600.0,
                    reliability_cfg={"backend": "heuristic", "min_w": 0.05},
                    adaptive_state=adaptive_state,
                    ftit_cfg={"stale_k": 3, "stale_tol": 1.0e-9},
                )
                feature = extract_feature_vector(
                    event=event,
                    last_event=prev_event,
                    missing_fraction=float(flags.get("missing_fraction", 0.0)),
                    delay_seconds=float(flags.get("delay_seconds", 0.0)),
                    out_of_order=bool(flags.get("out_of_order", False)),
                    spike_ratio=float(flags.get("spike_ratio", 0.0)),
                    ooo_fraction_in_order=float(flags.get("ooo_fraction_in_order", 1.0)),
                    stale_tracker=flags.get("stale_tracker"),
                    fault_flags=flags.get("fault_flags"),
                )
                episode_features.append(feature)
                heuristic_w_values.append(float(heuristic_w))
                fault_row = event_log.iloc[idx].to_dict()
                rows.append(
                    {
                        "episode_id": f"{scenario}:{seed}",
                        "scenario": scenario,
                        "seed": seed,
                        "step": idx,
                        "heuristic_w": float(heuristic_w),
                        "target_w": _target_reliability(
                            load_true=float(x_true.loc[idx, "load_mw"]),
                            load_obs=float(x_obs.loc[idx, "load_mw"]),
                            renew_true=float(x_true.loc[idx, "renewables_mw"]),
                            renew_obs=float(x_obs.loc[idx, "renewables_mw"]),
                            fault_row=fault_row,
                        ),
                        "fault_present": int(
                            bool(fault_row.get("dropout"))
                            or bool(fault_row.get("stale_sensor"))
                            or bool(fault_row.get("delay_jitter"))
                            or bool(fault_row.get("out_of_order"))
                            or bool(fault_row.get("spikes"))
                        ),
                        "sequence": _pad_sequence(episode_features, seq_len=seq_len),
                    }
                )
                prev_event = event
    frame = pd.DataFrame(rows)
    return frame


def _stack_sequences(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = np.stack(frame["sequence"].to_list(), axis=0).astype(np.float32)
    target_w = frame["target_w"].to_numpy(dtype=np.float32)
    faults = frame["fault_present"].to_numpy(dtype=np.float32)
    return sequences, target_w, faults


def train_deep_oqe(
    frame: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    seq_len: int,
    model_path: Path,
) -> dict[str, Any]:
    train_mask = frame["seed"].isin(TRAIN_SEEDS)
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    eval_frame = frame.loc[~train_mask].reset_index(drop=True)

    x_train, y_train, f_train = _stack_sequences(train_frame)
    x_eval, y_eval, f_eval = _stack_sequences(eval_frame)

    cfg = DeepOQEConfig(seq_len=seq_len, hidden_size=48, num_layers=1, dropout=0.10, min_w=0.05)
    model = DeepOQEModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(f_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    history: list[dict[str, float]] = []
    for epoch in range(max(1, epochs)):
        model.train()
        batch_losses: list[float] = []
        for xb, yb, fb in train_loader:
            pred_w, fault_logit = model(xb)
            loss = F.mse_loss(pred_w, yb) + 0.25 * F.binary_cross_entropy_with_logits(fault_logit, fb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0})

    model.eval()
    with torch.no_grad():
        pred_train, fault_train = model(torch.from_numpy(x_train))
        pred_eval, fault_eval = model(torch.from_numpy(x_eval))

    pred_train_np = pred_train.detach().cpu().numpy()
    pred_eval_np = pred_eval.detach().cpu().numpy()
    fault_prob_eval = torch.sigmoid(fault_eval).detach().cpu().numpy()
    fault_prob_train = torch.sigmoid(fault_train).detach().cpu().numpy()

    metadata = {
        "train_scenarios": TRAIN_SCENARIOS,
        "train_seeds": TRAIN_SEEDS,
        "eval_seeds": EVAL_SEEDS,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "feature_names": list(FEATURE_NAMES),
        "train_loss_last": history[-1]["train_loss"] if history else 0.0,
        "train_reliability_mae": float(np.mean(np.abs(pred_train_np - y_train))),
        "eval_reliability_mae": float(np.mean(np.abs(pred_eval_np - y_eval))),
    }
    save_model(model_path, model=model, cfg=cfg, metadata=metadata)

    eval_frame = eval_frame.copy()
    eval_frame["predicted_w"] = pred_eval_np
    eval_frame["predicted_fault_prob"] = fault_prob_eval
    train_frame = train_frame.copy()
    train_frame["predicted_w"] = pred_train_np
    train_frame["predicted_fault_prob"] = fault_prob_train

    return {
        "model_path": str(model_path),
        "history": history,
        "train_frame": train_frame,
        "eval_frame": eval_frame,
        "metadata": metadata,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_deep_oqe_artifacts(
    result: dict[str, Any],
    out_dir: Path,
    *,
    paper_table_dir: Path,
    paper_fig_dir: Path,
) -> dict[str, Any]:
    eval_frame = result["eval_frame"]
    low_threshold = 0.35
    low_rate_fault = float((eval_frame.loc[eval_frame["fault_present"] == 1, "predicted_w"] < low_threshold).mean())
    low_rate_clean = float((eval_frame.loc[eval_frame["fault_present"] == 0, "predicted_w"] < low_threshold).mean())
    lift = low_rate_fault / max(low_rate_clean, 1e-6)

    bucket_rows: list[dict[str, Any]] = []
    eval_frame = eval_frame.copy()
    eval_frame["bucket"] = pd.cut(
        eval_frame["predicted_w"],
        bins=[0.0, 0.40, 0.70, 1.01],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    for bucket, sub in eval_frame.groupby("bucket", observed=True):
        bucket_rows.append(
            {
                "bucket": str(bucket),
                "count": int(len(sub)),
                "predicted_w_mean": float(sub["predicted_w"].mean()),
                "target_w_mean": float(sub["target_w"].mean()),
                "mae": float(np.mean(np.abs(sub["predicted_w"] - sub["target_w"]))),
                "fault_rate": float(sub["fault_present"].mean()),
            }
        )

    auc = float(roc_auc_score(eval_frame["fault_present"], eval_frame["predicted_fault_prob"]))
    summary = {
        "model_path": result["model_path"],
        "eval_reliability_mae": float(result["metadata"]["eval_reliability_mae"]),
        "heuristic_eval_reliability_mae": float(np.mean(np.abs(eval_frame["heuristic_w"] - eval_frame["target_w"]))),
        "fault_auc": auc,
        "low_reliability_fault_lift": float(lift),
        "low_reliability_threshold": low_threshold,
        "train_scenarios": TRAIN_SCENARIOS,
        "eval_scenarios": EVAL_SCENARIOS,
    }

    csv_path = out_dir / "battery_deep_oqe_summary.csv"
    json_path = out_dir / "battery_deep_oqe_summary.json"
    md_path = out_dir / "battery_deep_oqe_summary.md"
    bucket_csv = out_dir / "battery_deep_oqe_buckets.csv"
    fig_path = out_dir / "fig_battery_deep_oqe_summary.png"

    _write_csv(
        csv_path,
        [
            {
                "metric": "eval_reliability_mae",
                "heuristic": summary["heuristic_eval_reliability_mae"],
                "deep": summary["eval_reliability_mae"],
            },
            {
                "metric": "fault_auc",
                "heuristic": "",
                "deep": summary["fault_auc"],
            },
            {
                "metric": "low_reliability_fault_lift",
                "heuristic": "",
                "deep": summary["low_reliability_fault_lift"],
            },
        ],
        ["metric", "heuristic", "deep"],
    )
    _write_csv(
        bucket_csv,
        bucket_rows,
        ["bucket", "count", "predicted_w_mean", "target_w_mean", "mae", "fault_rate"],
    )
    json_path.write_text(json.dumps({"summary": summary, "buckets": bucket_rows}, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        dedent(
            f"""\
            # Battery DeepOQE Summary

            - Model: `{summary['model_path']}`
            - Eval reliability MAE: `{summary['eval_reliability_mae']:.4f}`
            - Heuristic reliability MAE: `{summary['heuristic_eval_reliability_mae']:.4f}`
            - Fault AUC: `{summary['fault_auc']:.4f}`
            - Low-reliability fault lift: `{summary['low_reliability_fault_lift']:.4f}`
            """
        ),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Heuristic", "DeepOQE"], [summary["heuristic_eval_reliability_mae"], summary["eval_reliability_mae"]], color=["#a6761d", "#1b9e77"])
    axes[0].set_title("Reliability MAE")
    axes[0].set_ylabel("MAE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(["Fault AUC", "Fault Lift"], [summary["fault_auc"], summary["low_reliability_fault_lift"]], color=["#7570b3", "#d95f02"])
    axes[1].set_title("Fault Sensitivity")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    paper_table_dir.mkdir(parents=True, exist_ok=True)
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    table_tex = paper_table_dir / "tbl_battery_deep_oqe_summary.tex"
    table_tex.write_text(
        "\n".join(
            [
                r"\begin{table}[ht]",
                r"\centering",
                r"\caption{Battery DeepOQE diagnostics on the held-out degraded-telemetry episodes.}",
                r"\label{tab:battery-deep-oqe-summary}",
                r"\begin{tabular}{lrr}",
                r"\toprule",
                r"Metric & Heuristic & DeepOQE \\",
                r"\midrule",
                f"Reliability MAE & {summary['heuristic_eval_reliability_mae']:.4f} & {summary['eval_reliability_mae']:.4f} \\\\",
                f"Fault AUC & --- & {summary['fault_auc']:.4f} \\\\",
                f"Low-reliability fault lift & --- & {summary['low_reliability_fault_lift']:.3f} \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    shutil.copy2(fig_path, paper_fig_dir / "fig_battery_deep_oqe_summary.png")
    return {
        "summary": summary,
        "summary_csv": str(csv_path),
        "bucket_csv": str(bucket_csv),
        "figure": str(fig_path),
        "table_tex": str(table_tex),
    }


def _bench_records_for_controller(
    *,
    buffers: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> list[StepRecord]:
    n = len(buffers["soc_true_mwh"])
    max_power = max(float(constraints.get("max_power_mw", 1.0)), 1e-6)
    min_soc = float(constraints.get("min_soc_mwh", 0.0))
    max_soc = float(constraints.get("max_soc_mwh", 1.0))
    records: list[StepRecord] = []
    for step in range(n):
        proposed_charge = float(buffers["proposed_charge_mw"][step])
        proposed_discharge = float(buffers["proposed_discharge_mw"][step])
        safe_charge = float(buffers["safe_charge_mw"][step])
        safe_discharge = float(buffers["safe_discharge_mw"][step])
        true_soc = float(buffers["soc_true_mwh"][step])
        observed_soc = float(buffers["soc_observed_mwh"][step])
        cert = buffers["certificates"][step]
        cert_present = isinstance(cert, Mapping)
        intervention = abs(proposed_charge - safe_charge) > 1e-6 or abs(proposed_discharge - safe_discharge) > 1e-6
        predicted_valid = cert_present and bool(buffers["guarantee_checks_passed"][step] > 0.5)
        true_valid = not (true_soc < min_soc - 1e-9 or true_soc > max_soc + 1e-9)
        proposed_next_obs_soc = observed_soc + float(constraints.get("time_step_hours", 1.0)) * (
            float(constraints.get("charge_efficiency", 1.0)) * proposed_charge
            - proposed_discharge / max(float(constraints.get("discharge_efficiency", 1.0)), 1e-6)
        )
        true_margin = min(true_soc - min_soc, max_soc - true_soc)
        observed_margin = min(proposed_next_obs_soc - min_soc, max_soc - proposed_next_obs_soc)
        observed_safe = bool(min_soc - 1e-9 <= proposed_next_obs_soc <= max_soc + 1e-9)
        useful_work = max(
            0.0,
            1.0 - (abs(safe_charge - proposed_charge) + abs(safe_discharge - proposed_discharge)) / max_power,
        )
        records.append(
            StepRecord(
                step=step,
                true_state={"soc_mwh": true_soc},
                observed_state={"soc_mwh": observed_soc, "reliability_w": float(buffers["w_t"][step])},
                action={"charge_mw": safe_charge, "discharge_mw": safe_discharge},
                true_constraint_violated=not true_valid,
                observed_constraint_satisfied=observed_safe,
                true_margin=float(true_margin),
                observed_margin=float(observed_margin),
                intervened=intervention,
                fallback_used=intervention and (abs(safe_charge) + abs(safe_discharge) < abs(proposed_charge) + abs(proposed_discharge)),
                certificate_valid=true_valid,
                certificate_predicted_valid=predicted_valid,
                useful_work=useful_work,
                audit_fields_present=4 if cert_present else 0,
                audit_fields_required=4,
                domain_metrics={"reliability_w": float(buffers["w_t"][step])},
            )
        )
    return records


def _battery_dispatch_regime(charge_mw: float, discharge_mw: float) -> str:
    if charge_mw > discharge_mw + 1e-6:
        return "charge"
    if discharge_mw > charge_mw + 1e-6:
        return "discharge"
    return "hold"


def _battery_runtime_artifacts_for_controller(
    *,
    lane: str,
    scenario: str,
    seed: int,
    controller: str,
    buffers: Mapping[str, Any],
    constraints: Mapping[str, Any],
    previous_certificate_hash: str | None = None,
) -> tuple[list[StepRecord], list[dict[str, Any]], list[dict[str, Any]], str | None]:
    records = _bench_records_for_controller(buffers=buffers, constraints=constraints)
    trace_rows: list[dict[str, Any]] = []
    cert_rows: list[dict[str, Any]] = []
    stored_prev_hash = previous_certificate_hash
    dt_hours = float(constraints.get("time_step_hours", 1.0))
    charge_eff = float(constraints.get("charge_efficiency", 1.0))
    discharge_eff = max(float(constraints.get("discharge_efficiency", 1.0)), 1e-6)
    min_soc = float(constraints.get("min_soc_mwh", 0.0))
    max_soc = float(constraints.get("max_soc_mwh", 1.0))
    controller_label = f"{lane}:{controller}"

    for step, record in enumerate(records):
        cert = buffers["certificates"][step]
        cert_payload = dict(cert) if isinstance(cert, Mapping) else {}
        proposed_charge = float(buffers["proposed_charge_mw"][step])
        proposed_discharge = float(buffers["proposed_discharge_mw"][step])
        safe_charge = float(buffers["safe_charge_mw"][step])
        safe_discharge = float(buffers["safe_discharge_mw"][step])
        true_soc = float(buffers["soc_true_mwh"][step])
        observed_soc = float(buffers["soc_observed_mwh"][step])
        proposed_next_obs_soc = observed_soc + dt_hours * (
            charge_eff * proposed_charge - proposed_discharge / discharge_eff
        )
        interval_lower = float(buffers["interval_lower"][step])
        interval_upper = float(buffers["interval_upper"][step])
        reliability_w = float(buffers["w_t"][step])
        drift_score = float(buffers["rac_sensitivity_norm"][step]) if "rac_sensitivity_norm" in buffers else 0.0
        cert_payload.setdefault("scenario_id", scenario)
        cert_payload.setdefault("fault_family", scenario)
        cert_payload.setdefault("lane", lane)
        cert_payload.setdefault("controller_label", controller_label)
        cert_payload.setdefault("dispatch_regime", _battery_dispatch_regime(safe_charge, safe_discharge))
        cert_payload.setdefault("true_margin", float(record.true_margin if record.true_margin is not None else 0.0))
        cert_payload.setdefault("observed_margin", float(record.observed_margin if record.observed_margin is not None else 0.0))
        if cert_payload and lane == "deep" and controller == "dc3s_wrapped":
            cert_payload["prev_hash"] = stored_prev_hash
            cert_payload["certificate_hash"] = recompute_certificate_hash(cert_payload)
            stored_prev_hash = str(cert_payload["certificate_hash"])
            cert_rows.append(cert_payload)
        inflation_values = buffers.get("rac_inflation")
        default_inflation = 1.0
        if isinstance(inflation_values, Sequence) and len(inflation_values) > step:
            default_inflation = float(inflation_values[step])
        trace_rows.append(
            {
                "trace_id": f"{lane}:{scenario}:{seed}:{controller}:{step}",
                "lane": lane,
                "controller": controller,
                "controller_label": controller_label,
                "fault_family": scenario,
                "seed": int(seed),
                "step_index": int(step),
                "reliability_w": reliability_w,
                "widening_factor": float(cert_payload.get("inflation", default_inflation)),
                "drift_score": drift_score,
                "dispatch_regime": _battery_dispatch_regime(safe_charge, safe_discharge),
                "candidate_charge_mw": proposed_charge,
                "candidate_discharge_mw": proposed_discharge,
                "safe_charge_mw": safe_charge,
                "safe_discharge_mw": safe_discharge,
                "true_value": true_soc,
                "observed_value": proposed_next_obs_soc,
                "true_margin": float(record.true_margin if record.true_margin is not None else min(true_soc - min_soc, max_soc - true_soc)),
                "observed_margin": float(record.observed_margin if record.observed_margin is not None else min(proposed_next_obs_soc - min_soc, max_soc - proposed_next_obs_soc)),
                "interval_lower": interval_lower,
                "interval_upper": interval_upper,
                "true_constraint_violated": bool(record.true_constraint_violated),
                "observed_constraint_satisfied": bool(record.observed_constraint_satisfied),
                "intervened": bool(record.intervened),
                "fallback_used": bool(record.fallback_used),
                "certificate_valid": bool(record.certificate_valid),
                "certificate_predicted_valid": bool(record.certificate_predicted_valid),
                "useful_work": float(record.useful_work),
            }
        )
    return records, trace_rows, cert_rows, stored_prev_hash


def build_replay_comparison(
    *,
    model_path: Path,
    out_dir: Path,
    paper_table_dir: Path,
    paper_fig_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    runtime_traces: list[dict[str, Any]] = []
    runtime_summary_rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    previous_certificate_hash: str | None = None
    lanes = {
        "heuristic": {"reliability": {"backend": "heuristic", "min_w": 0.05}},
        "deep": {
            "reliability": {
                "backend": "deep",
                "min_w": 0.05,
                "deep": {"model_path": str(model_path), "seq_len": 8, "strict": True},
            }
        },
    }
    for lane_name, dc3s_overrides in lanes.items():
        for scenario in REPLAY_SCENARIOS:
            for seed in REPLAY_SEEDS:
                payload = run_single(
                    scenario=scenario,
                    seed=seed,
                    horizon=REPLAY_HORIZON,
                    dc3s_overrides=dc3s_overrides,
                    return_controller_buffers=True,
                    controllers_filter=("dc3s_wrapped", "dc3s_ftit"),
                )
                main_rows = payload["main_rows"]
                for controller in ("dc3s_wrapped", "dc3s_ftit"):
                    main_row = next(row for row in main_rows if row["controller"] == controller)
                    records, trace_rows, cert_rows, previous_certificate_hash = _battery_runtime_artifacts_for_controller(
                        lane=lane_name,
                        scenario=scenario,
                        seed=int(seed),
                        controller=controller,
                        buffers=payload["controller_buffers"][controller],
                        constraints=payload["constraints"],
                        previous_certificate_hash=previous_certificate_hash,
                    )
                    bench = compute_bench_metrics(records)
                    runtime_traces.extend(trace_rows)
                    certificate_rows.extend(cert_rows)
                    controller_label = f"{lane_name}:{controller}"
                    rows.append(
                        {
                            "lane": lane_name,
                            "scenario": scenario,
                            "seed": int(seed),
                            "controller": controller,
                            "true_soc_violation_rate": float(main_row["true_soc_violation_rate"]),
                            "intervention_rate": float(main_row["intervention_rate"]),
                            "certificate_presence_rate": float(main_row["certificate_presence_rate"]),
                            "mean_reliability_w": float(main_row["mean_reliability_w"]),
                            "tsvr": float(bench.tsvr),
                            "oasg": float(bench.oasg),
                            "cva": float(bench.cva),
                            "gdq": float(bench.gdq),
                            "audit_completeness": float(bench.audit_completeness),
                            "recovery_latency": float(bench.recovery_latency),
                        }
                    )
                    runtime_summary_rows.append(
                        {
                            "controller": controller_label,
                            "tsvr": float(bench.tsvr),
                            "oasg": float(bench.oasg),
                            "cva": float(bench.cva),
                            "gdq": float(bench.gdq),
                            "intervention_rate": float(bench.intervention_rate),
                            "audit_completeness": float(bench.audit_completeness),
                            "recovery_latency": float(bench.recovery_latency),
                            "n_steps": int(bench.n_steps),
                        }
                    )

    csv_path = out_dir / "battery_deep_oqe_safety_metrics.csv"
    md_path = out_dir / "battery_deep_oqe_safety_metrics.md"
    fig_path = out_dir / "fig_battery_deep_oqe_safety_metrics.png"
    runtime_summary_path = out_dir / "runtime_summary.csv"
    runtime_traces_path = out_dir / "runtime_traces.csv"
    fault_coverage_path = out_dir / "fault_family_coverage.csv"
    audit_db_path = out_dir / "battery_runtime.duckdb"
    _write_csv(
        csv_path,
        rows,
        [
            "lane",
            "scenario",
            "seed",
            "controller",
            "true_soc_violation_rate",
            "intervention_rate",
            "certificate_presence_rate",
            "mean_reliability_w",
            "tsvr",
            "oasg",
            "cva",
            "gdq",
            "audit_completeness",
            "recovery_latency",
        ],
    )

    summary_rows = (
        pd.DataFrame(rows)
        .groupby(["lane", "controller"], as_index=False)[
            [
                "true_soc_violation_rate",
                "intervention_rate",
                "tsvr",
                "oasg",
                "cva",
                "gdq",
                "audit_completeness",
                "recovery_latency",
            ]
        ]
        .mean(numeric_only=True)
    )
    runtime_summary_df = (
        pd.DataFrame(runtime_summary_rows)
        .groupby("controller", as_index=False)
        .agg(
            {
                "tsvr": "mean",
                "oasg": "mean",
                "cva": "mean",
                "gdq": "mean",
                "intervention_rate": "mean",
                "audit_completeness": "mean",
                "recovery_latency": "mean",
                "n_steps": "sum",
            }
        )
    )
    runtime_summary_df.to_csv(runtime_summary_path, index=False)
    trace_df = pd.DataFrame(runtime_traces)
    trace_df.to_csv(runtime_traces_path, index=False)
    coverage_rows = []
    if not trace_df.empty:
        for (controller_label, fault_family), group in trace_df.groupby(["controller_label", "fault_family"], dropna=False):
            y_true = group["true_value"].to_numpy(dtype=float)
            lower = group["interval_lower"].to_numpy(dtype=float)
            upper = group["interval_upper"].to_numpy(dtype=float)
            coverage_rows.append(
                {
                    "controller": str(controller_label),
                    "fault_family": str(fault_family),
                    "target": "soc_mwh",
                    "coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
                    "mean_width": float(np.mean(upper - lower)),
                }
            )
    pd.DataFrame(coverage_rows, columns=["controller", "fault_family", "target", "coverage", "mean_width"]).to_csv(
        fault_coverage_path,
        index=False,
    )
    if audit_db_path.exists():
        audit_db_path.unlink()
    if certificate_rows:
        store_certificates_batch(certificate_rows, duckdb_path=str(audit_db_path), table_name="dispatch_certificates")
    md_lines = ["# Battery DeepOQE Safety Metrics", ""]
    for _, row in summary_rows.iterrows():
        md_lines.append(
            f"- `{row['lane']}` / `{row['controller']}`: TSVR `{row['tsvr']:.4f}`, IR `{row['intervention_rate']:.4f}`, GDQ `{row['gdq']:.4f}`, CVA `{row['cva']:.4f}`"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pivot_ir = summary_rows.pivot(index="controller", columns="lane", values="intervention_rate").fillna(0.0)
    pivot_gdq = summary_rows.pivot(index="controller", columns="lane", values="gdq").fillna(0.0)
    pivot_ir.plot(kind="bar", ax=axes[0], color=["#a6761d", "#1b9e77"])
    axes[0].set_title("Intervention rate")
    axes[0].grid(axis="y", alpha=0.25)
    pivot_gdq.plot(kind="bar", ax=axes[1], color=["#a6761d", "#1b9e77"])
    axes[1].set_title("GDQ")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    paper_table_dir.mkdir(parents=True, exist_ok=True)
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    table_tex = paper_table_dir / "tbl_battery_deep_oqe_safety_metrics.tex"
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Battery replay comparison for heuristic versus DeepOQE reliability backends.}",
        r"\label{tab:battery-deep-oqe-safety}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Lane & Controller & TSVR & IR & GDQ & CVA \\",
        r"\midrule",
    ]
    for _, row in summary_rows.iterrows():
        latex_lines.append(
            f"{_latex_escape(row['lane'])} & {_latex_escape(row['controller'])} & {row['tsvr']:.4f} & {row['intervention_rate']:.4f} & {row['gdq']:.4f} & {row['cva']:.4f} \\\\"
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    table_tex.write_text("\n".join(latex_lines), encoding="utf-8")
    shutil.copy2(fig_path, paper_fig_dir / "fig_battery_deep_oqe_safety_metrics.png")
    return {
        "rows": rows,
        "summary": summary_rows.to_dict(orient="records"),
        "csv_path": str(csv_path),
        "figure_path": str(fig_path),
        "table_tex": str(table_tex),
        "runtime_summary_csv": str(runtime_summary_path),
        "runtime_traces_csv": str(runtime_traces_path),
        "fault_family_coverage_csv": str(fault_coverage_path),
        "audit_db_path": str(audit_db_path),
    }


def _inject_raw_track_faults(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    n = len(out)
    fault_dropout = np.zeros(n, dtype=int)
    fault_stale = np.zeros(n, dtype=int)
    fault_spikes = np.zeros(n, dtype=int)
    fault_delay = np.zeros(n, dtype=int)
    fault_ooo = np.zeros(n, dtype=int)

    for idx in range(96, n, 241):
        fault_dropout[idx] = 1
        for col in ("load_mw", "wind_mw", "solar_mw"):
            out.loc[idx, col] = np.nan

    for start in range(128, n, 360):
        end = min(start + 3, n)
        if start == 0:
            continue
        for col in ("load_mw", "wind_mw", "solar_mw"):
            out.loc[start:end - 1, col] = float(out.loc[start - 1, col])
        fault_stale[start:end] = 1

    for idx in range(72, n, 187):
        scale = 1.15 if (idx // 187) % 2 == 0 else 0.82
        out.loc[idx, "wind_mw"] = float(out.loc[idx, "wind_mw"]) * scale
        out.loc[idx, "solar_mw"] = max(0.0, float(out.loc[idx, "solar_mw"]) * (2.0 - scale))
        fault_spikes[idx] = 1

    for idx in range(48, n, 211):
        fault_delay[idx] = 1
    for idx in range(60, n, 317):
        fault_ooo[idx] = 1

    out[["load_mw", "wind_mw", "solar_mw"]] = out[["load_mw", "wind_mw", "solar_mw"]].ffill().bfill()
    out["fault_dropout"] = fault_dropout
    out["fault_stale_sensor"] = fault_stale
    out["fault_spikes"] = fault_spikes
    out["fault_delay_jitter"] = fault_delay
    out["fault_out_of_order"] = fault_ooo
    return out


def _attach_deep_reliability(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    out = frame.copy()
    adaptive_state: dict[str, Any] = {}
    prev_event: Mapping[str, Any] | None = None
    reliability: list[float] = []
    for idx, row in out.iterrows():
        ts = pd.to_datetime(row["timestamp"], utc=True)
        if int(row["fault_delay_jitter"]) > 0:
            ts = ts + pd.Timedelta(seconds=3600)
        event = {
            "ts_utc": ts.isoformat(),
            "load_mw": float(row["load_mw"]),
            "renewables_mw": float(row["wind_mw"] + row["solar_mw"]),
            "dropout": bool(row["fault_dropout"]),
            "stale_sensor": bool(row["fault_stale_sensor"]),
            "delay_jitter": bool(row["fault_delay_jitter"]),
            "out_of_order": bool(row["fault_out_of_order"]),
            "spikes": bool(row["fault_spikes"]),
        }
        w_t, _flags = compute_reliability(
            event=event,
            last_event=prev_event,
            expected_cadence_s=3600.0,
            reliability_cfg={
                "backend": "deep",
                "min_w": 0.05,
                "deep": {"model_path": str(model_path), "seq_len": 8, "strict": True},
            },
            adaptive_state=adaptive_state,
            ftit_cfg={"stale_k": 3, "stale_tol": 1.0e-9},
        )
        reliability.append(float(w_t))
        prev_event = event
    out["reliability_w"] = reliability
    return out


def _forecast_windows(
    frame: pd.DataFrame,
    *,
    target: str,
    lookback: int,
    horizon: int,
    stride: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        start_idx = 0
        end_idx = int(len(frame) * 0.75)
    else:
        start_idx = int(len(frame) * 0.75) - lookback
        end_idx = len(frame) - horizon
    values = frame[RAW_TRACK_INPUTS].to_numpy(dtype=np.float32)
    target_values = frame[target].to_numpy(dtype=np.float32)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for idx in range(max(0, start_idx), max(start_idx, end_idx), max(1, stride)):
        end = idx + lookback
        future = end + horizon
        if future > len(frame):
            break
        xs.append(values[idx:end])
        ys.append(target_values[end:future])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float], sample_weight: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for idx, quantile in enumerate(quantiles):
        errors = target - pred[:, :, idx]
        q = float(quantile)
        pinball = torch.maximum(q * errors, (q - 1.0) * errors)
        losses.append((pinball.mean(dim=1) * sample_weight).mean())
    return torch.stack(losses).mean()


def run_raw_sequence_track(
    *,
    features_path: Path,
    model_path: Path,
    out_dir: Path,
    paper_table_dir: Path,
    paper_fig_dir: Path,
    epochs: int,
    batch_size: int,
    lookback: int,
    horizon: int,
    train_stride: int,
    eval_stride: int,
    engineered_baseline_path: Path,
) -> dict[str, Any]:
    base = pd.read_parquet(features_path).sort_values("timestamp").reset_index(drop=True)
    frame = base[
        [
            "timestamp",
            "load_mw",
            "wind_mw",
            "solar_mw",
            "price_eur_mwh",
            "carbon_kg_per_mwh",
            "hour",
            "dayofweek",
            "is_weekend",
            "is_daylight",
        ]
    ].copy()
    frame = _inject_raw_track_faults(frame)
    frame = _attach_deep_reliability(frame, model_path)

    baseline_payload = json.loads(engineered_baseline_path.read_text(encoding="utf-8"))
    comparison_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []

    for target in FORECAST_TARGETS:
        x_train, y_train = _forecast_windows(frame, target=target, lookback=lookback, horizon=horizon, stride=train_stride, split="train")
        x_test, y_test = _forecast_windows(frame, target=target, lookback=lookback, horizon=horizon, stride=eval_stride, split="test")
        if len(x_train) == 0 or len(x_test) == 0:
            continue

        feature_mean = x_train.mean(axis=(0, 1), keepdims=True)
        feature_std = x_train.std(axis=(0, 1), keepdims=True)
        feature_std[feature_std < 1e-6] = 1.0
        target_mean = y_train.mean()
        target_std = y_train.std()
        if target_std < 1e-6:
            target_std = 1.0

        x_train_n = (x_train - feature_mean) / feature_std
        x_test_n = (x_test - feature_mean) / feature_std
        y_train_n = (y_train - target_mean) / target_std
        y_test_n = (y_test - target_mean) / target_std

        model = PatchTSTForecaster(
            n_features=x_train_n.shape[-1],
            lookback=lookback,
            horizon=horizon,
            patch_len=24,
            stride=12,
            d_model=64,
            n_heads=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            n_quantiles=3,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train_n)),
            batch_size=batch_size,
            shuffle=True,
        )
        reliability_idx = RAW_TRACK_INPUTS.index("reliability_w")

        for _epoch in range(max(1, epochs)):
            model.train()
            for xb, yb in train_loader:
                pred = model(xb)
                weight = 1.0 + 0.5 * (1.0 - torch.clamp(xb[:, -1, reliability_idx], -3.0, 3.0))
                loss = _quantile_loss(pred, yb, [0.1, 0.5, 0.9], weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_test = model(torch.from_numpy(x_test_n)).detach().cpu().numpy()

        q10 = pred_test[:, :, 0] * target_std + target_mean
        q50 = pred_test[:, :, 1] * target_std + target_mean
        q90 = pred_test[:, :, 2] * target_std + target_mean
        metrics = compute_forecast_metrics(
            y_true=y_test.reshape(-1),
            y_pred=q50.reshape(-1),
            lower_90=q10.reshape(-1),
            upper_90=q90.reshape(-1),
        )
        comparison_rows.append(
            {
                "target": target,
                "track": "raw_sequence",
                "model": "patchtst_reliability_conditioned",
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "picp_90": float(metrics["picp_90"]),
                "mean_interval_width": float(metrics["mean_interval_width"]),
            }
        )

        engineered = baseline_payload["targets"].get(target, {}).get("gbm", {})
        if engineered:
            uq = engineered.get("uncertainty", {})
            comparison_rows.append(
                {
                    "target": target,
                    "track": "engineered_tabular",
                    "model": "gbm",
                    "rmse": float(engineered.get("rmse", np.nan)),
                    "mae": float(engineered.get("mae", np.nan)),
                    "picp_90": float(uq.get("picp_90", np.nan)),
                    "mean_interval_width": float(uq.get("mean_interval_width", np.nan)),
                }
            )

        y_true_flat = y_test.reshape(-1)
        q10_flat = q10.reshape(-1)
        q90_flat = q90.reshape(-1)
        if target == "wind_mw":
            vol = pd.Series(y_true_flat).rolling(window=24, min_periods=6).std().fillna(0.0).to_numpy()
            cutoff = float(np.quantile(vol, 0.66))
            labels = np.where(vol >= cutoff, "gusty", "calm")
        elif target == "solar_mw":
            labels = np.where(y_true_flat > 10.0, "daytime", "nighttime")
        else:
            reliability_flat = np.repeat(frame["reliability_w"].to_numpy(dtype=float)[lookback + horizon - 1 :: eval_stride][: len(x_test)], horizon)
            labels = np.where(reliability_flat < 0.6, "low_reliability", "high_reliability")
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            if not np.any(mask):
                continue
            slice_rows.append(
                {
                    "target": target,
                    "slice": label,
                    "picp_90": float(np.mean((y_true_flat[mask] >= q10_flat[mask]) & (y_true_flat[mask] <= q90_flat[mask]))),
                    "mean_interval_width": float(np.mean(q90_flat[mask] - q10_flat[mask])),
                    "n": int(mask.sum()),
                }
            )

    comparison_csv = out_dir / "battery_raw_sequence_track_benchmark.csv"
    slice_csv = out_dir / "battery_raw_sequence_track_slices.csv"
    md_path = out_dir / "battery_raw_sequence_track_benchmark.md"
    fig_path = out_dir / "fig_battery_raw_sequence_track_benchmark.png"
    _write_csv(
        comparison_csv,
        comparison_rows,
        ["target", "track", "model", "rmse", "mae", "picp_90", "mean_interval_width"],
    )
    _write_csv(slice_csv, slice_rows, ["target", "slice", "picp_90", "mean_interval_width", "n"])
    md_lines = ["# Battery Raw-Sequence Track Benchmark", ""]
    for row in comparison_rows:
        md_lines.append(
            f"- `{row['target']}` / `{row['track']}` / `{row['model']}`: RMSE `{row['rmse']:.2f}`, PICP90 `{row['picp_90']:.3f}`, width `{row['mean_interval_width']:.2f}`"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    comp_df = pd.DataFrame(comparison_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pivot_picp = comp_df.pivot(index="target", columns="track", values="picp_90").fillna(0.0)
    pivot_width = comp_df.pivot(index="target", columns="track", values="mean_interval_width").fillna(0.0)
    pivot_picp.plot(kind="bar", ax=axes[0], color=["#7570b3", "#1b9e77"])
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("PICP@90 by track")
    axes[0].grid(axis="y", alpha=0.25)
    pivot_width.plot(kind="bar", ax=axes[1], color=["#7570b3", "#1b9e77"])
    axes[1].set_title("Mean interval width by track")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    paper_table_dir.mkdir(parents=True, exist_ok=True)
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    table_tex = paper_table_dir / "tbl_battery_raw_sequence_track.tex"
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Engineered versus raw-sequence battery forecasting tracks. The raw-sequence row is the bounded deep-learning novelty track and is not a replacement for the locked engineered baseline.}",
        r"\label{tab:battery-raw-sequence-track}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Target & Track & Model & RMSE & PICP$_{90}$ & Width \\",
        r"\midrule",
    ]
    for row in comparison_rows:
        latex_lines.append(
            f"{_latex_escape(row['target'])} & {_latex_escape(row['track'])} & {_latex_escape(row['model'])} & {row['rmse']:.2f} & {row['picp_90']:.3f} & {row['mean_interval_width']:.1f} \\\\"
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    table_tex.write_text("\n".join(latex_lines), encoding="utf-8")
    shutil.copy2(fig_path, paper_fig_dir / "fig_battery_raw_sequence_track_benchmark.png")
    return {
        "comparison_rows": comparison_rows,
        "slice_rows": slice_rows,
        "comparison_csv": str(comparison_csv),
        "slice_csv": str(slice_csv),
        "figure_path": str(fig_path),
        "table_tex": str(table_tex),
    }


def write_register(
    *,
    out_dir: Path,
    deep_oqe: dict[str, Any],
    replay: dict[str, Any],
    raw_track: dict[str, Any],
) -> None:
    register = {
        "program": "orius_deep_learning_novelty_battery_v1",
        "scope": "bounded_battery_first",
        "artifacts": {
            "deep_oqe": deep_oqe,
            "replay": replay,
            "raw_track": raw_track,
        },
        "notes": [
            "This artifact family strengthens the battery-scoped deep-learning novelty surface without mutating the official equal-domain parity matrix.",
            "The engineered-tabular GBM baseline remains the canonical systems baseline.",
            "The raw-sequence PatchTST lane is a bounded novelty surface, not a replacement for the locked engineered-track benchmark.",
        ],
    }
    (out_dir / "battery_deep_learning_novelty_register.json").write_text(json.dumps(register, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "battery_deep_learning_novelty_register.md").write_text(
        dedent(
            f"""\
            # Battery Deep-Learning Novelty Register

            - Scope: `bounded_battery_first`
            - DeepOQE summary CSV: `{deep_oqe['summary_csv']}`
            - Replay comparison CSV: `{replay['csv_path']}`
            - Raw-track comparison CSV: `{raw_track['comparison_csv']}`
            """
        ),
        encoding="utf-8",
    )


def run_pipeline(
    *,
    out_dir: Path,
    model_dir: Path,
    paper_table_dir: Path,
    paper_fig_dir: Path,
    features_path: Path,
    engineered_baseline_path: Path,
    deep_oqe_epochs: int,
    forecast_epochs: int,
    batch_size: int,
    seq_len: int,
    lookback: int,
    horizon: int,
    train_stride: int,
    eval_stride: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    paper_fig_dir.mkdir(parents=True, exist_ok=True)

    deep_frame = build_deep_oqe_training_frame(seq_len=seq_len)
    model_path = model_dir / "battery_deepoqe.pt"
    deep_result = train_deep_oqe(
        deep_frame,
        epochs=deep_oqe_epochs,
        batch_size=batch_size,
        seq_len=seq_len,
        model_path=model_path,
    )
    deep_artifacts = write_deep_oqe_artifacts(
        deep_result,
        out_dir,
        paper_table_dir=paper_table_dir,
        paper_fig_dir=paper_fig_dir,
    )
    replay_artifacts = build_replay_comparison(
        model_path=model_path,
        out_dir=out_dir,
        paper_table_dir=paper_table_dir,
        paper_fig_dir=paper_fig_dir,
    )
    raw_track_artifacts = run_raw_sequence_track(
        features_path=features_path,
        model_path=model_path,
        out_dir=out_dir,
        paper_table_dir=paper_table_dir,
        paper_fig_dir=paper_fig_dir,
        epochs=forecast_epochs,
        batch_size=batch_size,
        lookback=lookback,
        horizon=horizon,
        train_stride=train_stride,
        eval_stride=eval_stride,
        engineered_baseline_path=engineered_baseline_path,
    )
    write_register(out_dir=out_dir, deep_oqe=deep_artifacts, replay=replay_artifacts, raw_track=raw_track_artifacts)
    return {
        "deep_oqe": deep_artifacts,
        "replay": replay_artifacts,
        "raw_track": raw_track_artifacts,
        "register_json": str(out_dir / "battery_deep_learning_novelty_register.json"),
        "register_md": str(out_dir / "battery_deep_learning_novelty_register.md"),
        "model_path": str(model_path),
        "out_dir": str(out_dir),
        "paper_table_dir": str(paper_table_dir),
        "paper_fig_dir": str(paper_fig_dir),
    }


def main() -> None:
    args = _parse_args()
    result = run_pipeline(
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        paper_table_dir=args.paper_table_dir,
        paper_fig_dir=args.paper_fig_dir,
        features_path=args.features,
        engineered_baseline_path=args.engineered_baseline,
        deep_oqe_epochs=args.deep_oqe_epochs,
        forecast_epochs=args.forecast_epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
