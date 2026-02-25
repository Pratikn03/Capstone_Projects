"""CLI and orchestration for CPSBench-IoT runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gridpulse.dc3s.guarantee_checks import evaluate_guarantee_checks

from .baselines import (
    dc3s_wrapped_step,
    deterministic_lp_step,
    init_dc3s_loop_state,
    naive_safe_clip_step,
    robust_fixed_interval_step,
)
from .metrics import compute_all_metrics
from .plant import BatteryPlant
from .scenarios import DEFAULT_SCENARIOS, FAULT_COLUMNS, generate_episode
from .telemetry_soc import SOCTelemetryChannel, SOCTelemetryFaultConfig


REQUIRED_OUTPUTS = (
    "dc3s_main_table.csv",
    "dc3s_fault_breakdown.csv",
    "cpsbench_merged_sweep.csv",
    "calibration_plot.png",
    "violation_vs_cost_curve.png",
    "fig_violation_rate.png",
    "fig_violation_severity_p95.png",
    "fig_true_soc_violation_vs_dropout.png",
    "fig_true_soc_severity_p95_vs_dropout.png",
    "dc3s_run_summary.json",
)

FAULT_SWEEP_LEVELS: dict[str, list[float]] = {
    "dropout": [0.0, 0.05, 0.10, 0.20, 0.30],
    "delay_seconds": [0.0, 1.0, 5.0, 15.0],
    "out_of_order": [0.0, 0.05, 0.15],
    "spike_sigma": [0.0, 1.0, 2.0, 3.0],
}


def _deep_update(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _load_optimization_cfg() -> dict[str, Any]:
    for p in ("configs/optimization.yaml", "configs/optimize.yaml"):
        payload = _load_yaml(p)
        if payload:
            return payload
    return {}


def _load_dc3s_cfg() -> dict[str, Any]:
    payload = _load_yaml("configs/dc3s.yaml")
    dc3s = payload.get("dc3s", {}) if isinstance(payload, dict) else {}
    return dc3s if isinstance(dc3s, dict) else {}


def _battery_constraints(cfg: dict[str, Any]) -> dict[str, float]:
    battery = dict(cfg.get("battery", {}))
    capacity = float(battery.get("capacity_mwh", 100.0))
    max_power = float(battery.get("max_power_mw", 50.0))
    eff = float(battery.get("efficiency", battery.get("efficiency_regime_a", 0.95)))
    return {
        "capacity_mwh": capacity,
        "max_power_mw": max_power,
        "max_charge_mw": float(battery.get("max_charge_mw", max_power)),
        "max_discharge_mw": float(battery.get("max_discharge_mw", max_power)),
        "min_soc_mwh": float(battery.get("min_soc_mwh", 0.0)),
        "max_soc_mwh": float(battery.get("max_soc_mwh", capacity)),
        "initial_soc_mwh": float(battery.get("initial_soc_mwh", capacity * 0.5)),
        "charge_efficiency": float(battery.get("charge_efficiency", eff)),
        "discharge_efficiency": float(battery.get("discharge_efficiency", eff)),
        "time_step_hours": float(cfg.get("time_step_hours", 1.0)),
        "degradation_cost_per_mwh": float(battery.get("degradation_cost_per_mwh", 10.0)),
        "max_grid_import_mw": float(cfg.get("grid", {}).get("max_import_mw", 500.0)),
    }


def _ensure_out_dir(out_dir: str | Path) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_telemetry_events(x_obs: pd.DataFrame, event_log: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx in range(len(x_obs)):
        payload: dict[str, Any] = {
            "ts_utc": pd.to_datetime(event_log.loc[idx, "arrived_timestamp"], utc=True).isoformat(),
            "device_id": "bench-device",
            "zone_id": "DE",
            "load_mw": float(x_obs.loc[idx, "load_mw"]),
            "renewables_mw": float(x_obs.loc[idx, "renewables_mw"]),
        }
        for fault_col in FAULT_COLUMNS:
            payload[fault_col] = bool(event_log.loc[idx, fault_col])
        events.append(payload)
    return events


def _soc_fault_config_for_episode(
    *,
    scenario: str,
    event_log: pd.DataFrame,
    seed: int,
) -> SOCTelemetryFaultConfig:
    dropout_prob = float(event_log["dropout"].mean()) if "dropout" in event_log.columns else 0.0
    stale_prob = float(event_log["stale_sensor"].mean()) if "stale_sensor" in event_log.columns else 0.0

    # Keep a minimum stress level for canonical telemetry-fault scenarios.
    if scenario in {"dropout", "drift_combo"} and dropout_prob == 0.0:
        dropout_prob = 0.20
    if scenario in {"stale_sensor", "drift_combo"} and stale_prob == 0.0:
        stale_prob = 0.20

    return SOCTelemetryFaultConfig(
        dropout_prob=min(max(dropout_prob, 0.0), 0.95),
        stale_prob=min(max(stale_prob, 0.0), 0.95),
        noise_std_mwh=0.25,
        seed=int(seed),
    )


def _run_controller_episode(
    *,
    controller: str,
    scenario: str,
    seed: int,
    timestamps: pd.Series,
    load_obs: np.ndarray,
    renew_obs: np.ndarray,
    load_true: np.ndarray,
    renew_true: np.ndarray,
    price: np.ndarray,
    carbon: np.ndarray,
    event_log: pd.DataFrame,
    telemetry_events: list[dict[str, Any]],
    optimization_cfg: dict[str, Any],
    dc3s_cfg: dict[str, Any],
) -> dict[str, Any]:
    n = len(load_obs)
    constraints = _battery_constraints(optimization_cfg)
    plant = BatteryPlant(
        soc_mwh=float(constraints["initial_soc_mwh"]),
        min_soc_mwh=float(constraints["min_soc_mwh"]),
        max_soc_mwh=float(constraints["max_soc_mwh"]),
        charge_eff=float(constraints["charge_efficiency"]),
        discharge_eff=float(constraints["discharge_efficiency"]),
        dt_hours=float(constraints["time_step_hours"]),
    )
    soc_channel = SOCTelemetryChannel(
        _soc_fault_config_for_episode(scenario=scenario, event_log=event_log, seed=int(seed))
    )

    proposed_charge = np.zeros(n, dtype=float)
    proposed_discharge = np.zeros(n, dtype=float)
    safe_charge = np.zeros(n, dtype=float)
    safe_discharge = np.zeros(n, dtype=float)
    soc_true = np.zeros(n, dtype=float)
    soc_observed = np.zeros(n, dtype=float)
    interval_lower = np.zeros(n, dtype=float)
    interval_upper = np.zeros(n, dtype=float)
    reliability_w = np.ones(n, dtype=float)
    drift_flag = np.zeros(n, dtype=bool)
    inflation = np.ones(n, dtype=float)
    guarantee_passed = np.ones(n, dtype=bool)
    bms_trip_mask = np.zeros(n, dtype=bool)
    certificates: list[dict[str, Any] | None] = []

    dc3s_state = init_dc3s_loop_state(dc3s_cfg) if controller == "dc3s_wrapped" else None
    q_base = max(50.0, float(np.quantile(np.abs(load_true - load_obs), 0.90)))
    expected_cost_usd = 0.0
    carbon_kg = 0.0

    for t in range(n):
        load_window = load_obs[t:]
        renew_window = renew_obs[t:]
        price_window = price[t:]
        carbon_window = carbon[t:]
        current_true_soc = float(plant.soc_mwh)
        current_observed_soc, _soc_meta = soc_channel.observe(current_true_soc)
        soc_observed[t] = float(current_observed_soc)

        if controller == "deterministic_lp":
            step = deterministic_lp_step(
                load_obs_window=load_window,
                renew_obs_window=renew_window,
                price_window=price_window,
                carbon_window=carbon_window,
                observed_soc_mwh=current_observed_soc,
                optimization_cfg=optimization_cfg,
            )
        elif controller == "robust_fixed_interval":
            step = robust_fixed_interval_step(
                load_obs_window=load_window,
                renew_obs_window=renew_window,
                price_window=price_window,
                observed_soc_mwh=current_observed_soc,
                optimization_cfg=optimization_cfg,
            )
        elif controller == "naive_safe_clip":
            step = naive_safe_clip_step(
                timestamp=pd.to_datetime(timestamps.iloc[t], utc=True),
                load_obs_t=float(load_window[0]),
                renew_obs_t=float(renew_window[0]),
                price_t=float(price_window[0]),
                carbon_t=float(carbon_window[0]),
                observed_soc_mwh=current_observed_soc,
                optimization_cfg=optimization_cfg,
            )
        elif controller == "dc3s_wrapped":
            assert dc3s_state is not None
            step = dc3s_wrapped_step(
                load_obs_window=load_window,
                renew_obs_window=renew_window,
                load_true_t=float(load_true[t]),
                telemetry_event=telemetry_events[t],
                price_window=price_window,
                observed_soc_mwh=current_observed_soc,
                dc3s_state=dc3s_state,
                command_id=f"{scenario}-{seed}-{controller}-{t:04d}",
                optimization_cfg=optimization_cfg,
                dc3s_cfg=dc3s_cfg,
                q_base=q_base,
            )
        else:
            raise ValueError(f"Unknown controller: {controller}")

        proposed = dict(step["proposed_action"])
        safe = dict(step["safe_action"])
        proposed_charge[t] = float(proposed.get("charge_mw", 0.0))
        proposed_discharge[t] = float(proposed.get("discharge_mw", 0.0))
        safe_charge[t] = float(safe.get("charge_mw", 0.0))
        safe_discharge[t] = float(safe.get("discharge_mw", 0.0))
        interval_lower[t] = float(step.get("interval_lower_t", load_window[0]))
        interval_upper[t] = float(step.get("interval_upper_t", load_window[0]))
        certificates.append(step.get("certificate"))
        reliability_w[t] = float(step.get("reliability_w", 1.0))
        drift_flag[t] = bool(step.get("drift_flag", False))
        inflation[t] = float(step.get("inflation", 1.0))
        guarantee_passed[t] = bool(step.get("guarantee_checks_passed", True))

        g_ok, _, _ = evaluate_guarantee_checks(
            current_soc=current_true_soc,
            action={"charge_mw": safe_charge[t], "discharge_mw": safe_discharge[t]},
            constraints=constraints,
        )
        if not g_ok:
            bms_trip_mask[t] = True
            applied_charge = 0.0
            applied_discharge = 0.0
        else:
            applied_charge = safe_charge[t]
            applied_discharge = safe_discharge[t]

        next_true_soc = plant.step(
            charge_mw=applied_charge,
            discharge_mw=applied_discharge,
        )
        soc_true[t] = float(next_true_soc)

        grid_import_true = max(
            0.0,
            float(load_true[t] - renew_true[t] - applied_discharge + applied_charge),
        )
        dt = float(constraints.get("time_step_hours", 1.0))
        deg = float(constraints.get("degradation_cost_per_mwh", 10.0))
        expected_cost_usd += float(price[t]) * grid_import_true * dt
        expected_cost_usd += deg * (abs(applied_charge) + abs(applied_discharge)) * dt
        carbon_kg += float(carbon[t]) * grid_import_true * dt

    return {
        "policy": controller,
        "proposed_charge_mw": proposed_charge,
        "proposed_discharge_mw": proposed_discharge,
        "safe_charge_mw": safe_charge,
        "safe_discharge_mw": safe_discharge,
        "soc_mwh": soc_true,
        "soc_true_mwh": soc_true,
        "soc_observed_mwh": soc_observed,
        "interval_lower": interval_lower,
        "interval_upper": interval_upper,
        "reliability_w": reliability_w,
        "drift_flag": drift_flag.astype(int),
        "inflation": inflation,
        "guarantee_checks_passed": guarantee_passed.astype(int),
        "bms_trip_mask": bms_trip_mask.astype(int),
        "certificates": certificates,
        "constraints": constraints,
        "expected_cost_usd": float(expected_cost_usd),
        "carbon_kg": float(carbon_kg),
    }


def run_single(
    *,
    scenario: str,
    seed: int,
    horizon: int = 168,
    fault_overrides: dict[str, Any] | None = None,
    dc3s_overrides: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run one scenario+seed pair and return metrics row payloads."""
    x_obs, x_true, event_log = generate_episode(
        scenario=scenario,
        seed=seed,
        horizon=horizon,
        fault_overrides=fault_overrides,
    )
    load_obs = x_obs["load_mw"].to_numpy(dtype=float)
    renew_obs = x_obs["renewables_mw"].to_numpy(dtype=float)
    load_true = x_true["load_mw"].to_numpy(dtype=float)
    renew_true = x_true["renewables_mw"].to_numpy(dtype=float)
    price = x_obs["price_per_mwh"].to_numpy(dtype=float)
    carbon = x_obs["carbon_kg_per_mwh"].to_numpy(dtype=float)
    telemetry_events = _to_telemetry_events(x_obs=x_obs, event_log=event_log)

    optimization_cfg = _load_optimization_cfg()
    dc3s_cfg = _load_dc3s_cfg()
    if dc3s_overrides:
        dc3s_cfg = _deep_update(dc3s_cfg, dc3s_overrides)

    controllers = ["deterministic_lp", "robust_fixed_interval", "naive_safe_clip", "dc3s_wrapped"]
    results = {
        controller: _run_controller_episode(
            controller=controller,
            scenario=scenario,
            seed=int(seed),
            timestamps=x_obs["timestamp"],
            load_obs=load_obs,
            renew_obs=renew_obs,
            load_true=load_true,
            renew_true=renew_true,
            price=price,
            carbon=carbon,
            event_log=event_log,
            telemetry_events=telemetry_events,
            optimization_cfg=optimization_cfg,
            dc3s_cfg=dc3s_cfg,
        )
        for controller in controllers
    }

    baseline_cost = float(results["deterministic_lp"]["expected_cost_usd"])
    main_rows: list[dict[str, Any]] = []
    fault_rows: list[dict[str, Any]] = []
    for controller, result in results.items():
        metrics = compute_all_metrics(
            y_true=load_true,
            y_pred=load_obs,
            lower_90=result["interval_lower"],
            upper_90=result["interval_upper"],
            proposed_charge_mw=result["proposed_charge_mw"],
            proposed_discharge_mw=result["proposed_discharge_mw"],
            safe_charge_mw=result["safe_charge_mw"],
            safe_discharge_mw=result["safe_discharge_mw"],
            soc_mwh=result["soc_mwh"],
            constraints=result["constraints"],
            certificates=result["certificates"],
            event_log=event_log,
            true_soc_mwh=result["soc_true_mwh"],
            bms_trip_mask=result["bms_trip_mask"],
            load_true=load_true,
            renewables_true=renew_true,
        )
        cost_delta_pct = None
        if baseline_cost > 0:
            cost_delta_pct = 100.0 * (float(result.get("expected_cost_usd", 0.0)) - baseline_cost) / baseline_cost
        main_rows.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "controller": controller,
                "policy": result["policy"],
                "expected_cost_usd": result.get("expected_cost_usd"),
                "carbon_kg": result.get("carbon_kg"),
                "cost_delta_pct": float(cost_delta_pct) if cost_delta_pct is not None else None,
                "mean_reliability_w": float(np.mean(result.get("reliability_w", np.ones_like(load_true)))),
                "drift_flag_rate": float(np.mean(np.asarray(result.get("drift_flag", np.zeros_like(load_true)), dtype=float))),
                "inflation_p95": float(np.quantile(np.asarray(result.get("inflation", np.ones_like(load_true)), dtype=float), 0.95)),
                "guarantee_checks_passed_rate": float(
                    np.mean(np.asarray(result.get("guarantee_checks_passed", np.ones_like(load_true)), dtype=float))
                ),
                **metrics,
            }
        )

        violations = np.asarray(metrics["true_soc_violation_mask"], dtype=bool)
        interventions = (
            np.abs(np.asarray(result["proposed_charge_mw"]) - np.asarray(result["safe_charge_mw"])) > 1e-6
        ) | (
            np.abs(np.asarray(result["proposed_discharge_mw"]) - np.asarray(result["safe_discharge_mw"])) > 1e-6
        )
        for fault_col in FAULT_COLUMNS:
            mask = event_log[fault_col].to_numpy(dtype=int) > 0
            fault_rows.append(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "controller": controller,
                    "fault_type": fault_col,
                    "fault_count": int(mask.sum()),
                    "violation_rate_at_fault": float(np.mean(violations[mask])) if mask.any() else 0.0,
                    "intervention_rate_at_fault": float(np.mean(interventions[mask])) if mask.any() else 0.0,
                }
            )

    return {"main_rows": main_rows, "fault_rows": fault_rows}


def _plot_calibration(main_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in main_df.groupby("controller", sort=True):
        ax.scatter(sub["picp_90"], sub["mean_interval_width"], label=controller, alpha=0.85)
    ax.axvline(0.90, color="black", linestyle="--", linewidth=1.2, label="Nominal 90%")
    ax.set_xlabel("PICP@90")
    ax.set_ylabel("Mean Interval Width")
    ax.set_title("CPSBench Calibration vs Sharpness")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_violation_vs_cost(main_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in main_df.groupby("controller", sort=True):
        ax.scatter(sub["expected_cost_usd"], sub["true_soc_violation_rate"], label=controller, alpha=0.85)
    ax.set_xlabel("Expected Cost (USD)")
    ax.set_ylabel("True SOC Violation Rate")
    ax.set_title("Violation vs Cost Tradeoff")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_true_soc_curves(main_df: pd.DataFrame, out_violation: Path, out_severity: Path) -> None:
    subset = main_df[main_df["scenario"].isin(["dropout", "drift_combo"])].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    if subset.empty:
        ax.text(0.5, 0.5, "No dropout/drift_combo rows in run", ha="center", va="center")
    else:
        for controller, sub in subset.groupby("controller", sort=True):
            ax.plot(sub["seed"], sub["true_soc_violation_rate"], marker="o", linestyle="-", label=controller)
        ax.legend()
    ax.set_xlabel("Seed")
    ax.set_ylabel("True SOC Violation Rate")
    ax.set_title("True SOC Violation Rate (faulted telemetry)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_violation, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if subset.empty:
        ax.text(0.5, 0.5, "No dropout/drift_combo rows in run", ha="center", va="center")
    else:
        for controller, sub in subset.groupby("controller", sort=True):
            ax.plot(sub["seed"], sub["true_soc_violation_severity_p95"], marker="o", linestyle="-", label=controller)
        ax.legend()
    ax.set_xlabel("Seed")
    ax.set_ylabel("True SOC Violation Severity P95 (MWh)")
    ax.set_title("True SOC Violation Severity (P95)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_severity, dpi=220)
    plt.close(fig)


def _plot_sweep_metric(
    *,
    sweep_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = (
        sweep_df.groupby(["fault_dimension", "severity", "controller"], as_index=False)[metric_col]
        .mean(numeric_only=True)
        .sort_values(["fault_dimension", "severity", "controller"])
    )
    for (fault_dimension, controller), sub in grouped.groupby(["fault_dimension", "controller"], sort=True):
        ax.plot(
            sub["severity"].to_numpy(dtype=float),
            sub[metric_col].to_numpy(dtype=float),
            marker="o",
            label=f"{fault_dimension}:{controller}",
            alpha=0.9,
        )
    ax.set_xlabel("Fault Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_fault_sweep(
    *,
    seeds: Iterable[int],
    horizon: int,
    out_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dim, levels in FAULT_SWEEP_LEVELS.items():
        scenario = "dropout"
        if dim == "delay_seconds":
            scenario = "delay_jitter"
        elif dim == "out_of_order":
            scenario = "out_of_order"
        elif dim == "spike_sigma":
            scenario = "spikes"

        for level in levels:
            overrides: dict[str, Any] = {}
            if dim == "dropout":
                overrides["dropout_rate"] = float(level)
            elif dim == "delay_seconds":
                overrides["delay_seconds"] = float(level)
            elif dim == "out_of_order":
                overrides["out_of_order_rate"] = float(level)
            elif dim == "spike_sigma":
                overrides["spike_sigma"] = float(level)

            for seed in seeds:
                payload = run_single(
                    scenario=scenario,
                    seed=int(seed),
                    horizon=horizon,
                    fault_overrides=overrides,
                )
                for row in payload["main_rows"]:
                    rows.append(
                        {
                            "fault_dimension": dim,
                            "severity": float(level),
                            "scenario": scenario,
                            **row,
                        }
                    )
    sweep_df = pd.DataFrame(rows).sort_values(["fault_dimension", "severity", "scenario", "seed", "controller"]).reset_index(drop=True)
    sweep_df.to_csv(out_dir / "cpsbench_merged_sweep.csv", index=False, float_format="%.6f")
    return sweep_df


def run_suite(
    *,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    out_dir: str | Path = "reports/publication",
    horizon: int = 168,
) -> dict[str, Any]:
    """Run CPSBench suite and persist canonical publication artifacts."""
    out = _ensure_out_dir(out_dir)
    scenarios_list = list(scenarios)
    seeds_list = [int(s) for s in seeds]

    all_main_rows: list[dict[str, Any]] = []
    all_fault_rows: list[dict[str, Any]] = []
    for scenario in scenarios_list:
        for seed in seeds_list:
            payload = run_single(scenario=scenario, seed=int(seed), horizon=horizon)
            all_main_rows.extend(payload["main_rows"])
            all_fault_rows.extend(payload["fault_rows"])

    main_df = pd.DataFrame(all_main_rows).sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
    if "true_soc_violation_mask" in main_df.columns:
        main_df = main_df.drop(columns=["true_soc_violation_mask"])
    fault_df = pd.DataFrame(all_fault_rows).sort_values(["scenario", "seed", "fault_type", "controller"]).reset_index(drop=True)

    main_csv = out / "dc3s_main_table.csv"
    fault_csv = out / "dc3s_fault_breakdown.csv"
    calibration_png = out / "calibration_plot.png"
    violation_png = out / "violation_vs_cost_curve.png"
    summary_json = out / "dc3s_run_summary.json"
    fig_violation_rate = out / "fig_violation_rate.png"
    fig_violation_severity = out / "fig_violation_severity_p95.png"
    fig_true_soc_rate = out / "fig_true_soc_violation_vs_dropout.png"
    fig_true_soc_severity = out / "fig_true_soc_severity_p95_vs_dropout.png"

    main_df.to_csv(main_csv, index=False, float_format="%.6f")
    fault_df.to_csv(fault_csv, index=False, float_format="%.6f")
    _plot_calibration(main_df, calibration_png)
    _plot_violation_vs_cost(main_df, violation_png)
    sweep_df = run_fault_sweep(seeds=seeds_list, horizon=horizon, out_dir=out)
    _plot_sweep_metric(
        sweep_df=sweep_df,
        metric_col="true_soc_violation_rate",
        ylabel="True SOC Violation Rate",
        title="Violation Rate vs Fault Severity",
        out_path=fig_violation_rate,
    )
    _plot_sweep_metric(
        sweep_df=sweep_df,
        metric_col="true_soc_violation_severity_p95",
        ylabel="True SOC Violation Severity P95 (MWh)",
        title="Violation Severity (P95) vs Fault Severity",
        out_path=fig_violation_severity,
    )
    _plot_true_soc_curves(main_df=main_df, out_violation=fig_true_soc_rate, out_severity=fig_true_soc_severity)

    summary = {
        "scenarios": scenarios_list,
        "seeds": seeds_list,
        "horizon": int(horizon),
        "rows_main": int(len(main_df)),
        "rows_fault_breakdown": int(len(fault_df)),
        "rows_sweep": int(len(sweep_df)),
        "controller_summary": {
            controller: {
                "mean_true_soc_violation_rate": float(sub["true_soc_violation_rate"].mean()),
                "mean_intervention_rate": float(sub["intervention_rate"].mean()),
                "mean_cost_usd": float(sub["expected_cost_usd"].dropna().mean()) if not sub["expected_cost_usd"].dropna().empty else None,
            }
            for controller, sub in main_df.groupby("controller", sort=True)
        },
        "artifacts": {name: str(out / name) for name in REQUIRED_OUTPUTS},
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPSBench-IoT benchmark")
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=str, default="reports/publication")
    parser.add_argument("--horizon", type=int, default=168)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_suite(
        scenarios=[args.scenario],
        seeds=[int(args.seed)],
        out_dir=args.out_dir,
        horizon=int(args.horizon),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
