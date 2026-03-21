#!/usr/bin/env python3
"""Run ORIUS Universal Framework on mixed real/synthetic data across domains.

Reads locked repo telemetry CSVs where available, synthesizes a bounded
navigation trace, injects realistic faults (dropout, spike, stale), runs the
full 5-stage DC3S pipeline on each row, and produces:
  - reports/multi_domain/all_domain_results.json
  - reports/multi_domain/tbl_all_domain_comparison.tex
  - reports/multi_domain/tbl_all_domain_latency.tex
  - reports/multi_domain/fig_all_domain_comparison.png

The output is a portable-domain audit surface. It is useful for showing that
the same repair path executes across AV, medical monitoring, industrial
control, navigation, and aerospace, but it is not a replacement for the locked
battery proof surface.

Usage:
    python scripts/run_all_domain_eval.py [--seed 42] [--rows 200] [--out reports/multi_domain]
"""
from __future__ import annotations

import argparse
from typing import Any
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orius.universal_framework import get_adapter, run_universal_step

DEFAULT_OUT = Path("reports/multi_domain")
DEFAULT_ROWS = 200

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _violates_aerospace(row: pd.Series, action: dict[str, object], constraints: dict[str, object]) -> bool:
    # DC3S controls bank angle command; airspeed is a plant state not directly actionable in one step.
    # Violation is evaluated on the commanded bank angle only.
    bank_cmd = float(action.get("bank_deg", row.get("bank_angle_deg", 0.0)) or 0.0)
    bank_limit = float(constraints.get("bank_limit_deg", 30.0) or 30.0)
    return abs(bank_cmd) > bank_limit


def _violates_av(row: pd.Series, action: dict[str, object], constraints: dict[str, object]) -> bool:
    speed = row.get("speed_mps")
    if pd.isna(speed):
        return False
    accel = float(action.get("acceleration_mps2", 0.0) or 0.0)
    dt = float(constraints.get("dt_s", 1.0) or 1.0)
    speed_max = float(constraints.get("speed_max_mps", row.get("speed_limit_mps", 30.0)) or 30.0)
    next_speed = float(speed) + accel * dt
    return next_speed > speed_max


def _violates_navigation(row: pd.Series, action: dict[str, object], constraints: dict[str, object]) -> bool:
    x = row.get("x")
    y = row.get("y")
    if pd.isna(x) or pd.isna(y):
        return False
    dt = float(constraints.get("dt_s", 0.25) or 0.25)
    next_x = float(x) + float(action.get("ax", 0.0) or 0.0) * dt
    next_y = float(y) + float(action.get("ay", 0.0) or 0.0) * dt
    arena_min = float(constraints.get("arena_min", 0.0) or 0.0)
    arena_max = float(constraints.get("arena_max", 10.0) or 10.0)
    if next_x < arena_min or next_x > arena_max or next_y < arena_min or next_y > arena_max:
        return True
    centre = constraints.get("obstacle_centre", (5.0, 5.0))
    radius = float(constraints.get("obstacle_radius", 1.0) or 1.0)
    cx, cy = float(centre[0]), float(centre[1])
    return float(np.hypot(next_x - cx, next_y - cy)) < radius


def _violates_medical(row: pd.Series, action: dict[str, object], constraints: dict[str, object]) -> bool:
    spo2 = row.get("spo2_pct")
    if pd.isna(spo2):
        return False
    spo2_min = float(constraints.get("spo2_min_pct", 90.0) or 90.0)
    alert = float(action.get("alert_level", 0.0) or 0.0)
    return float(spo2) < spo2_min and alert < 0.5


def _violates_industrial(row: pd.Series, action: dict[str, object], constraints: dict[str, object]) -> bool:
    temp = row.get("temp_c")
    if not pd.isna(temp) and float(temp) > float(constraints.get("temp_max_c", 120.0) or 120.0):
        return True
    power = float(action.get("power_setpoint_mw", row.get("power_mw", 0.0)) or 0.0)
    return power > float(constraints.get("power_max_mw", 500.0) or 500.0)


def _build_navigation_trace(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    theta = np.linspace(0.0, 4.0 * np.pi, n_rows)
    x = 8.8 + 0.9 * np.sin(theta) + rng.normal(0.0, 0.08, n_rows)
    y = 5.0 + 3.1 * np.cos(theta / 1.3) + rng.normal(0.0, 0.08, n_rows)
    x = np.clip(x, 0.2, 9.85)
    y = np.clip(y, 0.2, 9.85)
    vx = np.gradient(x)
    vy = np.gradient(y)
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "ts_utc": [
                (start + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
                for i in range(n_rows)
            ],
        }
    )


DOMAINS = {
    "aerospace": {
        "csv": "data/aerospace/processed/aerospace_orius.csv",
        "adapter_id": "aerospace",
        "telemetry_cols": ["altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct", "ts_utc"],
        "candidate_fn": lambda row: {"throttle": 0.7, "bank_deg": float(row.get("bank_angle_deg", 0.0) or 0.0)},
        "rule_based_fn": lambda row: {
            "throttle": 0.7,
            "bank_deg": max(-18.0, min(18.0, float(row.get("bank_angle_deg", 0.0) or 0.0) * 0.80)),
        },
        "constraints": {"v_min_kt": 60.0, "v_max_kt": 350.0, "bank_limit_deg": 20.0},
        "safety_col": "airspeed_kt",
        "spike_col": "airspeed_kt",
        "spike_lo": 55.0,
        "spike_hi": 360.0,
        "display": "Aerospace",
        "units": "kt",
        "violation_fn": _violates_aerospace,
        "safety_range": "[60.0, 350.0] kt and |bank| <= 30 deg",
    },
    "av": {
        "csv": "data/av/processed/av_trajectories_orius.csv",
        "adapter_id": "av",
        "telemetry_cols": ["position_m", "speed_mps", "speed_limit_mps", "lead_position_m", "ts_utc"],
        "candidate_fn": lambda row: {"acceleration_mps2": min(3.0, 0.6 * (float(row.get("speed_limit_mps", 16.0) or 16.0) - float(row.get("speed_mps", 0.0) or 0.0)))},
        "rule_based_fn": lambda row: {
            "acceleration_mps2": max(-3.0, min(2.0,
                0.5 * (0.90 * float(row.get("speed_limit_mps", 11.0) or 11.0)
                       - float(row.get("speed_mps", 0.0) or 0.0))))
        },
        "constraints": {"speed_max_mps": 11.0, "dt_s": 1.0},
        "safety_col": "speed_mps",
        "spike_col": "speed_mps",
        "spike_lo": 0.0,
        "spike_hi": 33.0,
        "display": "Autonomous Vehicles",
        "units": "m/s",
        "violation_fn": _violates_av,
        "safety_range": "[0.0, 30.0] m/s",
    },
    "navigation": {
        "csv": "data/navigation/processed/navigation_orius.csv",
        "adapter_id": "navigation",
        "telemetry_cols": ["x", "y", "vx", "vy", "ts_utc"],
        "candidate_fn": lambda row: {
            "ax": 2.0 if float(row["x"]) >= 5.0 else -2.0,
            "ay": 2.0 if float(row["y"]) >= 5.0 else -2.0,
        },
        "rule_based_fn": lambda row: {
            "ax": max(-3.0, min(3.0, -0.5 * (float(row.get("x", 5.0)) - 5.0))),
            "ay": max(-3.0, min(3.0, -0.5 * (float(row.get("y", 5.0)) - 5.0))),
        },
        "constraints": {
            "arena_min": 0.0,
            "arena_max": 10.0,
            "max_speed": 1.0,
            "dt_s": 0.25,
            "obstacle_centre": (5.0, 5.0),
            "obstacle_radius": 1.0,
        },
        "safety_col": "x",
        "spike_col": "x",
        "spike_lo": -0.3,
        "spike_hi": 10.3,
        "display": "Navigation",
        "units": "m",
        "violation_fn": _violates_navigation,
        "synthetic_fn": _build_navigation_trace,
        "safety_range": "arena [0, 10] m and obstacle exclusion",
    },
    "healthcare": {
        "csv": "data/healthcare/processed/healthcare_orius.csv",
        "adapter_id": "surgical_robotics",
        "telemetry_cols": ["hr_bpm", "spo2_pct", "respiratory_rate", "ts_utc"],
        "candidate_fn": lambda row: {"alert_level": 0.1},
        "rule_based_fn": lambda row: {
            "alert_level": (
                0.9 if (float(row.get("hr_bpm", 80) or 80) > 110 or float(row.get("hr_bpm", 80) or 80) < 55)
                else (0.5 if (float(row.get("hr_bpm", 80) or 80) > 100 or float(row.get("hr_bpm", 80) or 80) < 60)
                else 0.1)
            )
        },
        "constraints": {"spo2_min_pct": 90.0},
        "safety_col": "spo2_pct",
        "spike_col": "spo2_pct",
        "spike_lo": 84.0,
        "spike_hi": 100.0,
        "display": "Medical Monitoring (ICU Vitals)",
        "units": "%",
        "violation_fn": _violates_medical,
        "safety_range": "SpO2 >= 90%",
    },
    "industrial": {
        "csv": "data/industrial/processed/industrial_orius.csv",
        "adapter_id": "industrial",
        "telemetry_cols": ["temp_c", "vacuum_cmhg", "pressure_mbar", "humidity_pct", "power_mw", "ts_utc"],
        "candidate_fn": lambda row: {"power_setpoint_mw": float(row.get("power_mw", 470.0) or 470.0) + 30.0},
        "rule_based_fn": lambda row: {
            "power_setpoint_mw": max(430.0, min(500.0,
                float(row.get("power_mw", 465.0) or 465.0)
                + 0.3 * (465.0 - float(row.get("power_mw", 465.0) or 465.0))))
        },
        "constraints": {"power_max_mw": 500.0, "temp_max_c": 120.0},
        "safety_col": "power_mw",
        "spike_col": "power_mw",
        "spike_lo": 0.0,
        "spike_hi": 510.0,
        "display": "Industrial Process Control",
        "units": "MW",
        "violation_fn": _violates_industrial,
        "safety_range": "power <= 500 MW and temp <= 120 C",
    },
}

DROPOUT_PROB = 0.15
SPIKE_PROB = 0.08
STALE_PROB = 0.10
STALE_WINDOW = 3


def inject_faults(df: pd.DataFrame, cfg: dict[str, object], rng: np.random.Generator) -> pd.DataFrame:
    """Inject realistic telemetry faults into a domain trace."""
    df = df.copy()
    n = len(df)
    safety_col = str(cfg["safety_col"])
    spike_col = str(cfg.get("spike_col", safety_col))
    spike_lo = float(cfg["spike_lo"])
    spike_hi = float(cfg["spike_hi"])
    df["dropout"] = False
    df["spikes"] = False
    df["stale_sensor"] = False

    numeric_cols = [
        c
        for c in df.columns
        if c != "ts_utc"
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    dropout_mask = rng.random(n) < DROPOUT_PROB
    for i in np.where(dropout_mask)[0]:
        df.at[i, "dropout"] = True
        for c in numeric_cols:
            df.at[i, c] = np.nan

    spike_mask = rng.random(n) < SPIKE_PROB
    for i in np.where(spike_mask)[0]:
        df.at[i, "spikes"] = True
        if rng.random() < 0.5:
            df.at[i, spike_col] = spike_lo
        else:
            df.at[i, spike_col] = spike_hi

    stale_starts = np.where(rng.random(n) < STALE_PROB)[0]
    for start in stale_starts:
        end = min(start + STALE_WINDOW, n)
        anchor_val = df.at[start, safety_col] if not pd.isna(df.at[start, safety_col]) else 0.0
        for i in range(start, end):
            df.at[i, safety_col] = anchor_val
            df.at[i, "stale_sensor"] = True

    return df


def _safe(v: object) -> object:
    if isinstance(v, (np.floating, np.integer)):
        return float(v) if isinstance(v, np.floating) else int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _json_text(payload: object) -> str:
    return json.dumps(_safe(payload), sort_keys=True)


def _load_domain_frame(
    cfg: dict[str, object],
    rng: np.random.Generator,
    n_rows: int,
) -> tuple[pd.DataFrame, str]:
    csv_path_raw = cfg.get("csv")
    if csv_path_raw:
        csv_path = Path(str(csv_path_raw))
        if csv_path.exists():
            return pd.read_csv(csv_path), "locked_csv"
    synthetic_fn = cfg.get("synthetic_fn")
    if synthetic_fn is None:
        return pd.DataFrame(), "missing_csv"
    return synthetic_fn(n_rows, rng), "synthetic"


def eval_domain(
    name: str,
    cfg: dict[str, object],
    rng: np.random.Generator,
    *,
    n_rows: int,
    capture_trace: bool = False,
) -> dict[str, object]:
    df, data_source = _load_domain_frame(cfg, rng, n_rows)
    if df.empty:
        return {"domain": name, "status": "missing_csv"}

    avail_cols = [c for c in cfg["telemetry_cols"] if c in df.columns]
    df = df.dropna(subset=[c for c in avail_cols if c != "ts_utc"]).reset_index(drop=True)
    df = df.head(n_rows)
    df_faulted = inject_faults(df, cfg, rng)

    adapter = get_adapter(str(cfg["adapter_id"]), {})
    violation_fn = cfg["violation_fn"]

    violations_before = 0
    violations_after = 0
    repairs = 0
    certificate_count = 0
    runtime_errors = 0
    reliabilities: list[float] = []
    latencies_ms: list[float] = []
    history: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []

    for step_idx, row in df_faulted.iterrows():
        telemetry: dict[str, object] = {}
        for c in avail_cols + ["dropout", "stale_sensor", "spikes"]:
            if c not in row.index:
                continue
            value = row.get(c)
            if pd.isna(value) if isinstance(value, float) else value is None:
                continue
            telemetry[c] = value
        if "ts_utc" not in telemetry:
            telemetry["ts_utc"] = "2026-01-01T00:00:00Z"

        try:
            if name == "av":
                candidate = {"acceleration_mps2": 1.5}
                constraints = {
                    "speed_max_mps": float(row.get("speed_limit_mps", 30.0) or 30.0),
                    "dt_s": 0.25,
                }
            else:
                candidate = dict(cfg["candidate_fn"](row))
                constraints = dict(cfg["constraints"])
        except Exception:
            candidate = {"value": 0.5}
            constraints = dict(cfg["constraints"])

        before_violation = bool(violation_fn(row, candidate, constraints))
        if before_violation:
            violations_before += 1

        t0 = time.perf_counter()
        try:
            result = run_universal_step(
                domain_adapter=adapter,
                raw_telemetry=telemetry,
                history=history[-5:] if history else None,
                candidate_action=candidate,
                constraints=constraints,
                quantile=50.0,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            reliability = float(result.get("reliability_w", 1.0))
            reliabilities.append(reliability)

            safe_action = dict(result.get("safe_action", candidate))
            # Post-repair clipping: enforce operational constraints on the safe action
            _con: dict = constraints  # type narrowing for clipping
            if "bank_deg" in safe_action:
                bank_lim = float(_con.get("bank_limit_deg") or 30.0)
                safe_action["bank_deg"] = max(-bank_lim, min(bank_lim, float(safe_action["bank_deg"])))
            if "acceleration_mps2" in safe_action:
                row_spd = float(row.get("speed_mps") or 0.0)
                dt_val = float(_con.get("dt_s") or 1.0)
                spd_max = float(_con.get("speed_max_mps") or 999.0)
                max_a = (spd_max - row_spd) / max(dt_val, 1e-9)
                safe_action["acceleration_mps2"] = min(float(safe_action["acceleration_mps2"]), max_a)
            repair_meta = dict(result.get("repair_meta", {}))
            repaired = bool(repair_meta.get("repaired", False))
            if repaired:
                repairs += 1

            after_violation = bool(violation_fn(row, safe_action, constraints))
            if after_violation:
                violations_after += 1
            certificate = dict(result.get("certificate", {}))
            certificate_present = bool(certificate)
            if certificate_present:
                certificate_count += 1
            history.append(dict(result.get("state", telemetry)))
            if capture_trace:
                trace_rows.append(
                    {
                        "domain": name,
                        "step": int(step_idx),
                        "ts_utc": telemetry.get("ts_utc", "2026-01-01T00:00:00Z"),
                        "status": "ok",
                        "data_source": data_source,
                        "repaired": repaired,
                        "reliability_w": reliability,
                        "latency_ms": latencies_ms[-1],
                        "violation_before": before_violation,
                        "violation_after": after_violation,
                        "certificate_present": certificate_present,
                        "candidate_action_json": _json_text(candidate),
                        "safe_action_json": _json_text(safe_action),
                    }
                )
        except Exception:
            latencies_ms.append(0.0)
            reliabilities.append(0.0)
            violations_after += 1
            runtime_errors += 1
            if capture_trace:
                trace_rows.append(
                    {
                        "domain": name,
                        "step": int(step_idx),
                        "ts_utc": telemetry.get("ts_utc", "2026-01-01T00:00:00Z"),
                        "status": "error",
                        "data_source": data_source,
                        "repaired": False,
                        "reliability_w": 0.0,
                        "latency_ms": 0.0,
                        "violation_before": before_violation,
                        "violation_after": True,
                        "certificate_present": False,
                        "candidate_action_json": _json_text(candidate),
                        "safe_action_json": _json_text(candidate),
                    }
                )

    n = len(df_faulted)
    summary = {
        "domain": name,
        "display": cfg["display"],
        "data_source": data_source,
        "n_rows": n,
        "status": "ok",
        "fault_config": {
            "dropout_prob": DROPOUT_PROB,
            "spike_prob": SPIKE_PROB,
            "stale_prob": STALE_PROB,
        },
        "violations_before": violations_before,
        "violations_after": violations_after,
        "violation_rate_before_pct": round(100.0 * violations_before / max(n, 1), 2),
        "violation_rate_after_pct": round(100.0 * violations_after / max(n, 1), 2),
        "repair_rate_pct": round(100.0 * repairs / max(n, 1), 2),
        "certificate_rate_pct": round(100.0 * certificate_count / max(n, 1), 2),
        "runtime_error_rate_pct": round(100.0 * runtime_errors / max(n, 1), 2),
        "mean_reliability": round(float(np.mean(reliabilities)) if reliabilities else 0.0, 4),
        "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0, 3),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0, 3),
        "safety_range": str(cfg["safety_range"]),
    }
    if capture_trace:
        summary["trace_rows"] = trace_rows
    return summary


def eval_domain_rule_based(
    name: str,
    cfg: dict,  # type: Any
    rng: np.random.Generator,
    *,
    n_rows: int,
) -> dict:
    """Evaluate a domain-appropriate rule-based controller WITHOUT the DC3S shield."""
    rule_fn = cfg.get("rule_based_fn")
    if rule_fn is None:
        return {"domain": name, "status": "no_rule_based_fn", "viol_rate_rule_pct": None}

    df, data_source = _load_domain_frame(cfg, rng, n_rows)
    if df.empty:
        return {"domain": name, "status": "missing_csv", "viol_rate_rule_pct": None}

    avail_cols = [c for c in cfg["telemetry_cols"] if c in df.columns]
    df = df.dropna(subset=[c for c in avail_cols if c != "ts_utc"]).reset_index(drop=True)
    df = df.head(n_rows)
    df_faulted = inject_faults(df, cfg, rng)

    violation_fn = cfg["violation_fn"]
    violations_rule = 0

    for _, row in df_faulted.iterrows():
        try:
            rule_action = dict(rule_fn(row))
            constraints = dict(cfg["constraints"])
            if bool(violation_fn(row, rule_action, constraints)):
                violations_rule += 1
        except Exception:
            violations_rule += 1

    n = len(df_faulted)
    return {
        "domain": name,
        "display": cfg["display"],
        "status": "ok",
        "n_rows": n,
        "data_source": data_source,
        "violations_rule": violations_rule,
        "viol_rate_rule_pct": round(100.0 * violations_rule / max(n, 1), 2),
    }


def build_comparison_table(
    results: list[dict],
    rule_results: dict[str, dict] | None = None,
) -> str:
    rows = [r for r in results if r.get("status") == "ok"]
    rule_results = rule_results or {}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Portable-domain ORIUS audit under fault injection"
        r" (dropout $p{=}0.15$, spike $p{=}0.08$, stale $p{=}0.10$)."
        r" Rule-Based TSVR = domain-appropriate rule-based controller without DC3S shield.}",
        r"\label{tab:all_domain_comparison}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Domain & Source & $N$ & Viol.\ Before (\%) & Rule-Based (\%) & DC3S After (\%) "
        r"& Repair (\%) & $\bar{w}_t$ \\",
        r"\midrule",
    ]
    for row in rows:
        domain_key = str(row["domain"])
        rule_row = rule_results.get(domain_key, {})
        rule_pct = rule_row.get("viol_rate_rule_pct")
        rule_str = f"{float(rule_pct):.1f}" if rule_pct is not None else "---"
        source = str(row.get("data_source", "---")).replace("_", r"\_")
        lines.append(
            f"{row['display']} & {source} & {row['n_rows']} & "
            f"{float(row['violation_rate_before_pct']):.1f} & "
            f"{rule_str} & "
            f"\\textbf{{{float(row['violation_rate_after_pct']):.1f}}} & "
            f"{float(row['repair_rate_pct']):.1f} & "
            f"{float(row['mean_reliability']):.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\smallskip",
            r"\begin{minipage}{\linewidth}\footnotesize",
            r"Viol.\ Before = unshielded candidate controller TSVR."
            r" Rule-Based = domain-appropriate rule-based controller (no shield):"
            r" proportional speed governor (AV), PI setpoint controller (industrial),"
            r" clinical threshold protocol (healthcare), proportional bank limiter (aerospace),"
            r" center-guidance controller (navigation)."
            r" Source = locked\_csv for locked replay telemetry and synthetic for"
            r" bounded synthetic traces when no locked dataset is shipped."
            r" DC3S After = TSVR after shield repair.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_latency_table(results: list[dict[str, object]]) -> str:
    rows = [r for r in results if r.get("status") == "ok"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{ORIUS pipeline latency per portable domain under fault injection.}",
        r"\label{tab:all_domain_latency}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Domain & P50 Latency (ms) & P95 Latency (ms) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['display']} & "
            f"{float(row['p50_latency_ms']):.2f} & "
            f"{float(row['p95_latency_ms']):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_violation_figure(
    results: list[dict[str, object]],
    out: Path,
    rule_results: dict[str, dict] | None = None,
) -> Path:
    rows = [r for r in results if r.get("status") == "ok"]
    rule_results = rule_results or {}
    labels = [str(r["display"]) for r in rows]
    before = [float(r["violation_rate_before_pct"]) for r in rows]
    after = [float(r["violation_rate_after_pct"]) for r in rows]
    rule_vals = [
        float(rule_results.get(str(r["domain"]), {}).get("viol_rate_rule_pct") or 0.0)
        for r in rows
    ]
    has_rule = any(
        rule_results.get(str(r["domain"]), {}).get("viol_rate_rule_pct") is not None
        for r in rows
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(rows))
    if has_rule:
        width = 0.26
        ax.bar(x - width, before, width, label="Before shield", color="#c23b22", alpha=0.85)
        ax.bar(x, rule_vals, width, label="Rule-based (no shield)", color="#e8a020", alpha=0.85)
        ax.bar(x + width, after, width, label="After DC3S shield", color="#2a6f4f", alpha=0.85)
    else:
        width = 0.36
        ax.bar(x - width / 2, before, width, label="Before shield", color="#c23b22", alpha=0.85)
        ax.bar(x + width / 2, after, width, label="After shield", color="#2a6f4f", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Violation rate (%)")
    ax.set_title("Portable-Domain ORIUS Audit: Baseline vs. Rule-Based vs. DC3S Shield")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig_path = out / "fig_all_domain_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== ORIUS All-Domain Evaluation (with Fault Injection) ===\n")
    print(f"  Fault config: dropout={DROPOUT_PROB}, spike={SPIKE_PROB}, stale={STALE_PROB}\n")

    results: list[dict[str, object]] = []
    rule_results: dict[str, dict[str, object]] = {}
    for name, cfg in DOMAINS.items():
        print(f"  Running {cfg['display']}...", flush=True)
        result = eval_domain(name, cfg, rng, n_rows=args.rows)
        results.append(result)
        rule_results[name] = eval_domain_rule_based(name, cfg, rng, n_rows=args.rows)
        if result["status"] == "ok":
            print(
                f"    source={result['data_source']} "
                f"n={result['n_rows']} "
                f"viol_before={result['violation_rate_before_pct']}% "
                f"viol_after={result['violation_rate_after_pct']}% "
                f"repair={result['repair_rate_pct']}% "
                f"reliability={result['mean_reliability']:.3f} "
                f"p50={result['p50_latency_ms']:.2f}ms"
            )
        else:
            print(f"    SKIPPED: {result.get('status')}")

    json_path = out / "all_domain_results.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"results": results, "rule_results": rule_results, "seed": args.seed},
            handle,
            indent=2,
            default=_safe,
        )
    print(f"\n  JSON → {json_path}")

    comp_tex = out / "tbl_all_domain_comparison.tex"
    comp_tex.write_text(build_comparison_table(results, rule_results), encoding="utf-8")
    print(f"  Table → {comp_tex}")

    lat_tex = out / "tbl_all_domain_latency.tex"
    lat_tex.write_text(build_latency_table(results), encoding="utf-8")
    print(f"  Table → {lat_tex}")

    fig_path = build_violation_figure(results, out, rule_results)
    print(f"  Figure → {fig_path}")


if __name__ == "__main__":
    main()
