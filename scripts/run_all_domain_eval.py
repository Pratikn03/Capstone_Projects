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
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")
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
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})


def _violates_aerospace(row: pd.Series, action: dict[str, object], constraints: dict[str, object]) -> bool:
    airspeed = row.get("airspeed_kt")
    bank_cmd = float(action.get("bank_deg", row.get("bank_angle_deg", 0.0)) or 0.0)
    v_min = float(constraints.get("v_min_kt", 60.0) or 60.0)
    v_max = float(constraints.get("v_max_kt", 350.0) or 350.0)
    if not pd.isna(airspeed):
        speed = float(airspeed)
        if speed < v_min or speed > v_max:
            return True
    return abs(bank_cmd) > 30.0


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
        "candidate_fn": lambda row: {"throttle": 0.7, "bank_deg": float(row["bank_angle_deg"]) * 0.9},
        "constraints": {"v_min_kt": 60.0, "v_max_kt": 350.0},
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
        "candidate_fn": lambda row: {"acceleration_mps2": 1.5},
        "constraints": {"speed_max_mps": 30.0, "dt_s": 0.25},
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
    reliabilities: list[float] = []
    latencies_ms: list[float] = []
    history: list[dict[str, object]] = []

    for _, row in df_faulted.iterrows():
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

        if violation_fn(row, candidate, constraints):
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
            repair_meta = dict(result.get("repair_meta", {}))
            if bool(repair_meta.get("repaired", False)):
                repairs += 1

            if violation_fn(row, safe_action, constraints):
                violations_after += 1
            history.append(dict(result.get("state", telemetry)))
        except Exception:
            latencies_ms.append(0.0)
            reliabilities.append(0.0)
            violations_after += 1

    n = len(df_faulted)
    return {
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
        "mean_reliability": round(float(np.mean(reliabilities)) if reliabilities else 0.0, 4),
        "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0, 3),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0, 3),
        "safety_range": str(cfg["safety_range"]),
    }


def build_comparison_table(results: list[dict[str, object]]) -> str:
    rows = [r for r in results if r.get("status") == "ok"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Portable-domain ORIUS audit under fault injection"
        r" (dropout $p{=}0.15$, spike $p{=}0.08$, stale $p{=}0.10$).}",
        r"\label{tab:all_domain_comparison}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Domain & Source & $N$ & Faults Inj.\ & Viol.\ Before (\%) & Viol.\ After (\%) "
        r"& Repair Rate (\%) & $\bar{w}_t$ \\",
        r"\midrule",
    ]
    for row in rows:
        n_faults = int(round(float(row["n_rows"]) * (DROPOUT_PROB + SPIKE_PROB + STALE_PROB)))
        data_source = str(row.get("data_source", "---")).replace("_", r"\_")
        lines.append(
            f"{row['display']} & {data_source} & {row['n_rows']} & $\\approx${n_faults} & "
            f"{float(row['violation_rate_before_pct']):.1f} & "
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
            r"Violations Before = fraction of steps where faulted telemetry or aggressive"
            r" candidate action breaches the domain safety constraint before the ORIUS"
            r" shield intervenes. Violations After = fraction remaining after shield"
            r" repair. Navigation uses a synthetic bounded-arena trace because no locked"
            r" real navigation telemetry dataset is currently included in the repo.",
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


def build_violation_figure(results: list[dict[str, object]], out: Path) -> Path:
    rows = [r for r in results if r.get("status") == "ok"]
    labels = [str(r["display"]) for r in rows]
    before = [float(r["violation_rate_before_pct"]) for r in rows]
    after = [float(r["violation_rate_after_pct"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(rows))
    width = 0.36
    ax.bar(x - width / 2, before, width, label="Before shield", color="#c23b22", alpha=0.85)
    ax.bar(x + width / 2, after, width, label="After shield", color="#2a6f4f", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Violation rate (%)")
    ax.set_title("Portable-Domain ORIUS Audit: Before vs. After Shield Repair")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig_path = out / "fig_all_domain_comparison.png"
    fig.savefig(fig_path)
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
    for name, cfg in DOMAINS.items():
        print(f"  Running {cfg['display']}...", flush=True)
        result = eval_domain(name, cfg, rng, n_rows=args.rows)
        results.append(result)
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
        json.dump({"results": results, "seed": args.seed}, handle, indent=2, default=_safe)
    print(f"\n  JSON → {json_path}")

    comp_tex = out / "tbl_all_domain_comparison.tex"
    comp_tex.write_text(build_comparison_table(results), encoding="utf-8")
    print(f"  Table → {comp_tex}")

    lat_tex = out / "tbl_all_domain_latency.tex"
    lat_tex.write_text(build_latency_table(results), encoding="utf-8")
    print(f"  Table → {lat_tex}")

    fig_path = build_violation_figure(results, out)
    print(f"  Figure → {fig_path}")


if __name__ == "__main__":
    main()
