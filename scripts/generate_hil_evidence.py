#!/usr/bin/env python3
"""Generate Hardware-in-the-Loop (HIL) evidence package for thesis ch27.

Runs the closed-loop IoT simulator under multiple scenarios and produces:
  - reports/hil/hil_summary.json
  - reports/hil/hil_step_log.csv
  - reports/hil/fig_hil_soc_trace.png
  - reports/hil/hil_certificate_audit.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iot.edge_agent.agent import EdgeAgent
from iot.edge_agent.drivers.sim import SimBatteryDriver
from services.api.main import app
from services.api.config import get_api_keys
from services.api.routers import dc3s as dc3s_router
from fastapi.testclient import TestClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HIL evidence package")
    parser.add_argument("--steps", type=int, default=48, help="Steps per scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports/hil")
    return parser.parse_args()


def _synthetic_predictor(seed: int):
    rng = np.random.default_rng(seed)

    def _predict_target(*, target: str, horizon: int, features_df: pd.DataFrame,
                        forecast_cfg: dict[str, Any], required: bool):
        idx = np.arange(horizon, dtype=float)
        noise = rng.normal(0.0, 5.0, size=horizon)
        if target == "load_mw":
            y = 52.0 + 4.0 * np.sin((2.0 * math.pi * idx / 24.0) - 0.5) + 0.05 * noise
        elif target == "wind_mw":
            y = 8.0 + 1.8 * np.sin((2.0 * math.pi * (idx + 3.0) / 24.0)) + 0.05 * noise
        else:
            y = np.maximum(0.0, 4.0 * np.sin(math.pi * ((idx % 24.0) - 6.0) / 12.0) + 0.03 * noise)
        return np.asarray(y, dtype=float), Path(f"synthetic_{target}.bin")

    return _predict_target


def _build_telemetry(step: int, start_ts: datetime, scenario: str) -> dict[str, Any]:
    ts = start_ts + timedelta(hours=step)
    load = 52.0 + 4.0 * math.sin((2.0 * math.pi * step / 24.0) - 0.4)
    renew = max(0.0, 12.0 + 2.5 * math.sin((2.0 * math.pi * (step + 4.0) / 24.0)))
    payload = {
        "ts_utc": ts.isoformat(),
        "load_mw": float(load),
        "renewables_mw": float(renew),
    }
    if scenario == "dropout" and step % 5 == 0:
        payload["dropout"] = True
    if scenario == "spike" and step % 8 == 0:
        payload["load_mw"] = float(load * 1.8)
    return payload


def run_hil_scenario(
    *, scenario: str, steps: int, seed: int,
    device_id: str = "hil-sim-001", zone_id: str = "DE",
) -> dict[str, Any]:
    api_key = "hil-evidence-key"
    os.environ["ORIUS_API_KEYS"] = json.dumps({api_key: ["read", "write"]})
    get_api_keys.cache_clear()

    client = TestClient(app)
    driver = SimBatteryDriver()
    agent = EdgeAgent(
        client=client, device_id=device_id,
        zone_id=zone_id, driver=driver, api_key=api_key,
    )
    start_ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    auth_headers = {"X-ORIUS-Key": api_key}

    step_log = []
    certificates = []
    violations = 0
    interventions = 0
    certs_ok = 0

    features_df = pd.DataFrame({"price_eur_mwh": [60.0], "carbon_kg_per_mwh": [420.0]})

    with ExitStack() as stack:
        stack.enter_context(patch.object(dc3s_router, "_load_features_df", return_value=features_df))
        stack.enter_context(patch.object(dc3s_router, "_predict_target", side_effect=_synthetic_predictor(seed)))
        stack.enter_context(patch.object(dc3s_router, "_resolve_conformal_q", return_value=np.full(24, 4.0, dtype=float)))

        for step in range(steps):
            telemetry = _build_telemetry(step, start_ts=start_ts, scenario=scenario)
            agent.send_telemetry(telemetry)

            actual_load = float(telemetry.get("load_mw") or 52.0)
            predicted_load = actual_load * 0.985
            req = {
                "device_id": device_id,
                "zone_id": zone_id,
                "current_soc_mwh": driver.current_soc_mwh,
                "telemetry_event": telemetry,
                "last_actual_load_mw": actual_load,
                "last_pred_load_mw": predicted_load,
                "controller": "deterministic",
                "horizon": 24,
                "enqueue_iot": True,
                "queue_ttl_seconds": 30,
                "include_certificate": True,
            }
            resp = client.post("/dc3s/step", json=req)
            resp.raise_for_status()
            payload = resp.json()

            command_id = str(payload["command_id"])
            proposed = payload["proposed_action"]
            safe = payload["safe_action"]
            intervened = (
                abs(float(proposed["charge_mw"]) - float(safe["charge_mw"])) > 1e-6
                or abs(float(proposed["discharge_mw"]) - float(safe["discharge_mw"])) > 1e-6
            )
            if intervened:
                interventions += 1

            cmd = agent.poll_next_command(peek=False)
            if cmd is None:
                raise RuntimeError("Expected queued command, got empty queue")

            charge_mw, discharge_mw = EdgeAgent._extract_action(cmd)
            applied = driver.apply_command(charge_mw=charge_mw, discharge_mw=discharge_mw)
            violated = bool(applied.get("violation", False))
            if violated:
                violations += 1

            ack_status = "nacked" if violated else "acked"
            agent.send_ack(
                command_id=command_id, status=ack_status,
                certificate_id=cmd.get("certificate_id"),
                reason=None if ack_status == "acked" else "safety_violation",
                payload=applied,
            )

            audit = client.get(f"/iot/audit/{command_id}", headers=auth_headers)
            audit.raise_for_status()
            cert = audit.json()
            required_fields = {"command_id", "certificate_hash", "safe_action", "proposed_action"}
            cert_complete = required_fields.issubset(set(cert.keys()))
            if cert_complete:
                certs_ok += 1

            step_log.append({
                "step": step,
                "timestamp": telemetry["ts_utc"],
                "scenario": scenario,
                "soc_mwh": float(driver.current_soc_mwh),
                "proposed_charge_mw": float(proposed["charge_mw"]),
                "proposed_discharge_mw": float(proposed["discharge_mw"]),
                "safe_charge_mw": float(safe["charge_mw"]),
                "safe_discharge_mw": float(safe["discharge_mw"]),
                "intervened": intervened,
                "violated": violated,
                "cert_complete": cert_complete,
                "reliability_w": float(payload.get("reliability", {}).get("w_t", 1.0)),
            })

            certificates.append({
                "command_id": command_id,
                "cert_hash": cert.get("certificate_hash", ""),
                "has_safe_action": "safe_action" in cert,
                "has_proposed_action": "proposed_action" in cert,
                "step": step,
                "scenario": scenario,
            })

    return {
        "scenario": scenario,
        "steps": steps,
        "violations": violations,
        "interventions": interventions,
        "cert_completeness": float(certs_ok / max(steps, 1)),
        "step_log": step_log,
        "certificates": certificates,
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["nominal", "dropout"]
    all_step_logs = []
    all_certs = []
    summaries = []

    for scenario in scenarios:
        print(f"Running HIL scenario: {scenario} ({args.steps} steps)")
        result = run_hil_scenario(
            scenario=scenario, steps=args.steps, seed=args.seed,
        )
        all_step_logs.extend(result["step_log"])
        all_certs.extend(result["certificates"])
        summaries.append({
            "scenario": result["scenario"],
            "steps": result["steps"],
            "violations": result["violations"],
            "interventions": result["interventions"],
            "cert_completeness": result["cert_completeness"],
        })
        print(f"  Violations: {result['violations']}, Interventions: {result['interventions']}, "
              f"Cert completeness: {result['cert_completeness']:.2%}")

    step_df = pd.DataFrame(all_step_logs)
    step_csv = out_dir / "hil_step_log.csv"
    step_df.to_csv(step_csv, index=False, float_format="%.4f")
    print(f"\n  Step log -> {step_csv} ({len(step_df)} rows)")

    cert_df = pd.DataFrame(all_certs)
    cert_csv = out_dir / "hil_certificate_audit.csv"
    cert_df.to_csv(cert_csv, index=False)
    print(f"  Certificate audit -> {cert_csv} ({len(cert_df)} rows)")

    summary = {
        "scenarios": summaries,
        "total_steps": int(step_df["step"].count()),
        "total_violations": int(step_df["violated"].sum()),
        "total_interventions": int(step_df["intervened"].sum()),
        "overall_cert_completeness": float(step_df["cert_complete"].mean()),
        "seed": args.seed,
    }
    summary_path = out_dir / "hil_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Summary -> {summary_path}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        sub = step_df[step_df["scenario"] == scenario]
        x = sub["step"].to_numpy()
        ax.plot(x, sub["soc_mwh"], label="SOC", color="#1f77b4", linewidth=1.4)
        ax.axhline(0.5, color="red", linewidth=0.8, linestyle=":", label="Min SOC")
        ax.axhline(9.5, color="red", linewidth=0.8, linestyle=":", label="Max SOC")
        intervened = sub["intervened"].to_numpy(dtype=bool)
        if np.any(intervened):
            ax.scatter(x[intervened], sub["soc_mwh"].to_numpy()[intervened],
                       color="orange", zorder=5, s=25, label="Intervention")
        ax.set_ylabel("SOC (MWh)")
        ax.set_title(f"HIL Closed-Loop: {scenario}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    fig_path = out_dir / "fig_hil_soc_trace.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"  Figure -> {fig_path}")


if __name__ == "__main__":
    main()
