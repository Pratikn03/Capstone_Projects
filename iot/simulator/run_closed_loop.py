"""Run an end-to-end IoT closed-loop simulation against in-process FastAPI routes."""
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

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from iot.edge_agent.agent import EdgeAgent
from iot.edge_agent.drivers.sim import SimBatteryDriver
from services.api.main import app
from services.api.config import get_api_keys
from services.api.routers import dc3s as dc3s_router


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GridPulse IoT closed-loop simulation")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device-id", type=str, default="sim-battery-001")
    parser.add_argument("--zone-id", type=str, default="DE")
    parser.add_argument("--scenario", type=str, default="nominal", choices=["nominal", "dropout"])
    return parser.parse_args()


def _synthetic_predictor(seed: int):
    rng = np.random.default_rng(seed)

    def _predict_target(*, target: str, horizon: int, features_df: pd.DataFrame, forecast_cfg: dict[str, Any], required: bool):
        idx = np.arange(horizon, dtype=float)
        noise = rng.normal(0.0, 5.0, size=horizon)
        if target == "load_mw":
            y = 52.0 + 4.0 * np.sin((2.0 * math.pi * idx / 24.0) - 0.5) + 1.2 * np.sin(2.0 * math.pi * idx / 12.0) + 0.05 * noise
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
    if scenario == "dropout" and step % 6 == 0:
        payload["dropout"] = True
    return payload


def run_closed_loop(*, steps: int, seed: int, device_id: str, zone_id: str, scenario: str) -> dict[str, Any]:
    api_key = "iot-sim-key"
    os.environ["GRIDPULSE_API_KEYS"] = json.dumps({api_key: ["read", "write"]})
    get_api_keys.cache_clear()

    client = TestClient(app)
    driver = SimBatteryDriver()
    agent = EdgeAgent(client=client, device_id=device_id, zone_id=zone_id, driver=driver, api_key=api_key)
    start_ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    auth_headers = {"X-GridPulse-Key": api_key}

    safety_violations = 0
    interventions = 0
    certificates_ok = 0
    command_count = 0

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
            dc3s_resp = client.post("/dc3s/step", json=req)
            dc3s_resp.raise_for_status()
            dc3s_payload = dc3s_resp.json()

            command_id = str(dc3s_payload["command_id"])
            command_count += 1
            proposed = dc3s_payload["proposed_action"]
            safe = dc3s_payload["safe_action"]
            if abs(float(proposed["charge_mw"]) - float(safe["charge_mw"])) > 1e-6 or abs(
                float(proposed["discharge_mw"]) - float(safe["discharge_mw"])
            ) > 1e-6:
                interventions += 1

            cmd = agent.poll_next_command(peek=False)
            if cmd is None:
                raise RuntimeError("Expected queued command, got empty queue")

            charge_mw, discharge_mw = EdgeAgent._extract_action(cmd)
            applied = driver.apply_command(charge_mw=charge_mw, discharge_mw=discharge_mw)
            if bool(applied.get("violation", False)):
                safety_violations += 1
            ack_status = "nacked" if bool(applied.get("violation", False)) else "acked"
            agent.send_ack(
                command_id=command_id,
                status=ack_status,
                certificate_id=cmd.get("certificate_id"),
                reason=None if ack_status == "acked" else "safety_violation",
                payload=applied,
            )

            audit = client.get(f"/iot/audit/{command_id}", headers=auth_headers)
            audit.raise_for_status()
            cert = audit.json()
            required_fields = {"command_id", "certificate_hash", "safe_action", "proposed_action"}
            if required_fields.issubset(set(cert.keys())):
                certificates_ok += 1
    summary = {
        "scenario": scenario,
        "steps": int(steps),
        "commands_processed": int(command_count),
        "safety_violations": int(safety_violations),
        "interventions": int(interventions),
        "certificate_completeness_rate": float(certificates_ok / max(command_count, 1)),
    }
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_closed_loop(
        steps=int(args.steps),
        seed=int(args.seed),
        device_id=str(args.device_id),
        zone_id=str(args.zone_id),
        scenario=str(args.scenario),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.scenario == "nominal" and summary["safety_violations"] != 0:
        raise SystemExit("Nominal scenario safety violations must be zero")


if __name__ == "__main__":
    main()
