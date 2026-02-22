"""Run edge-agent loop against real HTTP gateway and GridPulse API."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import yaml

from iot.edge_agent.drivers.http_gateway import HTTPGatewayDriver


class GridPulseApiClient:
    """Thin HTTP client for GridPulse API calls with scoped API key headers."""

    def __init__(self, *, base_url: str, api_key: str, timeout_s: float = 10.0, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.session = session or requests.Session()

    def close(self) -> None:
        self.session.close()

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        headers = {"X-GridPulse-Key": self.api_key}
        resp = self.session.request(
            method=method.upper(),
            url=url,
            params=params,
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"GridPulse API {method.upper()} {url} failed: {resp.status_code} {resp.text}")
        body = resp.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"GridPulse API {method.upper()} {url} returned non-object JSON")
        return body

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json(method="POST", path=path, payload=payload)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request_json(method="GET", path=path, params=params)


def _load_iot_cfg(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing IoT config: {cfg_path}")
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    iot_cfg = payload.get("iot", {}) if isinstance(payload, dict) else {}
    if not isinstance(iot_cfg, dict):
        raise RuntimeError("configs/iot.yaml must contain top-level 'iot' object")
    return iot_cfg


def _extract_action(command_payload: dict[str, Any]) -> tuple[float, float]:
    command = dict(command_payload.get("command", {}))
    if "safe_action" in command and isinstance(command["safe_action"], dict):
        return float(command["safe_action"].get("charge_mw", 0.0)), float(command["safe_action"].get("discharge_mw", 0.0))
    return float(command.get("charge_mw", 0.0)), float(command.get("discharge_mw", 0.0))


def run_one_iteration(
    *,
    api: GridPulseApiClient,
    driver: HTTPGatewayDriver,
    device_id: str,
    zone_id: str,
    mode: str,
    controller: str,
    horizon: int,
    queue_ttl_seconds: int,
) -> dict[str, Any]:
    telemetry = driver.fetch_telemetry()
    telemetry.setdefault("ts_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    telemetry["load_mw"] = float(telemetry["load_mw"])
    telemetry["renewables_mw"] = float(telemetry["renewables_mw"])

    api.post(
        "/iot/telemetry",
        {
            "device_id": device_id,
            "zone_id": zone_id,
            "telemetry_event": telemetry,
        },
    )

    load_val = float(telemetry["load_mw"])
    step_payload = {
        "device_id": device_id,
        "zone_id": zone_id,
        "current_soc_mwh": float(telemetry.get("soc_mwh", 0.0) or 0.0),
        "telemetry_event": telemetry,
        "last_actual_load_mw": load_val,
        "last_pred_load_mw": load_val,
        "controller": controller,
        "horizon": int(horizon),
        "enqueue_iot": True,
        "queue_ttl_seconds": int(queue_ttl_seconds),
        "include_certificate": True,
    }
    step = api.post("/dc3s/step", step_payload)

    next_cmd = api.get(
        "/iot/command/next",
        params={"device_id": device_id, "peek": "false"},
    )
    cmd_status = str(next_cmd.get("status", "empty"))
    if cmd_status in {"empty", "hold"}:
        return {
            "status": cmd_status,
            "queue_status": step.get("queue_status"),
            "command_id": None,
            "ack_status": None,
        }

    command = dict(next_cmd.get("command") or {})
    command_id = str(command.get("command_id"))
    charge_mw, discharge_mw = _extract_action(command)

    if mode == "shadow":
        apply_payload = {
            "accepted": True,
            "violation": False,
            "shadow_mode": True,
            "applied": False,
            "recommended_action": {
                "charge_mw": float(charge_mw),
                "discharge_mw": float(discharge_mw),
            },
        }
        ack_status = "acked"
        ack_reason = None
    else:
        apply_payload = driver.apply_command(charge_mw=float(charge_mw), discharge_mw=float(discharge_mw))
        ack_status = "nacked" if bool(apply_payload.get("violation", False)) else "acked"
        ack_reason = None if ack_status == "acked" else "safety_violation"

    api.post(
        "/iot/ack",
        {
            "device_id": device_id,
            "command_id": command_id,
            "status": ack_status,
            "certificate_id": command.get("certificate_id"),
            "reason": ack_reason,
            "payload": apply_payload,
        },
    )
    return {
        "status": "ok",
        "queue_status": step.get("queue_status"),
        "command_id": command_id,
        "ack_status": ack_status,
        "shadow_mode": bool(apply_payload.get("shadow_mode", False)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GridPulse edge-agent loop against HTTP gateway")
    parser.add_argument("--config", default="configs/iot.yaml")
    parser.add_argument("--mode", choices=["shadow", "active"], default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--poll-interval-s", type=float, default=None)
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--zone-id", choices=["DE", "US"], default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_iot_cfg(args.config)
    defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults", {}), dict) else {}
    api_cfg = cfg.get("api", {}) if isinstance(cfg.get("api", {}), dict) else {}
    gateway_cfg = cfg.get("gateway", {}) if isinstance(cfg.get("gateway", {}), dict) else {}

    mode = args.mode or str(defaults.get("mode", "shadow"))
    iterations = int(args.iterations if args.iterations is not None else defaults.get("iterations", 24))
    poll_interval = float(args.poll_interval_s if args.poll_interval_s is not None else defaults.get("poll_interval_s", 5.0))
    device_id = str(args.device_id or defaults.get("device_id", "edge-device-001"))
    zone_id = str(args.zone_id or defaults.get("zone_id", "DE"))
    queue_ttl_seconds = int(defaults.get("queue_ttl_seconds", 30))
    controller = str(defaults.get("controller", "deterministic"))
    horizon = int(defaults.get("horizon", 24))

    api_key_env = str(api_cfg.get("api_key_env", "GRIDPULSE_IOT_API_KEY"))
    api_key = str(args.api_key or os.getenv(api_key_env) or api_cfg.get("api_key", "")).strip()
    if not api_key:
        raise RuntimeError(f"Missing GridPulse API key. Provide --api-key or set {api_key_env}.")

    gateway_token_env = str(gateway_cfg.get("auth_token_env", "GRIDPULSE_GATEWAY_TOKEN"))
    gateway_token = os.getenv(gateway_token_env, gateway_cfg.get("auth_token", None))

    api = GridPulseApiClient(
        base_url=str(args.api_base_url or api_cfg.get("base_url", "http://localhost:8000")),
        api_key=api_key,
        timeout_s=float(api_cfg.get("timeout_s", 10.0)),
    )
    driver = HTTPGatewayDriver(
        base_url=str(gateway_cfg.get("base_url", "http://localhost:9001")),
        telemetry_path=str(gateway_cfg.get("telemetry_path", "/telemetry/latest")),
        command_path=str(gateway_cfg.get("command_path", "/command/apply")),
        timeout_s=float(gateway_cfg.get("timeout_s", 5.0)),
        retries=int(gateway_cfg.get("retries", 2)),
        auth_header=gateway_cfg.get("auth_header"),
        auth_token=gateway_token,
    )

    acked = 0
    nacked = 0
    hold_events = 0
    empty_events = 0
    errors = 0
    try:
        for idx in range(iterations):
            try:
                out = run_one_iteration(
                    api=api,
                    driver=driver,
                    device_id=device_id,
                    zone_id=zone_id,
                    mode=mode,
                    controller=controller,
                    horizon=horizon,
                    queue_ttl_seconds=queue_ttl_seconds,
                )
            except Exception as exc:
                errors += 1
                out = {"status": "error", "error": str(exc)}

            if out.get("status") == "ok":
                if out.get("ack_status") == "acked":
                    acked += 1
                else:
                    nacked += 1
            elif out.get("status") == "hold":
                hold_events += 1
            elif out.get("status") == "empty":
                empty_events += 1

            out["iteration"] = idx + 1
            print(json.dumps(out, sort_keys=True))
            if idx < iterations - 1:
                time.sleep(max(0.0, poll_interval))
    finally:
        driver.close()
        api.close()

    summary = {
        "mode": mode,
        "iterations": iterations,
        "acked": acked,
        "nacked": nacked,
        "hold_events": hold_events,
        "empty_events": empty_events,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
