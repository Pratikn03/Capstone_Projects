"""Edge-agent loop primitives for telemetry, command polling, and ACK emission."""
from __future__ import annotations

from typing import Any, Mapping


class EdgeAgent:
    """Device-side loop client for IoT simulator and future hardware adapters."""

    def __init__(
        self,
        *,
        client: Any,
        device_id: str,
        zone_id: str,
        driver: Any,
        api_key: str | None = None,
    ) -> None:
        self.client = client
        self.device_id = device_id
        self.zone_id = zone_id
        self.driver = driver
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"X-GridPulse-Key": self.api_key}

    def send_telemetry(self, telemetry_event: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "device_id": self.device_id,
            "zone_id": self.zone_id,
            "telemetry_event": dict(telemetry_event),
        }
        resp = self.client.post("/iot/telemetry", json=payload, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def poll_next_command(self, *, peek: bool = False) -> dict[str, Any] | None:
        resp = self.client.get(
            "/iot/command/next",
            params={"device_id": self.device_id, "peek": str(peek).lower()},
            headers=self._headers(),
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") in {"empty", "hold"}:
            return None
        return payload.get("command")

    def send_ack(
        self,
        *,
        command_id: str,
        status: str,
        certificate_id: str | None = None,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        req = {
            "device_id": self.device_id,
            "command_id": command_id,
            "status": status,
            "certificate_id": certificate_id,
            "reason": reason,
            "payload": dict(payload or {}),
        }
        resp = self.client.post("/iot/ack", json=req, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_action(command_payload: Mapping[str, Any]) -> tuple[float, float]:
        command = dict(command_payload.get("command", {}))
        if "safe_action" in command and isinstance(command["safe_action"], Mapping):
            action = command["safe_action"]
            return float(action.get("charge_mw", 0.0)), float(action.get("discharge_mw", 0.0))
        if "charge_mw" in command or "discharge_mw" in command:
            return float(command.get("charge_mw", 0.0)), float(command.get("discharge_mw", 0.0))
        return 0.0, 0.0

    def step(self, telemetry_event: Mapping[str, Any]) -> dict[str, Any]:
        telemetry_resp = self.send_telemetry(telemetry_event)
        command = self.poll_next_command(peek=False)
        if command is None:
            return {"status": "idle", "telemetry": telemetry_resp}

        charge_mw, discharge_mw = self._extract_action(command)
        driver_result = self.driver.apply_command(charge_mw=charge_mw, discharge_mw=discharge_mw)
        ack_status = "acked" if not bool(driver_result.get("violation", False)) else "nacked"
        ack_resp = self.send_ack(
            command_id=str(command["command_id"]),
            status=ack_status,
            certificate_id=command.get("certificate_id"),
            reason=None if ack_status == "acked" else "safety_violation",
            payload=driver_result,
        )
        return {
            "status": "ok",
            "telemetry": telemetry_resp,
            "command": command,
            "driver_result": driver_result,
            "ack": ack_resp,
        }
