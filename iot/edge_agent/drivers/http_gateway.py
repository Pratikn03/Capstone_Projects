"""HTTP gateway driver for real-device pilot integration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests


@dataclass
class HTTPGatewayDriver:
    """Adapter for pulling telemetry and issuing commands via HTTP gateway endpoints."""

    base_url: str
    telemetry_path: str = "/telemetry/latest"
    command_path: str = "/command/apply"
    timeout_s: float = 5.0
    retries: int = 2
    auth_header: str | None = None
    auth_token: str | None = None
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.session is None:
            self.session = requests.Session()

    def close(self) -> None:
        if self.session is not None:
            self.session.close()

    def _build_url(self, path: str) -> str:
        clean = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{clean}"

    def _headers(self) -> dict[str, str]:
        if self.auth_header and self.auth_token:
            return {self.auth_header: self.auth_token}
        return {}

    def _request_json(self, *, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("HTTPGatewayDriver session is not initialized")

        last_exc: Exception | None = None
        url = self._build_url(path)
        for attempt in range(int(self.retries) + 1):
            try:
                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    json=payload,
                    headers=self._headers(),
                    timeout=float(self.timeout_s),
                )
                if resp.status_code >= 500 and attempt < int(self.retries):
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"Gateway {method.upper()} {url} failed: {resp.status_code} {resp.text}")
                body = resp.json()
                if not isinstance(body, dict):
                    raise RuntimeError(f"Gateway {method.upper()} {url} returned non-object JSON")
                return body
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= int(self.retries):
                    break
        raise RuntimeError(f"Gateway request failed after retries: {method.upper()} {url}") from last_exc

    def fetch_telemetry(self) -> dict[str, Any]:
        payload = self._request_json(method="GET", path=self.telemetry_path)
        normalized = dict(payload)

        ts = normalized.get("ts_utc") or normalized.get("timestamp") or normalized.get("ts")
        if ts is None:
            ts = datetime.now(timezone.utc).isoformat()
        normalized["ts_utc"] = str(ts)

        if "load_mw" not in normalized and "load" in normalized:
            normalized["load_mw"] = float(normalized["load"])
        if "renewables_mw" not in normalized and "renewables" in normalized:
            normalized["renewables_mw"] = float(normalized["renewables"])

        if "load_mw" not in normalized:
            raise RuntimeError("Gateway telemetry missing required field: load_mw")
        if "renewables_mw" not in normalized:
            raise RuntimeError("Gateway telemetry missing required field: renewables_mw")

        normalized["load_mw"] = float(normalized["load_mw"])
        normalized["renewables_mw"] = float(normalized["renewables_mw"])
        if "soc_mwh" in normalized and normalized["soc_mwh"] is not None:
            normalized["soc_mwh"] = float(normalized["soc_mwh"])
        return normalized

    def apply_command(self, *, charge_mw: float, discharge_mw: float) -> dict[str, Any]:
        command = {
            "charge_mw": float(charge_mw),
            "discharge_mw": float(discharge_mw),
        }
        payload = self._request_json(method="POST", path=self.command_path, payload=command)
        out = dict(payload)
        out.setdefault("accepted", True)
        out.setdefault("violation", not bool(out.get("accepted", True)))
        out.setdefault("applied_charge_mw", float(command["charge_mw"]))
        out.setdefault("applied_discharge_mw", float(command["discharge_mw"]))
        return out
