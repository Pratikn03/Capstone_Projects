"""Protocol-ready Modbus TCP driver for future pilot integrations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass
class ModbusTCPDriver:
    """Minimal Modbus TCP adapter with injectable client for testability."""

    host: str
    port: int = 502
    unit_id: int = 1
    timeout_s: float = 3.0
    register_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    command_registers: dict[str, dict[str, Any]] = field(default_factory=dict)
    write_mode: str = "single_register"
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            try:
                from pymodbus.client import ModbusTcpClient  # type: ignore
            except Exception as exc:  # pragma: no cover - exercised via tests with injected client
                raise RuntimeError(
                    "ModbusTCPDriver requires pymodbus or an injected client for testing."
                ) from exc
            self.client = ModbusTcpClient(host=self.host, port=int(self.port), timeout=float(self.timeout_s))
        if hasattr(self.client, "connect"):
            connected = self.client.connect()
            if connected is False:
                raise RuntimeError(f"Failed to connect Modbus TCP client to {self.host}:{self.port}")

    def close(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()

    def _read_registers(self, spec: Mapping[str, Any]) -> list[int]:
        address = _i(spec.get("address"), 0)
        count = max(_i(spec.get("count"), 1), 1)
        response = self.client.read_holding_registers(address=address, count=count, slave=int(self.unit_id))
        if hasattr(response, "isError") and response.isError():
            raise RuntimeError(f"Modbus read failed for address {address}")
        registers = list(getattr(response, "registers", []) or [])
        if len(registers) < count:
            raise RuntimeError(f"Modbus read returned insufficient registers for address {address}")
        return registers

    def _decode_registers(self, registers: list[int], spec: Mapping[str, Any]) -> float:
        signed = bool(spec.get("signed", False))
        scale = _f(spec.get("scale"), 1.0)
        offset = _f(spec.get("offset"), 0.0)
        raw = 0
        for reg in registers:
            raw = (raw << 16) | (_i(reg) & 0xFFFF)
        bits = 16 * len(registers)
        if signed and bits > 0 and raw >= 1 << (bits - 1):
            raw -= 1 << bits
        return float(raw) * scale + offset

    def _encode_value(self, value: float, spec: Mapping[str, Any]) -> list[int]:
        count = max(_i(spec.get("count"), 1), 1)
        scale = _f(spec.get("scale"), 1.0)
        offset = _f(spec.get("offset"), 0.0)
        if scale == 0:
            raise RuntimeError("Modbus register scale must be non-zero")
        raw = int(round((float(value) - offset) / scale))
        if raw < 0:
            raw &= (1 << (16 * count)) - 1
        registers = []
        for idx in range(count):
            shift = 16 * (count - idx - 1)
            registers.append((raw >> shift) & 0xFFFF)
        return registers

    def fetch_telemetry(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field_name, spec in self.register_map.items():
            registers = self._read_registers(spec)
            payload[field_name] = self._decode_registers(registers, spec)

        if "renewables_mw" not in payload:
            wind = _f(payload.get("wind_mw"), 0.0)
            solar = _f(payload.get("solar_mw"), 0.0)
            if "wind_mw" in payload or "solar_mw" in payload:
                payload["renewables_mw"] = wind + solar

        ts_seconds = payload.pop("timestamp_s", None)
        if ts_seconds is not None:
            payload["ts_utc"] = datetime.fromtimestamp(float(ts_seconds), tz=timezone.utc).isoformat()
        else:
            payload["ts_utc"] = datetime.now(timezone.utc).isoformat()

        if "load_mw" not in payload:
            raise RuntimeError("Modbus telemetry missing required field: load_mw")
        if "renewables_mw" not in payload:
            raise RuntimeError("Modbus telemetry missing required field: renewables_mw")
        payload["load_mw"] = float(payload["load_mw"])
        payload["renewables_mw"] = float(payload["renewables_mw"])
        if "soc_mwh" in payload:
            payload["soc_mwh"] = float(payload["soc_mwh"])
        return payload

    def _write_single(self, *, address: int, value: int) -> None:
        response = self.client.write_register(address=address, value=value, slave=int(self.unit_id))
        if hasattr(response, "isError") and response.isError():
            raise RuntimeError(f"Modbus write failed for address {address}")

    def _write_many(self, *, address: int, values: list[int]) -> None:
        response = self.client.write_registers(address=address, values=values, slave=int(self.unit_id))
        if hasattr(response, "isError") and response.isError():
            raise RuntimeError(f"Modbus block write failed for address {address}")

    def apply_command(self, *, charge_mw: float, discharge_mw: float) -> dict[str, Any]:
        if "charge_mw" not in self.command_registers or "discharge_mw" not in self.command_registers:
            raise RuntimeError("Modbus command_registers must define charge_mw and discharge_mw")

        charge_spec = self.command_registers["charge_mw"]
        discharge_spec = self.command_registers["discharge_mw"]
        charge_regs = self._encode_value(float(charge_mw), charge_spec)
        discharge_regs = self._encode_value(float(discharge_mw), discharge_spec)

        if self.write_mode == "multi_register":
            charge_addr = _i(charge_spec.get("address"), 0)
            discharge_addr = _i(discharge_spec.get("address"), 0)
            contiguous = discharge_addr == charge_addr + len(charge_regs)
            if contiguous:
                self._write_many(address=charge_addr, values=charge_regs + discharge_regs)
            else:
                for idx, value in enumerate(charge_regs):
                    self._write_single(address=charge_addr + idx, value=value)
                for idx, value in enumerate(discharge_regs):
                    self._write_single(address=discharge_addr + idx, value=value)
        else:
            for idx, value in enumerate(charge_regs):
                self._write_single(address=_i(charge_spec.get("address"), 0) + idx, value=value)
            for idx, value in enumerate(discharge_regs):
                self._write_single(address=_i(discharge_spec.get("address"), 0) + idx, value=value)

        return {
            "accepted": True,
            "violation": False,
            "applied_charge_mw": float(charge_mw),
            "applied_discharge_mw": float(discharge_mw),
            "driver": "modbus_tcp",
        }
