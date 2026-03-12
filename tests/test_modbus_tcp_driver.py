from __future__ import annotations

import pytest

from iot.edge_agent.drivers.http_gateway import HTTPGatewayDriver
from iot.edge_agent.drivers.modbus_tcp import ModbusTCPDriver
from iot.edge_agent.run_agent import _build_driver


class _FakeReadResponse:
    def __init__(self, registers):
        self.registers = list(registers)

    def isError(self):
        return False


class _FakeWriteResponse:
    def isError(self):
        return False


class _FakeModbusClient:
    def __init__(self) -> None:
        self.read_map = {}
        self.writes = []
        self.connected = False

    def connect(self):
        self.connected = True
        return True

    def close(self):
        self.connected = False

    def read_holding_registers(self, *, address, count, slave):
        return _FakeReadResponse(self.read_map.get((address, count), [0] * count))

    def write_register(self, *, address, value, slave):
        self.writes.append(("single", address, value, slave))
        return _FakeWriteResponse()

    def write_registers(self, *, address, values, slave):
        self.writes.append(("multi", address, list(values), slave))
        return _FakeWriteResponse()


def test_modbus_driver_fetches_and_normalizes_telemetry() -> None:
    client = _FakeModbusClient()
    client.read_map[(0, 1)] = [250]
    client.read_map[(1, 1)] = [80]
    client.read_map[(2, 1)] = [123]
    client.read_map[(3, 1)] = [1_700_000_000]

    driver = ModbusTCPDriver(
        host="127.0.0.1",
        client=client,
        register_map={
            "load_mw": {"address": 0, "count": 1, "scale": 1.0},
            "renewables_mw": {"address": 1, "count": 1, "scale": 1.0},
            "soc_mwh": {"address": 2, "count": 1, "scale": 0.1},
            "timestamp_s": {"address": 3, "count": 1, "scale": 1.0},
        },
        command_registers={
            "charge_mw": {"address": 100, "count": 1, "scale": 0.1},
            "discharge_mw": {"address": 101, "count": 1, "scale": 0.1},
        },
    )
    out = driver.fetch_telemetry()
    assert out["load_mw"] == 250.0
    assert out["renewables_mw"] == 80.0
    assert out["soc_mwh"] == 12.3
    assert "ts_utc" in out


def test_modbus_driver_writes_scaled_commands() -> None:
    client = _FakeModbusClient()
    driver = ModbusTCPDriver(
        host="127.0.0.1",
        client=client,
        register_map={
            "load_mw": {"address": 0, "count": 1, "scale": 1.0},
            "wind_mw": {"address": 1, "count": 1, "scale": 1.0},
            "solar_mw": {"address": 2, "count": 1, "scale": 1.0},
        },
        command_registers={
            "charge_mw": {"address": 100, "count": 1, "scale": 0.1},
            "discharge_mw": {"address": 101, "count": 1, "scale": 0.1},
        },
    )
    out = driver.apply_command(charge_mw=1.5, discharge_mw=2.0)
    assert out["accepted"] is True
    assert client.writes == [
        ("single", 100, 15, 1),
        ("single", 101, 20, 1),
    ]


def test_modbus_driver_multi_register_write_mode() -> None:
    client = _FakeModbusClient()
    driver = ModbusTCPDriver(
        host="127.0.0.1",
        client=client,
        write_mode="multi_register",
        register_map={
            "load_mw": {"address": 0, "count": 1, "scale": 1.0},
            "renewables_mw": {"address": 1, "count": 1, "scale": 1.0},
        },
        command_registers={
            "charge_mw": {"address": 100, "count": 1, "scale": 1.0},
            "discharge_mw": {"address": 101, "count": 1, "scale": 1.0},
        },
    )
    driver.apply_command(charge_mw=3.0, discharge_mw=4.0)
    assert client.writes == [("multi", 100, [3, 4], 1)]


def test_build_driver_defaults_to_http_gateway() -> None:
    driver = _build_driver(
        {
            "driver": {"kind": "http_gateway", "http_gateway": {"base_url": "http://gw.local"}},
            "gateway": {"base_url": "http://legacy.local"},
        }
    )
    assert isinstance(driver, HTTPGatewayDriver)
    assert driver.base_url == "http://gw.local"


def test_modbus_driver_requires_command_registers() -> None:
    client = _FakeModbusClient()
    driver = ModbusTCPDriver(host="127.0.0.1", client=client, register_map={}, command_registers={})
    with pytest.raises(RuntimeError, match="command_registers"):
        driver.apply_command(charge_mw=1.0, discharge_mw=0.0)
