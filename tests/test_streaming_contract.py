"""Streaming schema contract tests."""
from gridpulse.streaming.schemas import OPSDTelemetryEvent


def test_opsd_contract_minimal():
    evt = OPSDTelemetryEvent(utc_timestamp="2020-01-01T00:00:00Z")
    assert evt.utc_timestamp.startswith("2020-01-01")


def test_opsd_contract_with_fields():
    evt = OPSDTelemetryEvent(
        utc_timestamp="2020-01-01T01:00:00Z",
        DE_load_actual_entsoe_transparency=50000.0,
        DE_wind_generation_actual=10000.0,
        DE_solar_generation_actual=2000.0,
    )
    payload = evt.model_dump()
    assert payload["DE_load_actual_entsoe_transparency"] == 50000.0
