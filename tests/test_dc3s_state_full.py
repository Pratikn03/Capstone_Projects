"""Comprehensive tests for DC3S state persistence."""

from __future__ import annotations

import pytest

from orius.dc3s.state import DC3SStateStore


class TestDC3SStateStore:
    def test_constructor_creates_db(self, tmp_path):
        db = str(tmp_path / "state.duckdb")
        store = DC3SStateStore(db)
        assert (tmp_path / "state.duckdb").exists()
        store.close()

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "s.duckdb"))
        assert store.get("z", "d", "t") is None
        store.close()

    def test_upsert_then_get_round_trip(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "s.duckdb"))
        store.upsert(
            zone_id="DE",
            device_id="batt-1",
            target="load_mw",
            last_timestamp="2026-01-01T00:00:00Z",
            last_yhat=45000.0,
            last_y_true=45200.0,
            drift_state={"count": 10, "mean": 200.0},
            adaptive_state={"ftit": {"n": 5.0}},
            last_prev_hash="abc123",
            last_inflation=1.3,
            last_event={"load_mw": 45200.0},
            last_action={"charge_mw": 5.0},
        )
        state = store.get("DE", "batt-1", "load_mw")
        assert state is not None
        assert state["last_timestamp"] == "2026-01-01T00:00:00Z"
        assert state["last_yhat"] == pytest.approx(45000.0)
        assert state["last_y_true"] == pytest.approx(45200.0)
        assert state["drift_state"]["count"] == 10
        assert state["adaptive_state"]["ftit"]["n"] == 5.0
        assert state["last_prev_hash"] == "abc123"
        assert state["last_inflation"] == pytest.approx(1.3)
        assert state["last_event"]["load_mw"] == 45200.0
        assert state["last_action"]["charge_mw"] == 5.0
        store.close()

    def test_upsert_overwrites(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "s.duckdb"))
        store.upsert(zone_id="DE", device_id="b", target="t", last_yhat=1.0)
        store.upsert(zone_id="DE", device_id="b", target="t", last_yhat=2.0)
        state = store.get("DE", "b", "t")
        assert state["last_yhat"] == pytest.approx(2.0)
        store.close()

    def test_multiple_zones_coexist(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "s.duckdb"))
        store.upsert(zone_id="DE", device_id="b1", target="t", last_yhat=10.0)
        store.upsert(zone_id="US", device_id="b1", target="t", last_yhat=20.0)
        assert store.get("DE", "b1", "t")["last_yhat"] == pytest.approx(10.0)
        assert store.get("US", "b1", "t")["last_yhat"] == pytest.approx(20.0)
        store.close()

    def test_state_key_format(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "s.duckdb"))
        key = store._key("zone", "dev", "tgt")
        assert key == "zone:dev:tgt"
        store.close()

    def test_none_event_and_action(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "s.duckdb"))
        store.upsert(zone_id="z", device_id="d", target="t")
        state = store.get("z", "d", "t")
        assert state["last_event"] is None
        assert state["last_action"] is None
        store.close()

    def test_numeric_fields_preserved(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "s.duckdb"))
        store.upsert(
            zone_id="z",
            device_id="d",
            target="t",
            last_yhat=1.23456789,
            last_y_true=9.87654321,
            last_inflation=2.5,
        )
        state = store.get("z", "d", "t")
        assert state["last_yhat"] == pytest.approx(1.23456789)
        assert state["last_y_true"] == pytest.approx(9.87654321)
        assert state["last_inflation"] == pytest.approx(2.5)
        store.close()

    def test_close_is_safe(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "s.duckdb"))
        store.close()

    def test_custom_table_name(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "s.duckdb"), table_name="my_states")
        store.upsert(zone_id="z", device_id="d", target="t", last_yhat=42.0)
        assert store.get("z", "d", "t")["last_yhat"] == pytest.approx(42.0)
        store.close()
