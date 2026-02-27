import pytest

from gridpulse.dc3s.state import DC3SStateStore


def test_dc3s_state_store_roundtrip(tmp_path):
    db_path = str(tmp_path / "dc3s_state.duckdb")
    store = DC3SStateStore(db_path)
    
    try:
        # 1. State should be empty initially
        res = store.get("Z1", "D1", "T1")
        assert res is None
        
        # 2. Upsert a new state
        drift_state = {"ph": {"cumulative_sum": 5.0, "count": 10}}
        adaptive_state = {"window": [1.0, 2.0]}
        last_event = {"val": 100}
        last_action = {"charge_mw": 50.0}
        
        store.upsert(
            zone_id="Z1",
            device_id="D1",
            target="T1",
            last_timestamp="2026-01-01T00:00:00Z",
            last_yhat=105.0,
            last_y_true=100.0,
            drift_state=drift_state,
            adaptive_state=adaptive_state,
            last_prev_hash="hash123",
            last_inflation=1.2,
            last_event=last_event,
            last_action=last_action,
        )
        
        # 3. Retrieve and verify
        res = store.get("Z1", "D1", "T1")
        assert res is not None
        assert res["last_timestamp"] == "2026-01-01T00:00:00Z"
        assert res["last_yhat"] == 105.0
        assert res["last_y_true"] == 100.0
        assert res["drift_state"] == drift_state
        assert res["adaptive_state"] == adaptive_state
        assert res["last_prev_hash"] == "hash123"
        assert res["last_inflation"] == 1.2
        assert res["last_event"] == last_event
        assert res["last_action"] == last_action
        
        # 4. Verify different device_id is empty
        assert store.get("Z1", "D2", "T1") is None
        
        # 5. Overwrite state (upsert does REPLACE)
        store.upsert(
            zone_id="Z1",
            device_id="D1",
            target="T1",
            last_inflation=2.5,
        )
        res = store.get("Z1", "D1", "T1")
        assert res["last_inflation"] == 2.5
        assert res["last_yhat"] is None # It was replaced with None
        
    finally:
        store.close()
