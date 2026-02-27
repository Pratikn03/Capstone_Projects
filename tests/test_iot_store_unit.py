import pytest
from gridpulse.iot.store import IoTLoopStore

@pytest.fixture
def store(tmp_path):
    store_obj = IoTLoopStore(duckdb_path=":memory:")
    yield store_obj
    store_obj.close()
    
def test_record_telemetry(store):
    store.record_telemetry(
        device_id="d1",
        ts_utc="2023-01-01T12:00:00Z",
        payload={"load": 100},
        reliability_w=1.0,
        reliability_flags={}
    )
    state = store.get_state("d1")
    assert state is not None
    assert state["latest_ts_utc"] == "2023-01-01T12:00:00Z"
    assert state["latest_reliability_w"] == 1.0

def test_enqueue_command(store):
    cmd_id = "cmd-1"
    store.enqueue_command(
        device_id="d1",
        zone_id="DE",
        command_id=cmd_id,
        command={"setpoint": 200},
        certificate_id="cert-123"
    )
    assert cmd_id is not None
    
    # Check queue
    next_cmd = store.get_next_command(device_id="d1")
    assert next_cmd is not None
    assert next_cmd["command_id"] == cmd_id
    assert next_cmd["command"]["setpoint"] == 200
    assert next_cmd["status"] == "dispatched"

def test_record_ack(store):
    cmd_id = "cmd-2"
    store.enqueue_command(
        device_id="d1",
        zone_id="DE",
        command_id=cmd_id,
        command={"setpoint": 200},
    )
    store.get_next_command(device_id="d1") # dispatch it
    
    store.record_ack(
        device_id="d1",
        command_id=cmd_id,
        status="success",
        reason="ok",
        payload={"acked": True}
    )
    
    state = store.get_state("d1")
    assert state["last_ack"]["status"] == "success"

def test_expire_stale_commands(store):
    store.enqueue_command(
        device_id="d1",
        zone_id="DE",
        command_id="cmd-expire",
        command={"setpoint": 200},
        ttl_seconds=-10 # Already expired
    )
    expired_count = store.expire_stale_commands(device_id="d1")
    assert expired_count == 1
    
    # Queue should be empty now
    assert store.get_next_command(device_id="d1") is None

def test_hold_state(store):
    # Setup state
    store.record_telemetry(device_id="d1", ts_utc="2023-01-01", payload={}, reliability_w=1.0, reliability_flags={})
    
    # Enter hold
    store._state_upsert(device_id="d1", hold_active=True, hold_reason="maintenance")
    state = store.get_state("d1")
    assert state["hold_active"] is True
    assert state["hold_reason"] == "maintenance"
    
    # Reset hold
    store.reset_hold(device_id="d1")
    state2 = store.get_state("d1")
    assert state2["hold_active"] is False
    assert state2["hold_reason"] is None
