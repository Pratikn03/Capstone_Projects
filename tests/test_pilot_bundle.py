from __future__ import annotations

import json
from pathlib import Path

from gridpulse.dc3s.certificate import make_certificate, store_certificate
from gridpulse.iot.store import IoTLoopStore
from scripts.export_pilot_bundle import export_bundle
from scripts.summarize_chil_run import summarize_run
from scripts.validate_pilot_bundle import validate_bundle


def _seed_iot_and_cert(tmp_path: Path) -> tuple[Path, Path]:
    iot_db = tmp_path / "iot.duckdb"
    audit_db = tmp_path / "audit.duckdb"
    store = IoTLoopStore(str(iot_db))
    try:
        store.record_telemetry(
            device_id="edge-1",
            ts_utc="2026-03-12T12:00:00Z",
            payload={"ts_utc": "2026-03-12T12:00:00Z", "load_mw": 50.0, "renewables_mw": 10.0, "soc_mwh": 5.0},
            reliability_w=0.85,
            reliability_flags={"stale": False},
        )
        store.enqueue_command(
            device_id="edge-1",
            command_id="cmd-001",
            zone_id="DE",
            certificate_id="cmd-001",
            command={"safe_action": {"charge_mw": 0.0, "discharge_mw": 1.0}},
        )
        store.get_next_command(device_id="edge-1", peek=False)
        store.record_ack(
            device_id="edge-1",
            command_id="cmd-001",
            certificate_id="cmd-001",
            status="acked",
            payload={"accepted": True, "violation": False},
        )
    finally:
        store.close()

    cert = make_certificate(
        command_id="cmd-001",
        device_id="edge-1",
        zone_id="DE",
        controller="dc3s_ftit",
        proposed_action={"charge_mw": 0.0, "discharge_mw": 1.0},
        safe_action={"charge_mw": 0.0, "discharge_mw": 1.0},
        uncertainty={"lower": 1.0, "upper": 2.0},
        reliability={"flags": {"stale": False}},
        drift={"flag": False},
        model_hash="m",
        config_hash="c",
        intervened=True,
        reliability_w=0.85,
        guarantee_checks_passed=True,
    )
    store_certificate(cert, duckdb_path=str(audit_db), table_name="dispatch_certificates")
    return iot_db, audit_db


def test_summarize_chil_run(tmp_path) -> None:
    iot_db, audit_db = _seed_iot_and_cert(tmp_path)
    agent_log = tmp_path / "agent.log"
    agent_log.write_text(
        "\n".join(
            [
                json.dumps({"status": "ok", "ack_status": "acked"}),
                json.dumps({"status": "empty"}),
                json.dumps({"status": "hold"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = summarize_run(
        iot_db_path=str(iot_db),
        audit_db_path=str(audit_db),
        device_id="edge-1",
        agent_log_path=str(agent_log),
    )
    assert summary["telemetry_events"] == 1
    assert summary["acks"] == 1
    assert summary["empty_command_events"] == 1
    assert summary["certificate_completeness"] == 1.0
    assert summary["intervention_rate_by_reliability_bucket"]


def test_export_and_validate_pilot_bundle(tmp_path) -> None:
    iot_db, audit_db = _seed_iot_and_cert(tmp_path)
    bundle_dir = export_bundle(
        out_dir=str(tmp_path / "bundles"),
        iot_db_path=str(iot_db),
        audit_db_path=str(audit_db),
        device_id="edge-1",
    )
    result = validate_bundle(str(bundle_dir))
    assert result["passed"] is True
    assert result["command_rows"] == 1
    assert result["certificate_rows"] == 1
