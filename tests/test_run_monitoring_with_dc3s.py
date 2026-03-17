"""Integration-style test for monitoring summary with DC3S health block."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from orius.dc3s.certificate import make_certificate, store_certificate
import scripts.run_monitoring as run_monitoring


def _seed_cert(db_path: Path, command_id: str, *, w: float, drift: bool, infl: float, intervened: bool) -> None:
    cert = make_certificate(
        command_id=command_id,
        device_id="dev-1",
        zone_id="DE",
        controller="deterministic",
        proposed_action={"charge_mw": 1.0, "discharge_mw": 0.0},
        safe_action={"charge_mw": 0.0, "discharge_mw": 0.0},
        uncertainty={"lower": [1.0], "upper": [2.0], "meta": {"inflation": infl}},
        reliability={"w_t": w, "flags": {}},
        drift={"drift": drift},
        model_hash="m",
        config_hash="c",
        intervened=intervened,
        intervention_reason="projection_clip" if intervened else None,
        reliability_w=w,
        drift_flag=drift,
        inflation=infl,
    )
    store_certificate(cert, duckdb_path=str(db_path), table_name="dispatch_certificates")


def test_run_monitoring_writes_dc3s_health_and_retraining_reason(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    (tmp_path / "configs" / "monitoring.yaml").write_text(
        """
data_drift:
  p_value_threshold: 0.01
model_drift:
  metric: mape
  degradation_threshold: 0.15
retraining:
  cadence_days: 999
dc3s_health:
  enabled: true
  lookback_hours: 24
  min_commands: 1
  intervention_rate_threshold: 0.30
  low_reliability_w_threshold: 0.60
  low_reliability_rate_threshold: 0.25
  drift_flag_rate_threshold: 0.10
  inflation_p95_threshold: 2.0
  sustained_windows: 1
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "dc3s.yaml").write_text(
        """
dc3s:
  audit:
    duckdb_path: data/audit/dc3s_audit.duckdb
    table_name: dispatch_certificates
""".strip(),
        encoding="utf-8",
    )

    db_path = tmp_path / "data" / "audit" / "dc3s_audit.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_cert(db_path, "c1", w=0.30, drift=True, infl=2.6, intervened=True)
    _seed_cert(db_path, "c2", w=0.40, drift=True, infl=2.4, intervened=True)
    _seed_cert(db_path, "c3", w=0.50, drift=True, infl=2.3, intervened=True)

    train_df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=12, freq="h"), "x": range(12), "load_mw": range(12)})
    test_df = pd.DataFrame({"timestamp": pd.date_range("2026-01-02", periods=12, freq="h"), "x": range(12), "load_mw": range(12)})
    monkeypatch.setattr(run_monitoring, "_load_split", lambda: (train_df, test_df))
    monkeypatch.setattr(run_monitoring, "compute_data_drift", lambda *args, **kwargs: {"drift": False, "columns": {}})

    monkeypatch.setattr(sys, "argv", ["run_monitoring.py", "--disable-alerts"])
    run_monitoring.main()

    summary_path = tmp_path / "reports" / "monitoring_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "dc3s_health" in payload
    dc3s_health = payload["dc3s_health"]
    assert dc3s_health["commands_total"] == 3
    assert dc3s_health["triggered"] is True
    assert "intervention_rate" in set(dc3s_health["triggered_flags"])

    retraining = payload["retraining"]
    assert retraining["retrain"] is True
    assert "dc3s_intervention_spike" in retraining["reasons"]
