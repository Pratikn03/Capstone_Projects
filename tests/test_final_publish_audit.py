"""Tests for final publish audit orchestrator."""

from __future__ import annotations

import json
import sys

import scripts.final_publish_audit as fpa


def test_final_publish_audit_writes_outputs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpa, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    (tmp_path / "configs" / "publish_audit.yaml").write_text(
        """
publish_audit:
  reproducibility:
    out_dir: reports/publish
    run_id_prefix: publish
  go_no_go:
    safety_violations: 0
    certificate_completeness_rate_min: 0.99
    ack_success_rate_min: 0.99
    hold_rate_max: 0.01
    require_no_critical_alerts: true
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
    (tmp_path / "reports" / "monitoring_summary.json").write_text(
        json.dumps(
            {
                "data_drift": {"drift": False},
                "model_drift": {"decision": {"drift": False}},
                "dc3s_health": {"triggered": False},
                "retraining": {"retrain": False, "reasons": []},
            }
        ),
        encoding="utf-8",
    )

    def fake_run_step(*, name, cmd, logs_dir, env=None, timeout_s=None):
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{name}.log"
        log_path.write_text("ok", encoding="utf-8")
        output = ""
        if name == "refresh_data_delta":
            output = json.dumps({"ok": True})
        elif name == "audit_na_tables":
            output = json.dumps({"violations": 0, "fail": False})
        elif name == "audit_leakage" or name == "audit_code_health":
            output = json.dumps({"fail": False, "violations": []})
        elif name == "iot_sim_nominal":
            output = json.dumps(
                {
                    "safety_violations": 0,
                    "certificate_completeness_rate": 1.0,
                    "commands_processed": 10,
                    "interventions": 0,
                }
            )
        step = fpa.StepResult(
            name=name,
            ok=True,
            return_code=0,
            duration_s=0.1,
            command=cmd,
            log_path=str(log_path),
        )
        return step, output

    monkeypatch.setattr(fpa, "_run_step", fake_run_step)
    monkeypatch.setattr(
        fpa,
        "_compute_iot_ack_hold_metrics",
        lambda _db_path: {
            "ack_success_rate": 1.0,
            "hold_rate": 0.0,
            "db_exists": True,
            "acks_total": 10,
            "holds_total": 0,
            "commands_total": 10,
        },
    )
    monkeypatch.setattr(
        fpa,
        "_compute_dc3s_typed_readiness",
        lambda _db, _table: {
            "exists": True,
            "rows_total": 10,
            "null_ratios": {
                "intervened": 0.0,
                "intervention_reason": 0.0,
                "reliability_w": 0.0,
                "drift_flag": 0.0,
                "inflation": 0.0,
            },
        },
    )

    monkeypatch.setattr(sys, "argv", ["final_publish_audit.py", "--config", "configs/publish_audit.yaml"])
    fpa.main()

    out_dir = tmp_path / "reports" / "publish"
    assert (out_dir / "final_audit_report.md").exists()
    assert (out_dir / "final_audit_report.json").exists()
    decision = json.loads((out_dir / "go_no_go_decision.json").read_text(encoding="utf-8"))
    assert decision["go"] is True


def test_final_publish_audit_wires_new_cli_args(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpa, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    (tmp_path / "configs" / "publish_audit.yaml").write_text(
        """
publish_audit:
  reproducibility:
    out_dir: reports/publish
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
    (tmp_path / "reports" / "monitoring_summary.json").write_text(
        json.dumps(
            {
                "data_drift": {"drift": False},
                "model_drift": {"decision": {"drift": False}},
                "dc3s_health": {"triggered": False},
            }
        ),
        encoding="utf-8",
    )

    seen_commands: dict[str, list[str]] = {}

    def fake_run_step(*, name, cmd, logs_dir, env=None, timeout_s=None):
        seen_commands[name] = list(cmd)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{name}.log"
        log_path.write_text("ok", encoding="utf-8")
        if name == "audit_git_delta":
            (tmp_path / "reports" / "publish" / "github_delta_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "total_changed_files": 1,
                            "in_scope_files": 1,
                            "out_of_scope_files": 0,
                            "added_lines": 1,
                            "deleted_lines": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
        if name == "audit_figure_inventory":
            (tmp_path / "reports" / "publish" / "figure_inventory.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "files_total": 1,
                            "critical_missing": 0,
                            "critical_zero_size": 0,
                            "critical_ok": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
        output = (
            json.dumps({"ok": True})
            if name in {"refresh_data_delta", "audit_na_tables", "audit_leakage", "audit_code_health"}
            else ""
        )
        if name == "iot_sim_nominal":
            output = json.dumps({"safety_violations": 0, "certificate_completeness_rate": 1.0})
        step = fpa.StepResult(
            name=name, ok=True, return_code=0, duration_s=0.1, command=cmd, log_path=str(log_path)
        )
        return step, output

    monkeypatch.setattr(fpa, "_run_step", fake_run_step)
    monkeypatch.setattr(
        fpa, "_compute_iot_ack_hold_metrics", lambda _db_path: {"ack_success_rate": 1.0, "hold_rate": 0.0}
    )
    monkeypatch.setattr(
        fpa,
        "_compute_dc3s_typed_readiness",
        lambda _db, _table: {"exists": True, "rows_total": 1, "null_ratios": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "final_publish_audit.py",
            "--config",
            "configs/publish_audit.yaml",
            "--baseline-ref",
            "origin/main",
            "--max-runtime-hours",
            "6",
            "--iot-steps",
            "72",
        ],
    )
    fpa.main()

    assert "--baseline-ref" in seen_commands["audit_git_delta"]
    assert "origin/main" in seen_commands["audit_git_delta"]
    assert "--max-runtime-hours" in seen_commands["retrain_de_aggressive"]
    assert "6.0" in seen_commands["retrain_de_aggressive"]
    assert "--steps" in seen_commands["iot_sim_nominal"]
    assert "72" in seen_commands["iot_sim_nominal"]
