"""Tests for figure inventory audit script."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import scripts.audit_figure_inventory as fig_audit


def _write(path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_build_inventory_marks_critical_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fig_audit, "REPO_ROOT", tmp_path)
    (tmp_path / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    _write(tmp_path / "reports" / "figures" / "demo.png", b"png")
    _write(tmp_path / "reports" / "publication" / "dc3s_main_table.csv", b"csv")

    run_start = datetime.now(timezone.utc) - timedelta(hours=1)
    lock_time = datetime.now(timezone.utc) - timedelta(days=1)
    payload = fig_audit.build_inventory(run_start_utc=run_start, manifest_lock_utc=lock_time)

    assert payload["summary"]["files_total"] >= 2
    assert payload["summary"]["critical_missing"] >= 1
    assert payload["summary"]["critical_ok"] is False


def test_audit_figure_inventory_main_writes_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fig_audit, "REPO_ROOT", tmp_path)
    manifest = tmp_path / "paper" / "metrics_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"generated_at_utc": "2026-02-19T00:00:00Z"}), encoding="utf-8")

    required = [
        "dc3s_main_table.csv",
        "dc3s_fault_breakdown.csv",
        "calibration_plot.png",
        "violation_vs_cost_curve.png",
        "dc3s_run_summary.json",
    ]
    for name in required:
        _write(tmp_path / "reports" / "publication" / name, b"ok")
    _write(tmp_path / "reports" / "figures" / "impact_savings.png", b"ok")

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_figure_inventory.py",
            "--out-dir",
            "reports/publish",
            "--manifest",
            "paper/metrics_manifest.json",
            "--run-start-utc",
            datetime.now(timezone.utc).isoformat(),
        ],
    )
    fig_audit.main()

    out_json = tmp_path / "reports" / "publish" / "figure_inventory.json"
    out_md = tmp_path / "reports" / "publish" / "figure_freshness_report.md"
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary"]["critical_missing"] == 0
    assert payload["summary"]["critical_zero_size"] == 0
    assert payload["summary"]["critical_ok"] is True
