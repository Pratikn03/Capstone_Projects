from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import build_three_domain_runtime_stress_artifacts as stress
from scripts import run_three_domain_offline_freeze as freeze
from scripts.validate_nuplan_freeze_gate import validate as validate_nuplan_freeze


def test_offline_freeze_dry_run_plans_training_and_nuplan_av_gate(tmp_path: Path) -> None:
    result = freeze.run_freeze(
        release_id="TEST_FREEZE",
        out_dir=tmp_path,
        profile="aggressive",
        max_runtime_hours=12.0,
        dry_run=True,
    )
    planned = json.loads((tmp_path / "TEST_FREEZE" / "planned_commands.json").read_text(encoding="utf-8"))

    assert result["dry_run"] is True
    assert set(planned["training"]) == {"DE", "AV", "HEALTHCARE"}
    assert planned["model_request"] == "gbm,lstm,tcn,nbeats,tft,patchtst"
    assert planned["nuplan_full_gate"].endswith("nuplan_full_av_gate.json")

    for dataset in ("DE", "HEALTHCARE"):
        cmd = planned["training"][dataset]
        assert cmd[1:3] == ["scripts/train_dataset.py", "--dataset"]
        assert dataset in cmd
        assert "--candidate-run" in cmd
        assert "--run-id" in cmd
        assert "--models" in cmd
        assert "gbm,lstm,tcn,nbeats,tft,patchtst" in cmd
        assert "--profile" in cmd
        assert "aggressive" in cmd
        assert "--promote-on-accept" not in cmd

    av_command = planned["training"]["AV"]
    assert av_command[1] == "scripts/validate_nuplan_freeze_gate.py"
    assert "--summary" in av_command
    assert "reports/predeployment_external_validation/nuplan_closed_loop_summary.csv" in av_command
    assert "--traces" in av_command
    assert "reports/predeployment_external_validation/nuplan_closed_loop_traces.csv" in av_command
    assert "--manifest" in av_command
    assert "reports/predeployment_external_validation/nuplan_closed_loop_manifest.json" in av_command
    assert "--manifest-out" in av_command
    assert av_command[-1].endswith("nuplan_full_av_gate.json")

    for dataset in ("DE", "HEALTHCARE"):
        cmd = planned["promotion"][dataset]
        assert "--promote-on-accept" in cmd
        assert "--reports-only" in cmd
    assert "--promote-on-accept" not in planned["promotion"]["AV"]
    downstream_text = "\n".join(" ".join(cmd) for cmd in planned["downstream"])
    assert "build_nuplan_closed_loop_artifacts.py" in downstream_text
    assert "build_three_domain_runtime_stress_artifacts.py" in downstream_text
    assert "run_runtime_dry_run" not in downstream_text
    assert "synthetic_stress" not in downstream_text


def test_full_av_training_gate_reads_nuplan_manifest(tmp_path: Path) -> None:
    release_dir = tmp_path / "TEST_FULL"
    release_dir.mkdir(parents=True)
    gate = release_dir / "nuplan_full_av_gate.json"
    gate.write_text(
        json.dumps(
            {
                "pass": True,
                "source_dataset": "nuplan_singapore",
                "validation_surface": "nuplan_allzip_grouped_runtime_replay_surrogate",
                "primary_target": "nuplan_allzip_grouped_runtime_replay_surrogate",
                "orius_runtime_rows": 1_531_104,
                "trace_rows": 12_248_832,
                "orius_tsvr": 0.0001626277509561728,
                "certificate_valid_rate": 0.9999242376742533,
                "domain_postcondition_pass_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )

    row = freeze.training_gate(freeze.FREEZE_DOMAINS[1], "TEST_FULL", freeze_dir=release_dir)

    assert row["pass"] is True
    assert row["primary_target"] == "nuplan_allzip_grouped_runtime_replay_surrogate"
    assert row["nuplan_source_dataset"] == "nuplan_singapore"
    assert row["nuplan_trace_rows"] == 12_248_832


def test_nuplan_freeze_gate_accepts_legacy_trace_without_surface_column(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    traces = tmp_path / "traces.csv"
    source_manifest = tmp_path / "manifest.json"
    manifest_out = tmp_path / "gate.json"

    summary.write_text(
        "\n".join(
            [
                "validation_surface,status,source_dataset,orius_runtime_rows,orius_tsvr,certificate_valid_rate,domain_postcondition_pass_rate,carla_completed,road_deployed,full_autonomous_driving_closure_claimed,claim_boundary",
                "nuplan_allzip_grouped_runtime_replay_surrogate,completed_bounded_replay_not_carla,nuPlan,2,0.0,1.0,1.0,False,False,False,does not claim completed CARLA simulation road deployment full autonomous-driving field closure",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    traces.write_text(
        "\n".join(
            [
                "scenario_id,controller,true_constraint_violated",
                "s1,orius,False",
                "s1,baseline,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_manifest.write_text('{"status": "completed_bounded_replay_not_carla"}\n', encoding="utf-8")

    result = validate_nuplan_freeze(
        summary_path=summary,
        traces_path=traces,
        source_manifest_path=source_manifest,
        manifest_out=manifest_out,
        min_runtime_rows=2,
        min_trace_rows=2,
    )

    assert result["pass"] is True
    assert result["trace_stats"]["validation_surfaces"] == []


def test_uncapped_freeze_plan_omits_training_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(freeze, "_active_nuplan_writers", lambda: [])
    result = freeze.run_freeze(
        release_id="TEST_FREEZE_UNCAPPED",
        out_dir=tmp_path,
        profile="max",
        max_runtime_hours=0.0,
        dry_run=True,
    )
    planned = json.loads((tmp_path / "TEST_FREEZE_UNCAPPED" / "planned_commands.json").read_text(encoding="utf-8"))
    preflight = json.loads((tmp_path / "TEST_FREEZE_UNCAPPED" / "preflight_manifest.json").read_text(encoding="utf-8"))

    assert result["dry_run"] is True
    assert planned["runtime_cap_enabled"] is False
    assert planned["runtime_cap_hours"] is None
    assert preflight["runtime_cap_policy"] == "uncapped"
    for dataset in ("DE", "HEALTHCARE"):
        assert "--max-runtime-hours" not in planned["training"][dataset]


def test_runtime_stress_artifacts_are_real_runtime_fault_summaries(tmp_path: Path) -> None:
    manifest = stress.build_runtime_stress_artifacts(tmp_path)
    summary = pd.read_csv(tmp_path / "runtime_stress_summary.csv")
    traces = pd.read_csv(tmp_path / "runtime_stress_traces.csv")

    assert manifest["all_passed"] is True
    assert manifest["status"] == "real_runtime_stress_not_deployment"
    assert set(summary["domain"]) == {
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    }
    assert {"dropout", "stale_sensor"} <= set(summary.loc[summary["domain_key"] == "battery", "stress_family"])
    assert {"delay_jitter", "dropout", "out_of_order", "spikes", "stale"} <= set(
        summary.loc[summary["domain_key"] == "av", "stress_family"]
    )
    assert {"blackout", "noise", "stuck_sensor"} <= set(summary.loc[summary["domain_key"] == "healthcare", "stress_family"])
    assert not summary["synthetic_source"].astype(bool).any()
    assert not summary["proxy_source"].astype(bool).any()
    assert not summary["validation_harness_source"].astype(bool).any()
    assert summary["stress_gate_pass"].astype(bool).all()
    assert not traces["synthetic_source"].astype(bool).any()


def test_predeployment_manifest_status_and_hash_lock_are_verifiable(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("locked\n", encoding="utf-8")
    hash_rows = [
        {
            "domain": "three_domain",
            "artifact_class": "unit_test",
            "path": str(artifact),
            "sha256": freeze._sha256(artifact),
            "size_bytes": artifact.stat().st_size,
            "mtime_utc": "2026-04-21T00:00:00+00:00",
        }
    ]
    hash_paths = freeze.write_hash_lock(tmp_path, hash_rows)
    plans = freeze.planned_commands(
        "TEST_FREEZE",
        tmp_path,
        profile="aggressive",
        max_runtime_hours=12.0,
    )
    training_gates = [
        {"domain": domain.domain_label, "dataset": domain.dataset, "pass": True}
        for domain in freeze.FREEZE_DOMAINS
    ]
    manifest = freeze.write_release_manifest(
        release_id="TEST_FREEZE",
        freeze_dir=tmp_path,
        plans=plans,
        training_gates=training_gates,
        runtime={"pass": True},
        stress={"pass": True},
        hash_paths=hash_paths,
    )

    assert manifest["status"] == "predeployment_not_deployed"
    assert manifest["all_passed"] is True
    assert set(manifest["datasets"]) == {"DE", "AV", "HEALTHCARE"}
    assert "not road deployed" in manifest["claim_boundary"]
    assert "not live clinical deployed" in manifest["claim_boundary"]
    assert "not unrestricted field deployed" in manifest["claim_boundary"]

    locked = json.loads(Path(hash_paths["json"]).read_text(encoding="utf-8"))
    for row in locked["artifacts"]:
        path = Path(row["path"])
        assert path.exists()
        assert freeze._sha256(path) == row["sha256"]
