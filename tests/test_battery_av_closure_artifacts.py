from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from orius.dc3s.certificate import make_certificate, store_certificates_batch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_battery_av_closure_artifacts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_battery_av_closure_artifacts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


closure_script = _load_script()


def _write_certificates(db_path: Path) -> None:
    cert1 = make_certificate(
        command_id="cmd-1",
        device_id="dev",
        zone_id="zone",
        controller="ctrl",
        proposed_action={"x": 1.0},
        safe_action={"x": 0.5},
        uncertainty={"lower": 0.0, "upper": 1.0},
        reliability={"w_t": 0.8},
        drift={"drift": False},
        model_hash="model",
        config_hash="cfg",
        runtime_surface="bounded_runtime",
        closure_tier="defended_bounded_row",
        intervened=True,
        intervention_reason="clamp",
        reliability_w=0.8,
    )
    cert2 = make_certificate(
        command_id="cmd-2",
        device_id="dev",
        zone_id="zone",
        controller="ctrl",
        proposed_action={"x": 1.0},
        safe_action={"x": 0.0},
        uncertainty={"lower": 0.0, "upper": 1.0},
        reliability={"w_t": 0.6},
        drift={"drift": True},
        model_hash="model",
        config_hash="cfg",
        prev_hash=cert1["certificate_hash"],
        runtime_surface="bounded_runtime",
        closure_tier="defended_bounded_row",
        intervened=True,
        intervention_reason="fallback",
        reliability_w=0.6,
    )
    store_certificates_batch([cert1, cert2], duckdb_path=str(db_path), table_name="dispatch_certificates")


def test_build_closure_smoke(tmp_path: Path) -> None:
    battery_dir = tmp_path / "battery"
    av_dir = tmp_path / "av"
    overall_dir = tmp_path / "overall"
    battery_dir.mkdir()
    av_dir.mkdir()
    overall_dir.mkdir()

    pd.DataFrame(
        [
            {
                "trace_id": "b1",
                "controller_label": "deep:dc3s_wrapped",
                "controller": "dc3s_wrapped",
                "fault_family": "dropout",
                "step_index": 1,
                "observed_margin": 0.4,
                "true_margin": -0.2,
                "reliability_w": 0.5,
                "widening_factor": 1.4,
                "intervened": True,
                "fallback_used": False,
                "certificate_valid": True,
                "true_value": 0.15,
                "interval_lower": 0.10,
                "interval_upper": 0.20,
                "drift_score": 0.3,
                "dispatch_regime": "charge",
            },
            {
                "trace_id": "b2",
                "controller_label": "deep:dc3s_wrapped",
                "controller": "dc3s_wrapped",
                "fault_family": "dropout",
                "step_index": 2,
                "observed_margin": 0.2,
                "true_margin": 0.1,
                "reliability_w": 0.9,
                "widening_factor": 1.0,
                "intervened": False,
                "fallback_used": False,
                "certificate_valid": True,
                "true_value": 0.16,
                "interval_lower": 0.12,
                "interval_upper": 0.21,
                "drift_score": 0.1,
                "dispatch_regime": "hold",
            },
        ]
    ).to_csv(battery_dir / "runtime_traces.csv", index=False)
    pd.DataFrame([{"controller": "deep:dc3s_wrapped", "tsvr": 0.0, "oasg": 0.5, "cva": 1.0, "gdq": 0.8, "intervention_rate": 0.5, "audit_completeness": 1.0, "recovery_latency": 0.0, "n_steps": 2}]).to_csv(
        battery_dir / "runtime_summary.csv",
        index=False,
    )
    pd.DataFrame([{"controller": "deep:dc3s_wrapped", "fault_family": "dropout", "target": "soc_mwh", "coverage": 1.0, "mean_width": 0.1}]).to_csv(
        battery_dir / "fault_family_coverage.csv",
        index=False,
    )
    (battery_dir / "battery_deep_learning_novelty_register.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    _write_certificates(battery_dir / "battery_runtime.duckdb")

    pd.DataFrame(
        [
            {
                "trace_id": "a1",
                "scenario_id": "s1",
                "shard_id": "0",
                "controller": "orius",
                "fault_family": "delay_jitter",
                "step_index": 1,
                "observed_margin": 0.5,
                "true_margin": -0.3,
                "reliability_w": 0.4,
                "widening_factor": 1.6,
                "intervened": True,
                "fallback_used": False,
                "certificate_valid": True,
                "validity_score": 0.8,
                "ego_speed_mps": 12.0,
                "neighbor_count": 3,
                "target_ego_speed_1s": 11.5,
                "base_pred_ego_speed_lower_mps": 10.5,
                "base_pred_ego_speed_upper_mps": 12.5,
                "target_relative_gap_1s": 18.0,
                "base_pred_relative_gap_lower_m": 14.0,
                "base_pred_relative_gap_upper_m": 17.0,
                "shift_score": 0.4,
            },
            {
                "trace_id": "a2",
                "scenario_id": "s1",
                "shard_id": "0",
                "controller": "orius",
                "fault_family": "delay_jitter",
                "step_index": 2,
                "observed_margin": 0.1,
                "true_margin": 0.2,
                "reliability_w": 0.9,
                "widening_factor": 1.0,
                "intervened": False,
                "fallback_used": False,
                "certificate_valid": True,
                "validity_score": 1.0,
                "ego_speed_mps": 9.0,
                "neighbor_count": 1,
                "target_ego_speed_1s": 9.2,
                "base_pred_ego_speed_lower_mps": 8.4,
                "base_pred_ego_speed_upper_mps": 9.4,
                "target_relative_gap_1s": 20.0,
                "base_pred_relative_gap_lower_m": 18.0,
                "base_pred_relative_gap_upper_m": 21.0,
                "shift_score": 0.1,
            },
        ]
    ).to_csv(av_dir / "runtime_traces.csv", index=False)
    pd.DataFrame([{"controller": "orius", "tsvr": 0.0, "oasg": 0.5, "cva": 1.0, "gdq": 0.9, "intervention_rate": 0.5, "audit_completeness": 1.0, "recovery_latency": 0.0, "n_steps": 2}]).to_csv(
        av_dir / "runtime_summary.csv",
        index=False,
    )
    pd.DataFrame([{"controller": "orius", "fault_family": "delay_jitter", "target": "ego_speed_mps", "coverage": 1.0, "mean_width": 2.0}]).to_csv(
        av_dir / "fault_family_coverage.csv",
        index=False,
    )
    (av_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    _write_certificates(av_dir / "dc3s_av_waymo_dryrun.duckdb")

    report = closure_script.build_closure(
        battery_dir=battery_dir,
        av_dir=av_dir,
        overall_dir=overall_dir,
        docs_dir=tmp_path / "docs",
    )

    assert Path(report["release_summary"]).exists()
    assert Path(report["publication_override"]).exists()
    assert Path(report["executive_summary"]).exists()
    assert Path(report["claim_ledger"]).exists()
    assert Path(battery_dir / "observed_true_counterexamples.csv").exists()
    assert Path(av_dir / "observed_true_counterexamples.csv").exists()
    assert Path(battery_dir / "shift_aware_adaptive_summary.json").exists()
    assert Path(av_dir / "shift_aware_adaptive_summary.json").exists()
    assert Path(battery_dir / "certos_verification_summary.json").exists()
    assert Path(av_dir / "certos_verification_summary.json").exists()

    override = json.loads(Path(report["publication_override"]).read_text(encoding="utf-8"))
    release = json.loads(Path(report["release_summary"]).read_text(encoding="utf-8"))
    assert override["battery"]["resulting_tier"] == "reference"
    assert override["vehicle"]["resulting_tier"] == "runtime_contract_closed"
    assert release["battery"]["runtime_rows_total"] == 2
    assert release["battery"]["runtime_rows_canonical_controller"] == 2
    assert release["battery"]["runtime_trace_rows"] == 2
    assert release["av"]["runtime_rows_total"] == 2
    assert release["av"]["runtime_rows_canonical_controller"] == 2
    assert release["av"]["runtime_trace_rows"] == 2
