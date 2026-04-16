from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import orius.av_waymo.training as av_training
from orius.dc3s.online_calibration import OnlineCalibrator


REPO_ROOT = Path(__file__).resolve().parents[1]
MONOGRAPH_SCRIPT_PATH = REPO_ROOT / "scripts" / "build_orius_monograph_assets.py"
BATTERY_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_battery_deep_novelty.py"


def _load_monograph_script():
    spec = importlib.util.spec_from_file_location("build_orius_monograph_assets", MONOGRAPH_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_battery_script():
    spec = importlib.util.spec_from_file_location("run_battery_deep_novelty", BATTERY_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


monograph_script = _load_monograph_script()
battery_script = _load_battery_script()


@pytest.mark.parametrize(
    ("bucket", "expected"),
    [
        (0, "train"),
        (69, "train"),
        (70, "calibration"),
        (79, "calibration"),
        (80, "val"),
        (89, "val"),
        (90, "test"),
        (99, "test"),
    ],
)
def test_assign_split_uses_documented_70_10_10_10_policy(monkeypatch: pytest.MonkeyPatch, bucket: int, expected: str) -> None:
    monkeypatch.setattr(av_training, "_hash_percent", lambda _value: bucket)
    assert av_training.assign_split("scenario-x") == expected


def test_online_calibrator_state_dict_round_trips_window_size() -> None:
    calibrator = OnlineCalibrator(window_size=200, forgetting_factor=0.97, drift_forgetting_factor=0.6, post_drift_steps=15)
    for idx in range(50):
        calibrator.update(float(idx) / 10.0)
    state = calibrator.state_dict()

    restored = OnlineCalibrator.from_state_dict(state)

    assert state["window_size"] == 200
    assert restored._window.maxlen == 200
    assert restored.n_samples == 50


def test_online_calibrator_legacy_restore_falls_back_to_residual_count() -> None:
    calibrator = OnlineCalibrator(window_size=120)
    for idx in range(25):
        calibrator.update(float(idx))
    state = calibrator.state_dict()
    state.pop("window_size")

    restored = OnlineCalibrator.from_state_dict(state)

    assert restored._window.maxlen == 25


def test_submission_scope_override_demotes_industrial_and_healthcare() -> None:
    industrial = monograph_script._apply_submission_scope_override(
        "industrial",
        {"resulting_tier": "proof_validated", "exact_blocker": "industrial_train_validation_chain_complete"},
        submission_scope="battery_av_only",
    )
    healthcare = monograph_script._apply_submission_scope_override(
        "healthcare",
        {"resulting_tier": "proof_validated", "exact_blocker": "healthcare_train_validation_chain_complete"},
        submission_scope="battery_av_only",
    )
    vehicle = monograph_script._apply_submission_scope_override(
        "vehicle",
        {"resulting_tier": "proof_validated", "exact_blocker": "none"},
        submission_scope="battery_av_only",
    )

    assert industrial["resulting_tier"] == "proof_candidate_only"
    assert industrial["exact_blocker"] == "outside_current_submission_scope_battery_av_lane"
    assert healthcare["resulting_tier"] == "proof_candidate_only"
    assert healthcare["exact_blocker"] == "outside_current_submission_scope_battery_av_lane"
    assert vehicle["resulting_tier"] == "proof_validated"


def test_battery_runtime_certificate_chain_continues_across_controller_batches() -> None:
    constraints = {
        "max_power_mw": 10.0,
        "min_soc_mwh": 0.0,
        "max_soc_mwh": 20.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
    }
    certificate_template = {
        "command_id": "cmd-0",
        "device_id": "battery-1",
        "zone_id": "zone-a",
        "controller": "dc3s_wrapped",
        "safe_action": {"charge_mw": 1.0, "discharge_mw": 0.0},
        "proposed_action": {"charge_mw": 1.0, "discharge_mw": 0.0},
        "uncertainty": {"soc_lower_mwh": 5.0, "soc_upper_mwh": 6.0},
        "reliability": {"w_t": 0.9},
        "drift": {"drift": False},
        "model_hash": "model",
        "config_hash": "cfg",
        "certificate_hash": "placeholder",
    }
    buffers = {
        "soc_true_mwh": [5.0],
        "soc_observed_mwh": [5.0],
        "proposed_charge_mw": [1.0],
        "proposed_discharge_mw": [0.0],
        "safe_charge_mw": [1.0],
        "safe_discharge_mw": [0.0],
        "w_t": [0.9],
        "guarantee_checks_passed": [1.0],
        "interval_lower": [4.0],
        "interval_upper": [6.0],
        "certificates": [dict(certificate_template)],
    }

    _, _, first_certs, previous_hash = battery_script._battery_runtime_artifacts_for_controller(
        lane="deep",
        scenario="nominal",
        seed=0,
        controller="dc3s_wrapped",
        buffers=buffers,
        constraints=constraints,
        previous_certificate_hash=None,
    )
    second_buffers = dict(buffers)
    second_buffers["certificates"] = [dict(certificate_template, command_id="cmd-1")]
    _, _, second_certs, final_hash = battery_script._battery_runtime_artifacts_for_controller(
        lane="deep",
        scenario="spike",
        seed=1,
        controller="dc3s_wrapped",
        buffers=second_buffers,
        constraints=constraints,
        previous_certificate_hash=previous_hash,
    )

    assert first_certs[0]["prev_hash"] is None
    assert second_certs[0]["prev_hash"] == first_certs[0]["certificate_hash"]
    assert final_hash == second_certs[0]["certificate_hash"]
