from __future__ import annotations

from orius.dc3s.ftit import preview_fault_state, update


def _constraints() -> dict[str, float]:
    return {
        "capacity_mwh": 100.0,
        "max_power_mw": 40.0,
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "time_step_hours": 1.0,
    }


def _cfg() -> dict[str, float]:
    return {
        "decay": 0.98,
        "decay_e": 0.95,
        "dt_hours": 1.0,
        "gamma_min_mw": 0.0,
        "gamma_max_mw": 10.0,
        "gamma_power": 1.0,
        "e_min_mwh": 0.0,
        "e_max_mwh": 50.0,
        "sigma2_init": 1.0,
        "sigma2_decay": 0.95,
        "sigma2_floor": 1.0e-6,
        "alpha_dropout": 1.0,
        "alpha_stale_sensor": 1.0,
        "alpha_delay_jitter": 1.0,
        "alpha_out_of_order": 1.0,
        "alpha_spikes": 1.0,
    }


def test_preview_fault_state_monotonicity():
    cfg = _cfg()
    clean = preview_fault_state(
        adaptive_state={},
        fault_flags={key: False for key in ("dropout", "stale_sensor", "delay_jitter", "out_of_order", "spikes")},
        cfg=cfg,
    )
    noisy = preview_fault_state(
        adaptive_state={},
        fault_flags={key: True for key in ("dropout", "stale_sensor", "delay_jitter", "out_of_order", "spikes")},
        cfg=cfg,
    )
    assert noisy["w_t"] < clean["w_t"]


def test_update_gamma_increases_and_tube_tightens_as_reliability_drops():
    cfg = _cfg()
    clean = update(
        adaptive_state={},
        fault_flags={key: False for key in ("dropout", "stale_sensor", "delay_jitter", "out_of_order", "spikes")},
        constraints=_constraints(),
        cfg=cfg,
    )
    noisy = update(
        adaptive_state=clean["adaptive_state"],
        fault_flags={key: True for key in ("dropout", "stale_sensor", "delay_jitter", "out_of_order", "spikes")},
        constraints=_constraints(),
        cfg=cfg,
        sigma2_observation=4.0,
    )
    assert noisy["w_t"] < clean["w_t"]
    assert noisy["gamma_mw"] > clean["gamma_mw"]
    assert noisy["e_t_mwh"] >= clean["e_t_mwh"]
    clean_width = clean["soc_tube_upper_mwh"] - clean["soc_tube_lower_mwh"]
    noisy_width = noisy["soc_tube_upper_mwh"] - noisy["soc_tube_lower_mwh"]
    assert noisy_width <= clean_width


def test_update_tracks_all_five_fault_components():
    cfg = _cfg()
    for fault_key, result_key in (
        ("dropout", "p_drop"),
        ("stale_sensor", "p_stale"),
        ("delay_jitter", "p_delay"),
        ("out_of_order", "p_ooo"),
        ("spikes", "p_spike"),
    ):
        out = update(
            adaptive_state={},
            fault_flags={key: key == fault_key for key in ("dropout", "stale_sensor", "delay_jitter", "out_of_order", "spikes")},
            constraints=_constraints(),
            cfg=cfg,
        )
        assert out[result_key] > 0.0
