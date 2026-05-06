"""Comprehensive tests for DC3S FTIT (Fault-Tolerant Interval Tracking)."""

from __future__ import annotations

import pytest

from orius.dc3s.ftit import FTIT_FAULT_KEYS, preview_fault_state, update


def _constraints(**overrides):
    base = {
        "capacity_mwh": 100.0,
        "max_power_mw": 40.0,
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "time_step_hours": 1.0,
    }
    base.update(overrides)
    return base


def _cfg(**overrides):
    base = {
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
        "sigma2_floor": 1e-6,
        "alpha_dropout": 1.0,
        "alpha_stale_sensor": 1.0,
        "alpha_delay_jitter": 1.0,
        "alpha_out_of_order": 1.0,
        "alpha_spikes": 1.0,
    }
    base.update(overrides)
    return base


_CLEAN = {k: False for k in FTIT_FAULT_KEYS}
_NOISY = {k: True for k in FTIT_FAULT_KEYS}


class TestPreviewFaultState:
    def test_clean_flags_w_one(self):
        result = preview_fault_state(adaptive_state={}, fault_flags=_CLEAN, cfg=_cfg())
        assert result["w_t"] == pytest.approx(1.0, abs=0.01)

    def test_all_faults_w_below_one(self):
        result = preview_fault_state(adaptive_state={}, fault_flags=_NOISY, cfg=_cfg())
        assert result["w_t"] < 1.0

    def test_single_fault_types(self):
        for key in FTIT_FAULT_KEYS:
            flags = {k: k == key for k in FTIT_FAULT_KEYS}
            result = preview_fault_state(adaptive_state={}, fault_flags=flags, cfg=_cfg())
            assert result["p"][key] > 0.0

    def test_decay_effect(self):
        """Fast decay forgets past faults faster than slow decay."""
        state_fast = {}
        state_slow = {}
        fault_on = {k: (k == "dropout") for k in FTIT_FAULT_KEYS}
        fault_off = {k: False for k in FTIT_FAULT_KEYS}
        # Inject a fault, then turn it off and let it decay
        for flags in [fault_on, fault_on, fault_off, fault_off, fault_off]:
            fast = preview_fault_state(adaptive_state=state_fast, fault_flags=flags, cfg=_cfg(decay=0.5))
            slow = preview_fault_state(adaptive_state=state_slow, fault_flags=flags, cfg=_cfg(decay=0.99))
            state_fast = {"ftit": {"n": fast["n"], "s": fast["s"]}}
            state_slow = {"ftit": {"n": slow["n"], "s": slow["s"]}}
        # After decay, fast-decay should have lower residual rate
        assert fast["p"]["dropout"] < slow["p"]["dropout"]

    def test_alpha_weights(self):
        heavy = preview_fault_state(
            adaptive_state={},
            fault_flags={"dropout": True, **{k: False for k in FTIT_FAULT_KEYS if k != "dropout"}},
            cfg=_cfg(alpha_dropout=5.0),
        )
        light = preview_fault_state(
            adaptive_state={},
            fault_flags={"dropout": True, **{k: False for k in FTIT_FAULT_KEYS if k != "dropout"}},
            cfg=_cfg(alpha_dropout=0.1),
        )
        assert heavy["w_t"] < light["w_t"]

    def test_n_increments(self):
        r1 = preview_fault_state(adaptive_state={}, fault_flags=_CLEAN, cfg=_cfg())
        state = {"ftit": {"n": r1["n"], "s": r1["s"]}}
        r2 = preview_fault_state(adaptive_state=state, fault_flags=_CLEAN, cfg=_cfg())
        assert r2["n"] > r1["n"]


class TestUpdate:
    def test_clean_state_wide_tube(self):
        out = update(adaptive_state={}, fault_flags=_CLEAN, constraints=_constraints(), cfg=_cfg())
        width = out["soc_tube_upper_mwh"] - out["soc_tube_lower_mwh"]
        assert width > 0.0

    def test_degraded_state_narrower_tube(self):
        clean = update(adaptive_state={}, fault_flags=_CLEAN, constraints=_constraints(), cfg=_cfg())
        noisy = update(
            adaptive_state=clean["adaptive_state"], fault_flags=_NOISY, constraints=_constraints(), cfg=_cfg()
        )
        clean_w = clean["soc_tube_upper_mwh"] - clean["soc_tube_lower_mwh"]
        noisy_w = noisy["soc_tube_upper_mwh"] - noisy["soc_tube_lower_mwh"]
        assert noisy_w <= clean_w

    def test_gamma_increases_with_lower_reliability(self):
        clean = update(adaptive_state={}, fault_flags=_CLEAN, constraints=_constraints(), cfg=_cfg())
        noisy = update(
            adaptive_state=clean["adaptive_state"], fault_flags=_NOISY, constraints=_constraints(), cfg=_cfg()
        )
        assert noisy["gamma_mw"] > clean["gamma_mw"]

    def test_e_t_bounded(self):
        state = {}
        for _ in range(50):
            out = update(
                adaptive_state=state, fault_flags=_NOISY, constraints=_constraints(), cfg=_cfg(e_max_mwh=5.0)
            )
            state = out["adaptive_state"]
        assert out["e_t_mwh"] <= 5.0 + 1e-9

    def test_e_t_min_floor(self):
        out = update(
            adaptive_state={}, fault_flags=_CLEAN, constraints=_constraints(), cfg=_cfg(e_min_mwh=2.0)
        )
        assert out["e_t_mwh"] >= 2.0 - 1e-9

    def test_tube_collapse_to_midpoint(self):
        out = update(
            adaptive_state={"ftit": {"e_t_mwh": 50.0}},
            fault_flags=_NOISY,
            constraints=_constraints(min_soc_mwh=10.0, max_soc_mwh=90.0),
            cfg=_cfg(e_max_mwh=100.0, decay_e=1.0, gamma_max_mw=100.0),
        )
        assert out["soc_tube_lower_mwh"] <= out["soc_tube_upper_mwh"] + 1e-9

    def test_sigma2_tracking(self):
        out = update(
            adaptive_state={},
            fault_flags=_CLEAN,
            constraints=_constraints(),
            cfg=_cfg(sigma2_init=1.0, sigma2_decay=0.95),
            sigma2_observation=10.0,
        )
        assert out["sigma2"] > 1.0

    def test_sigma2_floor(self):
        out = update(
            adaptive_state={},
            fault_flags=_CLEAN,
            constraints=_constraints(),
            cfg=_cfg(sigma2_floor=0.5),
            sigma2_observation=0.0,
        )
        assert out["sigma2"] >= 0.5

    def test_stale_tracker_passthrough(self):
        tracker = {"last_values": {"load_mw": 10.0}, "unchanged_counts": {"load_mw": 2}}
        out = update(
            adaptive_state={},
            fault_flags=_CLEAN,
            constraints=_constraints(),
            cfg=_cfg(),
            stale_tracker=tracker,
        )
        assert out["adaptive_state"]["ftit"]["stale_tracker"] == tracker

    def test_sequential_updates(self):
        state = {}
        prev_e = 0.0
        for _ in range(5):
            out = update(adaptive_state=state, fault_flags=_NOISY, constraints=_constraints(), cfg=_cfg())
            assert out["e_t_mwh"] >= prev_e - 1e-9
            prev_e = out["e_t_mwh"]
            state = out["adaptive_state"]

    def test_gamma_power_shapes_curve(self):
        partial = {
            "dropout": True,
            "stale_sensor": False,
            "delay_jitter": False,
            "out_of_order": False,
            "spikes": False,
        }
        linear = update(
            adaptive_state={},
            fault_flags=partial,
            constraints=_constraints(),
            cfg=_cfg(gamma_power=1.0, alpha_dropout=0.5),
        )
        quadratic = update(
            adaptive_state={},
            fault_flags=partial,
            constraints=_constraints(),
            cfg=_cfg(gamma_power=2.0, alpha_dropout=0.5),
        )
        assert 0 < linear["gamma_mw"] < 10.0
        assert 0 < quadratic["gamma_mw"] < 10.0
        assert linear["gamma_mw"] != quadratic["gamma_mw"]

    def test_all_output_keys_present(self):
        out = update(adaptive_state={}, fault_flags=_CLEAN, constraints=_constraints(), cfg=_cfg())
        required = {
            "adaptive_state",
            "w_t",
            "p_drop",
            "p_stale",
            "p_delay",
            "p_ooo",
            "p_spike",
            "gamma_mw",
            "e_t_mwh",
            "soc_tube_lower_mwh",
            "soc_tube_upper_mwh",
            "sigma2",
        }
        assert required <= set(out.keys())

    def test_custom_constraints(self):
        out = update(
            adaptive_state={},
            fault_flags=_CLEAN,
            constraints=_constraints(min_soc_mwh=20.0, max_soc_mwh=80.0),
            cfg=_cfg(),
        )
        assert out["soc_tube_lower_mwh"] >= 20.0 - 1e-9
        assert out["soc_tube_upper_mwh"] <= 80.0 + 1e-9
