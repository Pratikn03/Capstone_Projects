"""Comprehensive tests for CPSBench scenario generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orius.cpsbench_iot.scenarios import DEFAULT_SCENARIOS, FAULT_COLUMNS, generate_episode


class TestNominal:
    def test_shape(self):
        obs, true, log = generate_episode("nominal", seed=42, horizon=24)
        assert len(obs) == 24
        assert len(true) == 24
        assert len(log) == 24

    def test_no_faults(self):
        _, _, log = generate_episode("nominal", seed=42, horizon=48)
        for col in FAULT_COLUMNS:
            assert log[col].sum() == 0

    def test_columns_present(self):
        obs, true, log = generate_episode("nominal", seed=42, horizon=12)
        for col in ("timestamp", "load_mw", "renewables_mw", "price_per_mwh", "carbon_kg_per_mwh"):
            assert col in obs.columns
            assert col in true.columns

    def test_deterministic(self):
        o1, _, _ = generate_episode("nominal", seed=0, horizon=24)
        o2, _, _ = generate_episode("nominal", seed=0, horizon=24)
        pd.testing.assert_frame_equal(o1, o2)

    def test_different_seeds_differ(self):
        o1, _, _ = generate_episode("nominal", seed=0, horizon=24)
        o2, _, _ = generate_episode("nominal", seed=99, horizon=24)
        assert not np.allclose(o1["load_mw"].values, o2["load_mw"].values)


class TestDropout:
    def test_dropout_flags_present(self):
        _, _, log = generate_episode("dropout", seed=42, horizon=48)
        assert log["dropout"].sum() > 0

    def test_obs_has_no_nans_after_ffill(self):
        obs, _, _ = generate_episode("dropout", seed=42, horizon=48)
        assert obs["load_mw"].isna().sum() == 0

    def test_custom_dropout_rate(self):
        _, _, log = generate_episode("dropout", seed=42, horizon=100, fault_overrides={"dropout_rate": 0.50})
        rate = log["dropout"].mean()
        assert rate > 0.3


class TestDelayJitter:
    def test_delay_flags(self):
        _, _, log = generate_episode("delay_jitter", seed=42, horizon=48)
        assert log["delay_jitter"].sum() > 0

    def test_arrived_timestamps_differ(self):
        _, _, log = generate_episode("delay_jitter", seed=42, horizon=48)
        delayed = log[log["delay_jitter"] == 1]
        assert not delayed.empty
        assert (delayed["arrived_timestamp"] != delayed["timestamp"]).any()


class TestOutOfOrder:
    def test_ooo_flags(self):
        _, _, log = generate_episode("out_of_order", seed=42, horizon=48)
        assert log["out_of_order"].sum() > 0

    def test_requires_min_length(self):
        _, _, log = generate_episode("out_of_order", seed=42, horizon=6)
        assert log["out_of_order"].sum() == 0


class TestSpikes:
    def test_spike_flags(self):
        _, _, log = generate_episode("spikes", seed=42, horizon=48)
        assert log["spikes"].sum() > 0

    def test_spike_sigma_recorded(self):
        _, _, log = generate_episode("spikes", seed=42, horizon=48)
        assert log["spike_sigma"].max() > 0.0


class TestStaleSensor:
    def test_stale_flags(self):
        _, _, log = generate_episode("stale_sensor", seed=42, horizon=48)
        assert log["stale_sensor"].sum() > 0


class TestDriftCombo:
    def test_all_fault_types_present(self):
        _, _, log = generate_episode("drift_combo", seed=42, horizon=168)
        for col in ("dropout", "delay_jitter", "spikes", "stale_sensor", "covariate_drift", "label_drift"):
            assert log[col].sum() > 0, f"{col} not present"

    def test_custom_drift_timestep(self):
        _, _, log = generate_episode("drift_combo", seed=42, horizon=168, drift_timestep=10)
        assert log["covariate_drift"].iloc[10] == 1


class TestFaultOverrides:
    def test_load_scale(self):
        obs1, _, _ = generate_episode("nominal", seed=42, horizon=24)
        obs2, _, _ = generate_episode("nominal", seed=42, horizon=24, fault_overrides={"load_scale": 2.0})
        ratio = obs2["load_mw"].mean() / obs1["load_mw"].mean()
        assert ratio == pytest.approx(2.0, abs=0.1)

    def test_price_scale(self):
        obs1, _, _ = generate_episode("nominal", seed=42, horizon=24)
        obs2, _, _ = generate_episode("nominal", seed=42, horizon=24, fault_overrides={"price_scale": 0.5})
        assert obs2["price_per_mwh"].mean() < obs1["price_per_mwh"].mean()

    def test_seasonal_shift(self):
        obs1, _, _ = generate_episode("nominal", seed=42, horizon=48)
        obs2, _, _ = generate_episode(
            "nominal", seed=42, horizon=48, fault_overrides={"seasonal_shift_hours": 6}
        )
        assert not np.allclose(obs1["load_mw"].values, obs2["load_mw"].values)


class TestEdgeCases:
    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            generate_episode("nonexistent", seed=42)

    def test_zero_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon must be > 0"):
            generate_episode("nominal", seed=42, horizon=0)

    def test_horizon_one(self):
        obs, true, log = generate_episode("nominal", seed=42, horizon=1)
        assert len(obs) == 1

    def test_all_default_scenarios(self):
        for sc in DEFAULT_SCENARIOS:
            obs, true, log = generate_episode(sc, seed=42, horizon=24)
            assert len(obs) == 24
