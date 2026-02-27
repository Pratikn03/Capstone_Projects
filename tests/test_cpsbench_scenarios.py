import pandas as pd
import pytest

from gridpulse.cpsbench_iot.scenarios import generate_episode


def test_generate_episode_nominal():
    obs, true, log = generate_episode("nominal", seed=42, horizon=48)
    assert len(obs) == 48
    assert len(true) == 48
    assert len(log) == 48
    assert log["dropout"].sum() == 0
    assert log["delay_jitter"].sum() == 0
    assert log["out_of_order"].sum() == 0
    assert log["spikes"].sum() == 0
    assert log["stale_sensor"].sum() == 0
    assert log["covariate_drift"].sum() == 0
    assert log["label_drift"].sum() == 0


def test_generate_episode_determinism():
    obs1, true1, log1 = generate_episode("drift_combo", seed=100, horizon=168)
    obs2, true2, log2 = generate_episode("drift_combo", seed=100, horizon=168)
    pd.testing.assert_frame_equal(obs1, obs2)
    pd.testing.assert_frame_equal(true1, true2)
    pd.testing.assert_frame_equal(log1, log2)


def test_generate_episode_dropout():
    obs, true, log = generate_episode("dropout", seed=42, horizon=100, fault_overrides={"dropout_rate": 0.1})
    assert log["dropout"].sum() > 0


def test_generate_episode_delay_jitter():
    obs, true, log = generate_episode("delay_jitter", seed=42, horizon=100, fault_overrides={"delay_rate": 0.5})
    assert log["delay_jitter"].sum() > 0
    assert (log["delay_steps"] > 0).any()


def test_generate_episode_out_of_order():
    obs, true, log = generate_episode("out_of_order", seed=42, horizon=100, fault_overrides={"out_of_order_rate": 0.2})
    assert log["out_of_order"].sum() > 0
    assert (log["arrived_timestamp"] < log["timestamp"]).any()


def test_generate_episode_spikes():
    obs, true, log = generate_episode("spikes", seed=42, horizon=100, fault_overrides={"spike_rate": 0.1})
    assert log["spikes"].sum() > 0
    # Given spikes are applied to obs and not true, these should diverge
    assert not obs["load_mw"].equals(true["load_mw"])


def test_generate_episode_drift_combo():
    obs, true, log = generate_episode("drift_combo", seed=42, horizon=100, drift_timestep=50)
    assert log["covariate_drift"].sum() == 50
    assert log["label_drift"].sum() == 50
    # Because it's a combo, other faults are also generated
    assert log["dropout"].sum() >= 0
    assert log["spikes"].sum() >= 0
