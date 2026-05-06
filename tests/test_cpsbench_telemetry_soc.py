"""Comprehensive tests for CPSBench SOC telemetry channel."""

from __future__ import annotations

import pytest

from orius.cpsbench_iot.telemetry_soc import SOCTelemetryChannel, SOCTelemetryFaultConfig


class TestCleanChannel:
    def test_pass_through(self):
        ch = SOCTelemetryChannel(SOCTelemetryFaultConfig(seed=42))
        obs, meta = ch.observe(50.0)
        assert obs == pytest.approx(50.0, abs=0.01)
        assert meta["dropout"] is False
        assert meta["stale"] is False

    def test_sequential_observations(self):
        ch = SOCTelemetryChannel(SOCTelemetryFaultConfig(seed=42))
        for soc in [50.0, 55.0, 60.0]:
            obs, _ = ch.observe(soc)
            assert obs == pytest.approx(soc, abs=0.01)


class TestDropout:
    def test_dropout_reuses_last(self):
        ch = SOCTelemetryChannel(SOCTelemetryFaultConfig(dropout_prob=1.0, seed=42))
        ch.observe(50.0)
        obs, meta = ch.observe(70.0)
        assert meta["dropout"] is True
        assert obs == pytest.approx(50.0, abs=0.01)

    def test_first_dropout_uses_truth(self):
        ch = SOCTelemetryChannel(SOCTelemetryFaultConfig(dropout_prob=1.0, seed=42))
        obs, meta = ch.observe(50.0)
        assert meta["dropout"] is True
        assert obs == pytest.approx(50.0, abs=0.01)


class TestStale:
    def test_stale_repeats_last(self):
        ch = SOCTelemetryChannel(SOCTelemetryFaultConfig(stale_prob=1.0, seed=42))
        ch.observe(50.0)
        obs, meta = ch.observe(70.0)
        assert meta["stale"] is True
        assert obs == pytest.approx(50.0, abs=0.01)

    def test_no_stale_on_first(self):
        ch = SOCTelemetryChannel(SOCTelemetryFaultConfig(stale_prob=1.0, seed=42))
        obs, meta = ch.observe(50.0)
        assert meta["stale"] is False


class TestNoise:
    def test_noise_adds_variation(self):
        ch = SOCTelemetryChannel(SOCTelemetryFaultConfig(noise_std_mwh=5.0, seed=42))
        obs, meta = ch.observe(50.0)
        assert meta["noise"] != 0.0
        assert obs != 50.0

    def test_zero_noise(self):
        ch = SOCTelemetryChannel(SOCTelemetryFaultConfig(noise_std_mwh=0.0, seed=42))
        obs, meta = ch.observe(50.0)
        assert meta["noise"] == pytest.approx(0.0)
        assert obs == pytest.approx(50.0)


class TestDeterminism:
    def test_same_seed_same_output(self):
        ch1 = SOCTelemetryChannel(SOCTelemetryFaultConfig(noise_std_mwh=2.0, seed=99))
        ch2 = SOCTelemetryChannel(SOCTelemetryFaultConfig(noise_std_mwh=2.0, seed=99))
        o1, _ = ch1.observe(50.0)
        o2, _ = ch2.observe(50.0)
        assert o1 == pytest.approx(o2)
