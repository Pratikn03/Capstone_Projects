"""Tests for the capacity estimation module (KSG MI, FaultChannelModel, Blahut-Arimoto)."""

from __future__ import annotations

import numpy as np
import pytest

from orius.universal_theory.capacity_estimation import (
    FaultChannelModel,
    blahut_arimoto,
    ksg_mutual_information,
)


class TestKSGMutualInformation:
    def test_gaussian_known_mi(self) -> None:
        rng = np.random.default_rng(42)
        n = 2000
        rho = 0.8
        x = rng.normal(size=n)
        y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
        result = ksg_mutual_information(x, y, k=5)
        expected = -0.5 * np.log(1 - rho**2)
        assert result["I_XY"] == pytest.approx(expected, abs=0.15)
        assert result["n_samples"] == n

    def test_independent_gives_near_zero(self) -> None:
        rng = np.random.default_rng(99)
        x = rng.normal(size=1000)
        y = rng.normal(size=1000)
        result = ksg_mutual_information(x, y, k=5)
        assert result["I_XY"] < 0.1

    def test_mi_is_nonnegative(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.uniform(size=500)
        y = rng.uniform(size=500)
        result = ksg_mutual_information(x, y)
        assert result["I_XY"] >= 0.0

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            ksg_mutual_information(np.array([1.0, 2.0]), np.array([3.0, 4.0]), k=5)


class TestFaultChannelModel:
    def test_no_fault_high_capacity(self) -> None:
        ch = FaultChannelModel(erasure_prob=0.0, delay_steps=0, noise_std=0.0)
        assert ch.capacity() > 1e5

    def test_full_erasure_zero_capacity(self) -> None:
        ch = FaultChannelModel(erasure_prob=1.0)
        assert ch.capacity() == 0.0

    def test_capacity_degrades_with_noise(self) -> None:
        clean = FaultChannelModel(noise_std=0.01)
        noisy = FaultChannelModel(noise_std=1.0)
        assert clean.capacity() > noisy.capacity()

    def test_capacity_degrades_with_erasure(self) -> None:
        low_e = FaultChannelModel(erasure_prob=0.1, noise_std=0.1)
        high_e = FaultChannelModel(erasure_prob=0.5, noise_std=0.1)
        assert low_e.capacity() > high_e.capacity()

    def test_compose_reduces_capacity(self) -> None:
        ch1 = FaultChannelModel(erasure_prob=0.1, noise_std=0.5)
        ch2 = FaultChannelModel(erasure_prob=0.2, noise_std=0.3)
        composed = ch1.compose(ch2)
        assert composed.capacity() <= ch1.capacity()
        assert composed.capacity() <= ch2.capacity()

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="erasure_prob"):
            FaultChannelModel(erasure_prob=1.5)
        with pytest.raises(ValueError, match="noise_std"):
            FaultChannelModel(noise_std=-1.0)


class TestBlahutArimoto:
    def test_converges(self) -> None:
        p_yx = np.array([[0.9, 0.1], [0.1, 0.9]])
        distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = blahut_arimoto(p_yx, distortion, beta=2.0)
        assert result["converged"] is True
        assert result["rate"] >= 0.0
        assert result["distortion"] >= 0.0

    def test_higher_beta_lower_distortion(self) -> None:
        p_yx = np.array([[0.8, 0.2], [0.2, 0.8]])
        distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
        low_beta = blahut_arimoto(p_yx, distortion, beta=0.5)
        high_beta = blahut_arimoto(p_yx, distortion, beta=10.0)
        assert high_beta["distortion"] <= low_beta["distortion"] + 0.01

    def test_policy_is_stochastic(self) -> None:
        p_yx = np.array([[0.7, 0.3], [0.3, 0.7]])
        distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = blahut_arimoto(p_yx, distortion, beta=5.0)
        policy = result["optimal_policy"]
        row_sums = np.sum(policy, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
