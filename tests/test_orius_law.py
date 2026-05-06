"""Tests for the ORIUS Rate-Distortion Safety Laws (L1-L4)."""

from __future__ import annotations

import numpy as np
import pytest

from orius.universal_theory.capacity_estimation import FaultChannelModel
from orius.universal_theory.orius_law import (
    LAW_REGISTER,
    achievability_converse_sandwich,
    capacity_bridge,
    capacity_bridge_proof,
    capacity_bridge_verify,
    critical_capacity,
    fano_binary_corollary,
    orius_grand_unification,
    rate_distortion_safety_law,
)


class TestL1RateDistortionSafetyLaw:
    def test_positive_loss_below_capacity(self) -> None:
        result = rate_distortion_safety_law(channel_capacity=0.5, H_entropy=1.0, alpha=0.10)
        assert result["D_star_lower"] > 0.0
        assert result["law_applies"] is True

    def test_zero_loss_at_full_capacity(self) -> None:
        result = rate_distortion_safety_law(channel_capacity=1.0, H_entropy=1.0, alpha=0.10)
        assert result["D_star_lower"] == pytest.approx(0.0)
        assert result["law_applies"] is False

    def test_loss_increases_as_capacity_drops(self) -> None:
        high = rate_distortion_safety_law(0.8, 1.0, 0.10)
        low = rate_distortion_safety_law(0.3, 1.0, 0.10)
        assert low["D_star_lower"] > high["D_star_lower"]

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            rate_distortion_safety_law(0.5, 1.0, alpha=0.0)

    def test_negative_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="channel_capacity"):
            rate_distortion_safety_law(-0.1, 1.0, alpha=0.10)


class TestL2CapacityBridge:
    def test_bounded_by_one(self) -> None:
        result = capacity_bridge(w_bar=0.8, kappa_d=1.0, H_X=1.0, channel_capacity=2.0)
        assert result["w_upper_bound"] <= 1.0

    def test_zero_capacity_gives_zero_w(self) -> None:
        result = capacity_bridge(w_bar=0.0, kappa_d=1.0, H_X=1.0, channel_capacity=0.0)
        assert result["w_upper_bound"] == pytest.approx(0.0)
        assert result["consistent"] is True

    def test_inconsistency_detected(self) -> None:
        result = capacity_bridge(w_bar=0.9, kappa_d=1.0, H_X=1.0, channel_capacity=0.3)
        assert result["consistent"] is False

    def test_negative_channel_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="channel_capacity"):
            capacity_bridge(w_bar=0.5, kappa_d=1.0, H_X=1.0, channel_capacity=-0.2)


class TestL3CriticalCapacity:
    def test_critical_capacity_exists(self) -> None:
        result = critical_capacity(alpha=0.10, kappa_d=1.0, H_X=1.0)
        assert result["C_star_d"] > 0.0

    def test_epsilon_must_be_less_than_alpha(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            critical_capacity(alpha=0.10, epsilon=0.10)

    def test_higher_kappa_reduces_critical_capacity(self) -> None:
        low_k = critical_capacity(alpha=0.10, kappa_d=0.5, H_X=1.0)
        high_k = critical_capacity(alpha=0.10, kappa_d=1.0, H_X=1.0)
        assert high_k["C_star_d"] < low_k["C_star_d"]

    def test_nonpositive_entropy_rejected(self) -> None:
        with pytest.raises(ValueError, match="H_X"):
            critical_capacity(alpha=0.10, kappa_d=1.0, H_X=0.0)


class TestL4AchievabilityConverseSandwich:
    def test_bounds_consistent(self) -> None:
        result = achievability_converse_sandwich(w_bar=0.7, alpha=0.10)
        assert result["lower_bound"] <= result["upper_bound"]

    def test_perfect_reliability_zero_bounds(self) -> None:
        result = achievability_converse_sandwich(w_bar=1.0, alpha=0.10)
        assert result["lower_bound"] == pytest.approx(0.0)
        assert result["upper_bound"] == pytest.approx(0.0)

    def test_gap_is_constant_factor(self) -> None:
        result = achievability_converse_sandwich(w_bar=0.6, alpha=0.10, K_factor=2.0)
        assert result["gap_is_constant"] is True
        ratio = result["upper_bound"] / max(result["lower_bound"], 1e-15)
        assert ratio == pytest.approx(2.0)

    def test_k_factor_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="K_factor"):
            achievability_converse_sandwich(w_bar=0.6, alpha=0.10, K_factor=0.5)


class TestLawRegister:
    def test_all_four_laws_present(self) -> None:
        assert "L1" in LAW_REGISTER
        assert "L2" in LAW_REGISTER
        assert "L3" in LAW_REGISTER
        assert "L4" in LAW_REGISTER

    def test_types_match(self) -> None:
        assert LAW_REGISTER["L1"]["type"] == "fundamental_law"
        assert LAW_REGISTER["L2"]["type"] == "bridge_theorem"
        assert LAW_REGISTER["L3"]["type"] == "impossibility_law"
        assert LAW_REGISTER["L4"]["type"] == "characterization"

    def test_laws_are_marked_as_stylized(self) -> None:
        assert LAW_REGISTER["L1"]["status"] == "stylized_not_defended"
        assert LAW_REGISTER["L2"]["status"] == "stylized_not_defended"
        assert LAW_REGISTER["L3"]["status"] == "stylized_not_defended"
        assert LAW_REGISTER["L4"]["status"] == "stylized_not_defended"
        assert LAW_REGISTER["L4"]["dependencies"] == ["L1", "L2", "T3_upper_envelope"]


class TestGrandUnification:
    def test_assembles_all_paths(self) -> None:
        result = orius_grand_unification(
            [0.7] * 100,
            alpha=0.10,
            n_cal=500,
            delta=0.05,
            margin=5.0,
            sigma_d=0.1,
        )
        assert result["gap_closed"] is False
        assert "path_a_rate_distortion" in result
        assert "path_b_capacity_bridge" in result
        assert "path_c_critical_capacity" in result
        assert "path_d_sandwich" in result
        assert "path_e_trajectory_pac" in result
        assert "scope_note" in result

    def test_degraded_reliability_still_closes(self) -> None:
        result = orius_grand_unification(
            [0.4] * 50,
            alpha=0.10,
            n_cal=500,
            delta=0.05,
        )
        assert result["gap_closed"] is False
        assert result["path_d_sandwich"]["lower_bound"] > 0.0


class TestCapacityBridgeProof:
    def test_consistent_with_fault_channel_model(self) -> None:
        ch = FaultChannelModel(erasure_prob=0.1, noise_std=0.5)
        result = capacity_bridge_proof(ch, H_X=1.0, kappa_d=1.0)
        assert result["channel_capacity"] == pytest.approx(ch.capacity())
        assert result["w_upper_bound"] <= 1.0
        assert len(result["steps"]) == 5

    def test_full_erasure_gives_zero_bound(self) -> None:
        ch = FaultChannelModel(erasure_prob=1.0)
        result = capacity_bridge_proof(ch, H_X=1.0)
        assert result["channel_capacity"] == pytest.approx(0.0)
        assert result["w_upper_bound"] == pytest.approx(0.0)

    def test_invalid_H_X_raises(self) -> None:
        ch = FaultChannelModel()
        with pytest.raises(ValueError, match="H_X"):
            capacity_bridge_proof(ch, H_X=0.0)


class TestCapacityBridgeVerify:
    def test_bootstrap_ci_contains_point_estimate(self) -> None:
        rng = np.random.default_rng(42)
        w_seq = rng.uniform(0.3, 0.8, size=200)
        result = capacity_bridge_verify(w_seq, channel_capacity=1.0, H_X=1.0)
        assert result["ci_lower"] <= result["kappa_d_hat"] <= result["ci_upper"]
        assert result["n_samples"] == 200

    def test_low_w_gives_kappa_below_one(self) -> None:
        w_seq = [0.2] * 100
        result = capacity_bridge_verify(w_seq, channel_capacity=1.0, H_X=1.0)
        assert result["kappa_d_hat"] <= 1.0
        assert result["bridge_holds"] is True

    def test_empty_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            capacity_bridge_verify([], channel_capacity=1.0, H_X=1.0)


class TestFanoBinaryCorollary:
    def test_positive_error_below_capacity(self) -> None:
        result = fano_binary_corollary(channel_capacity=0.5, H_binary=1.0, alpha=0.10)
        assert result["P_e_lower"] > 0.0
        assert result["safety_bound"] > 0.0
        assert result["law_applies"] is True

    def test_zero_error_at_full_capacity(self) -> None:
        result = fano_binary_corollary(channel_capacity=1.0, H_binary=1.0, alpha=0.10)
        assert result["P_e_lower"] == pytest.approx(0.0)
        assert result["safety_bound"] == pytest.approx(0.0)

    def test_matches_l1_for_binary(self) -> None:
        l1 = rate_distortion_safety_law(0.6, 1.0, 0.10)
        fano = fano_binary_corollary(0.6, 1.0, 0.10)
        assert l1["D_star_lower"] == pytest.approx(fano["safety_bound"])


class TestWtAsCapacityProxy:
    def test_roundtrip_with_capacity_bridge(self) -> None:
        from orius.dc3s.quality import w_t_as_capacity_proxy

        result = w_t_as_capacity_proxy(w_t=0.6, kappa_d=1.0, H_X=1.0)
        bridge = capacity_bridge(w_bar=0.6, kappa_d=1.0, H_X=1.0, channel_capacity=result["C_implied"])
        assert bridge["consistent"] is True

    def test_zero_w_gives_zero_capacity(self) -> None:
        from orius.dc3s.quality import w_t_as_capacity_proxy

        result = w_t_as_capacity_proxy(w_t=0.0)
        assert result["C_implied"] == pytest.approx(0.0)
        assert result["above_critical"] is False

    def test_high_w_above_critical(self) -> None:
        from orius.dc3s.quality import w_t_as_capacity_proxy

        result = w_t_as_capacity_proxy(w_t=0.9, kappa_d=1.0, H_X=1.0)
        assert result["above_critical"] is True
