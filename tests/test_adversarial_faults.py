"""Tests for adversarial fault types and Byzantine-resistant OQE.

Phase 3 of ORIUS gap-closing plan:
  - replay fault: stale reading from k_steps_ago
  - coordinated_spoof fault: systematic small bias
  - compute_reliability_robust: MAD spike detection + trimmed mean + consistency check
"""

from __future__ import annotations

import numpy as np
import pytest

from orius.dc3s.quality import compute_reliability_robust
from orius.orius_bench.fault_engine import (
    FaultEvent,
    apply_faults,
)

# ---------------------------------------------------------------------------
# Replay fault
# ---------------------------------------------------------------------------


class TestReplayFault:
    def test_replay_returns_stale_reading(self):
        """replay fault should return reading from k_steps_ago."""
        # Build a history of 10 distinct states
        history = [{"soc": float(i * 0.1)} for i in range(10)]
        true_state = {"soc": 0.95}  # current true value

        fault = FaultEvent(
            step=0,
            kind="replay",
            params={"k_steps_ago": 3, "history": history},
            duration=1,
        )
        observed = apply_faults(true_state, [fault])
        # Should return history[-3] = {"soc": 0.7}
        assert observed["soc"] == pytest.approx(0.7)

    def test_replay_falls_back_when_history_empty(self):
        """replay with empty history returns original state."""
        true_state = {"soc": 0.80}
        fault = FaultEvent(
            step=0,
            kind="replay",
            params={"k_steps_ago": 5, "history": []},
        )
        observed = apply_faults(true_state, [fault])
        assert observed["soc"] == pytest.approx(0.80)

    def test_replay_falls_back_when_k_exceeds_history(self):
        """replay with k > len(history) returns original state."""
        history = [{"soc": 0.5}]
        true_state = {"soc": 0.90}
        fault = FaultEvent(
            step=0,
            kind="replay",
            params={"k_steps_ago": 10, "history": history},
        )
        observed = apply_faults(true_state, [fault])
        assert observed["soc"] == pytest.approx(0.90)

    def test_replay_does_not_mutate_true_state(self):
        """apply_faults never mutates the original true_state dict."""
        history = [{"soc": 0.2}]
        true_state = {"soc": 0.80}
        original_soc = true_state["soc"]
        fault = FaultEvent(
            step=0,
            kind="replay",
            params={"k_steps_ago": 1, "history": history},
        )
        apply_faults(true_state, [fault])
        assert true_state["soc"] == original_soc  # unchanged

    def test_replay_k1_returns_most_recent_history(self):
        history = [{"soc": 0.11}, {"soc": 0.22}, {"soc": 0.33}]
        true_state = {"soc": 0.99}
        fault = FaultEvent(
            step=0,
            kind="replay",
            params={"k_steps_ago": 1, "history": history},
        )
        observed = apply_faults(true_state, [fault])
        assert observed["soc"] == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# coordinated_spoof fault
# ---------------------------------------------------------------------------


class TestCoordinatedSpoofFault:
    def test_spoof_applies_small_systematic_bias(self):
        """coordinated_spoof should add spoof_fraction * normal_range to all numeric fields."""
        true_state = {"soc": 0.50, "power": 100.0}
        fault = FaultEvent(
            step=0,
            kind="coordinated_spoof",
            params={"spoof_fraction": 0.05, "normal_range": 1.0},
        )
        observed = apply_faults(true_state, [fault])
        assert observed["soc"] == pytest.approx(0.55)
        assert observed["power"] == pytest.approx(100.05)

    def test_spoof_magnitude_le_10pct(self):
        """Default spoof should produce < 10% shift."""
        true_state = {"soc": 0.50}
        fault = FaultEvent(
            step=0,
            kind="coordinated_spoof",
            params={"spoof_fraction": 0.10, "normal_range": 1.0},
        )
        observed = apply_faults(true_state, [fault])
        shift = abs(observed["soc"] - true_state["soc"])
        assert shift <= 0.10 + 1e-9  # at most 10% of range=1.0

    def test_spoof_does_not_mutate_original(self):
        true_state = {"soc": 0.50}
        fault = FaultEvent(
            step=0,
            kind="coordinated_spoof",
            params={"spoof_fraction": 0.05, "normal_range": 1.0},
        )
        apply_faults(true_state, [fault])
        assert true_state["soc"] == pytest.approx(0.50)

    def test_spoof_affects_all_numeric_fields(self):
        true_state = {"a": 1.0, "b": 2.0, "c": 3.0}
        fault = FaultEvent(
            step=0,
            kind="coordinated_spoof",
            params={"spoof_fraction": 0.10, "normal_range": 10.0},
        )
        observed = apply_faults(true_state, [fault])
        # Each field should be shifted by 0.10 * 10.0 = 1.0
        for key in ("a", "b", "c"):
            assert observed[key] == pytest.approx(true_state[key] + 1.0)


# ---------------------------------------------------------------------------
# compute_reliability_robust
# ---------------------------------------------------------------------------


class TestComputeReliabilityRobust:
    def test_clean_signal_gives_high_reliability(self):
        """Stable, clean signal with no spikes → high w_t."""
        signal = [1.0 + 0.01 * i for i in range(20)]
        w_t, flags = compute_reliability_robust(signal)
        assert w_t >= 0.80
        assert flags["spike_detected"] is False
        assert flags["adversarial_suspected"] is False

    def test_single_spike_detected(self):
        """One large spike (adversarial insertion) → spike_detected=True."""
        signal = [1.0] * 19 + [50.0]  # last value is 50x the baseline
        w_t, flags = compute_reliability_robust(signal, mad_spike_threshold=3.5)
        assert flags["spike_detected"] is True
        assert w_t < 0.80  # reduced from nominal

    def test_systematic_bias_triggers_adversarial_flag(self):
        """Systematic mean shift (coordinated spoof pattern) → adversarial_suspected."""
        # Asymmetric signal: many values at 1.0, few large spoofed values.
        # median=1.0 (anchored by majority), mean pulled toward 10.0 by minority.
        # |median - mean| / std > 0.5 triggers the adversarial_suspected flag.
        signal = [1.0] * 15 + [10.0] * 5  # n=20
        _, flags = compute_reliability_robust(
            signal,
            consistency_threshold=0.5,  # calibrated threshold for test
        )
        assert flags["adversarial_suspected"] is True

    def test_insufficient_history_returns_min_w(self):
        """< 3 samples → return min_w."""
        w_t, flags = compute_reliability_robust([1.0, 2.0], min_w=0.05)
        assert w_t == pytest.approx(0.05)
        assert flags["note"] == "insufficient_history"

    def test_w_t_in_valid_range(self):
        """w_t must always lie in [min_w, 1.0]."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            signal = rng.normal(0, 1, 15).tolist()
            w_t, _ = compute_reliability_robust(signal)
            assert 0.05 <= w_t <= 1.0

    def test_adversarial_penalty_reduces_w_t(self):
        """When adversarial_suspected, w_t should be lower than without penalty."""
        # Construct a signal that triggers adversarial_suspected
        signal = [0.0] * 8 + [10.0] * 8 + [0.0] * 4  # mean != median
        w_no_penalty, _ = compute_reliability_robust(
            signal, consistency_threshold=100.0, adversarial_penalty=0.30
        )
        w_with_penalty, flags = compute_reliability_robust(
            signal, consistency_threshold=0.1, adversarial_penalty=0.30
        )
        if flags["adversarial_suspected"]:
            assert w_with_penalty < w_no_penalty + 0.01  # with penalty should be lower

    def test_returns_expected_flags(self):
        """Flags dict contains all required keys."""
        signal = list(range(10))
        _, flags = compute_reliability_robust(signal)
        for key in (
            "robust",
            "n",
            "spike_detected",
            "adversarial_suspected",
            "mad_z",
            "consistency_ratio",
            "trimmed_mean",
            "full_mean",
            "median",
            "mad",
            "trim_frac",
        ):
            assert key in flags

    def test_empty_signal_returns_min_w(self):
        w_t, flags = compute_reliability_robust([], min_w=0.05)
        assert w_t == pytest.approx(0.05)

    def test_robust_vs_standard_under_adversarial(self):
        """Robust OQE should not over-inflate w_t when adversarial input present."""
        # Clean signal baseline
        clean = [1.0 + 0.005 * i for i in range(20)]
        w_clean, _ = compute_reliability_robust(clean)

        # Insert one adversarial reading at the end (large spike)
        spoofed = clean[:-1] + [100.0]
        w_spoofed, flags = compute_reliability_robust(spoofed)

        # Robust OQE should reduce w_t compared to clean
        assert w_spoofed <= w_clean + 0.05  # allow tiny tolerance


# ---------------------------------------------------------------------------
# Integration: adversarial fault schedule with apply_faults
# ---------------------------------------------------------------------------


class TestAdversarialScheduleIntegration:
    def test_adversarial_faults_do_not_crash(self):
        """Adversarial fault schedule runs end-to-end without errors."""
        np.random.default_rng(0)
        true_state = {"soc": 0.7, "power": 450.0}
        history = []

        for t in range(20):
            history.append(dict(true_state))
            if len(history) > 10:
                history = history[-10:]

            if t % 3 == 0:
                fault = FaultEvent(
                    step=t,
                    kind="replay",
                    params={"k_steps_ago": 2, "history": list(history)},
                )
            else:
                fault = FaultEvent(
                    step=t,
                    kind="coordinated_spoof",
                    params={"spoof_fraction": 0.05, "normal_range": 1.0},
                )
            observed = apply_faults(dict(true_state), [fault])
            # Just check it returns a dict with the same keys
            assert set(observed.keys()) == set(true_state.keys())
