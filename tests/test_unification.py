"""
Tests for the reference unification reductions (orius/universal/unification.py).

Verifies that CBFAsORIUS and RobustMPCAsORIUS:
  1. Pass all five T11 invariant checks (ContractVerifier.check())
  2. Reproduce the expected behaviour of the underlying prior frameworks
  3. Demonstrate the limitation (constant / fixed w_t) that ORIUS generalises
"""
import numpy as np
import pytest

from orius.universal.contract import ContractVerifier, UniversalAdapterProtocol
from orius.universal.unification import CBFAsORIUS, RobustMPCAsORIUS


# ── Barrier function fixture ───────────────────────────────────────────────────

def barrier_h(x: np.ndarray) -> float:
    """Simple 1-D barrier: h(x) = 1 − |x[0]|.  Safe set: |x| ≤ 1."""
    return float(1.0 - abs(x[0]))


# ── CBFAsORIUS tests ───────────────────────────────────────────────────────────

class TestCBFAsORIUS:

    def setup_method(self):
        self.adapter = CBFAsORIUS(h_fn=barrier_h, gamma=1.0, action_dim=1)
        self.verifier = ContractVerifier(self.adapter, alpha=0.05, epsilon=1e-6)
        self.z_safe = np.array([0.3])    # h = 0.7 > 0 → inside safe set
        self.z_unsafe = np.array([0.98]) # h = 0.02 → near boundary
        self.q_t = 0.05

    def test_passes_all_five_invariants_safe_state(self):
        """At a safe state, CBFAsORIUS passes all T11 invariant checks."""
        self.verifier.check(self.z_safe, q_t=self.q_t)

    def test_passes_all_five_invariants_near_boundary(self):
        """Near the constraint boundary, CBFAsORIUS still passes all invariants."""
        self.verifier.check(self.z_unsafe, q_t=self.q_t)

    def test_implements_universal_adapter_protocol(self):
        assert isinstance(self.adapter, UniversalAdapterProtocol)

    def test_observe_returns_unit_reliability(self):
        """CBF always returns w_t = 1.0 — the quality-ignorant assumption."""
        z_t, w_t = self.adapter.observe({"state": [0.5]})
        assert w_t == 1.0, f"CBF should return w_t=1.0, got {w_t}"

    def test_repair_clips_to_safe_set(self):
        """CBF-QP projection clips an out-of-bounds candidate."""
        safe_set = self.adapter.uncertainty_set(self.z_safe, w_t=0.5, q_t=self.q_t)
        if not safe_set.is_empty:
            out_of_bounds = safe_set.upper + 2.0  # deliberately outside
            result = self.adapter.repair(out_of_bounds, safe_set)
            assert np.all(result.action <= safe_set.upper + 1e-9)
            assert np.all(result.action >= safe_set.lower - 1e-9)
            assert result.was_repaired

    def test_fallback_is_zero(self):
        """CBF fallback is zero action (stop)."""
        fb = self.adapter.fallback()
        assert np.allclose(fb, np.zeros(1))

    def test_unification_argument_w_t_never_drops_below_one(self):
        """Demonstrate the CBF limitation: w_t is always 1.0 regardless of fault."""
        for fault_scenario in [
            {"state": [0.5]},                          # normal telemetry
            {"state": [0.5], "dropout": True},         # dropout fault
            {"state": [0.5], "stale": True, "age": 5}, # stale measurement
        ]:
            _, w_t = self.adapter.observe(fault_scenario)
            assert w_t == 1.0, (
                f"CBFAsORIUS always returns w_t=1.0, but got {w_t} under {fault_scenario}. "
                "This demonstrates why CBF fails under telemetry degradation (T9)."
            )

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            CBFAsORIUS(h_fn=barrier_h, gamma=-1.0)


# ── RobustMPCAsORIUS tests ────────────────────────────────────────────────────

class TestRobustMPCAsORIUS:

    def setup_method(self):
        self.adapter = RobustMPCAsORIUS(
            tube_radius=0.1,
            max_radius=1.0,
            action_lower=-1.0,
            action_upper=1.0,
            action_dim=1,
        )
        self.verifier = ContractVerifier(self.adapter, alpha=0.05, epsilon=1e-6)
        self.z_t = np.array([0.5])
        self.q_t = 0.1

    def test_passes_all_five_invariants(self):
        """RobustMPCAsORIUS with r=0.1 passes all T11 invariant checks."""
        self.verifier.check(self.z_t, q_t=self.q_t)

    def test_implements_universal_adapter_protocol(self):
        assert isinstance(self.adapter, UniversalAdapterProtocol)

    def test_constant_reliability_matches_tube_formula(self):
        """w_t = 1 − r/r_max for all inputs."""
        expected_w = 1.0 - 0.1 / 1.0  # = 0.9
        for raw in [{"state": [0.0]}, {"state": [0.5]}, {"state": [-0.9]}]:
            _, w_t = self.adapter.observe(raw)
            assert abs(w_t - expected_w) < 1e-9, (
                f"Expected constant w_t={expected_w}, got {w_t}. "
                "RobustMPC uses a fixed reliability (the unification point)."
            )

    def test_tightened_set_is_narrower_than_nominal(self):
        """At operating w_t < 1, tightened set ⊆ nominal set."""
        w_nominal = 1.0
        w_operating = self.adapter.constant_w
        nominal = self.adapter.uncertainty_set(self.z_t, w_t=w_nominal, q_t=self.q_t)
        operating = self.adapter.uncertainty_set(self.z_t, w_t=w_operating, q_t=self.q_t)
        if not operating.is_empty:
            assert np.all(operating.lower >= nominal.lower - 1e-9)
            assert np.all(operating.upper <= nominal.upper + 1e-9)

    def test_repair_clips_to_tightened_set(self):
        safe_set = self.adapter.uncertainty_set(self.z_t, w_t=self.adapter.constant_w,
                                               q_t=self.q_t)
        if not safe_set.is_empty:
            outside = safe_set.upper + 5.0
            result = self.adapter.repair(outside, safe_set)
            assert np.all(result.action <= safe_set.upper + 1e-9)
            assert np.all(result.action >= safe_set.lower - 1e-9)

    def test_fallback_is_midpoint(self):
        fb = self.adapter.fallback()
        expected_midpoint = 0.0  # midpoint of [-1, 1]
        assert np.allclose(fb, np.array([expected_midpoint]))

    def test_unification_argument_same_w_t_regardless_of_fault(self):
        """Demonstrate the Robust MPC limitation: w_t is constant."""
        expected_w = self.adapter.constant_w
        for scenario in [
            {"state": [0.0]},
            {"state": [0.0], "dropout": True},
            {"state": [0.0], "spike": 0.5},
        ]:
            _, w_t = self.adapter.observe(scenario)
            assert abs(w_t - expected_w) < 1e-9, (
                f"RobustMPCAsORIUS always returns w_t={expected_w}, got {w_t}. "
                "This is the Robust MPC assumption — ORIUS generalises by using dynamic w_t."
            )

    def test_invalid_tube_radius_raises(self):
        with pytest.raises(ValueError):
            RobustMPCAsORIUS(tube_radius=-0.1, max_radius=1.0,
                             action_lower=-1.0, action_upper=1.0)

    def test_tube_exceeds_max_raises(self):
        with pytest.raises(ValueError):
            RobustMPCAsORIUS(tube_radius=1.5, max_radius=1.0,
                             action_lower=-1.0, action_upper=1.0)


# ── Cross-framework unification check ─────────────────────────────────────────

class TestUnificationInterpretation:
    """Verify the formal unification claim at the level of ContractVerifier."""

    def test_cbf_and_robust_mpc_both_pass_t11_contract(self):
        """Both prior frameworks satisfy T11 — they are valid DC3S adapters."""
        cbf = CBFAsORIUS(h_fn=barrier_h, gamma=1.0)
        rmpc = RobustMPCAsORIUS(0.1, 1.0, -1.0, 1.0)
        z_t = np.array([0.3])
        q_t = 0.08

        for adapter in [cbf, rmpc]:
            verifier = ContractVerifier(adapter, alpha=0.05)
            verifier.check(z_t, q_t)  # must not raise

    def test_cbf_has_higher_mean_reliability_than_robust_mpc(self):
        """CBF (w_t=1) is more optimistic about observation quality than Robust MPC."""
        cbf = CBFAsORIUS(h_fn=barrier_h)
        rmpc = RobustMPCAsORIUS(0.1, 1.0, -1.0, 1.0)
        raw = {"state": [0.3]}

        _, w_cbf = cbf.observe(raw)
        _, w_rmpc = rmpc.observe(raw)

        assert w_cbf == 1.0
        assert w_rmpc < 1.0
        assert w_cbf > w_rmpc, (
            "CBF assumes perfect reliability (w=1.0); Robust MPC assumes "
            "constant degraded reliability — both are special cases of ORIUS."
        )
