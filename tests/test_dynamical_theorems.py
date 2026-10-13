"""Tests for ORIUS Dynamical Theorems A through E."""

from __future__ import annotations

import numpy as np
import pytest

from orius.universal_theory.dynamical_theorems import (
    BatteryStoragePlant,
    DoubleIntegratorRobot,
    Interval,
    TheoremEngine,
)


def test_interval_logic() -> None:
    iv1 = Interval(1.0, 5.0)
    iv2 = Interval(3.0, 7.0)
    inter = iv1.intersection(iv2)
    assert inter is not None
    assert inter.low == 3.0 and inter.high == 5.0
    assert iv1.diameter == 4.0
    assert iv1.contains(3.5)
    assert not iv1.contains(6.0)

    iv3 = Interval(6.0, 9.0)
    assert iv1.intersection(iv3) is None


def test_double_integrator_robot_step() -> None:
    robot = DoubleIntegratorRobot(dt=0.1, p_max=10.0, u_max=3.0)
    x0 = np.array([2.0, 1.0])
    u = np.array([2.0])
    xi = np.array([0.0])
    x1 = robot.step(x0, u, xi)
    # p1 = 2.0 + 1.0*0.1 + 0.5*2.0*0.01 = 2.11
    # v1 = 1.0 + 2.0*0.1 = 1.2
    assert pytest.approx(x1[0], 1e-4) == 2.11
    assert pytest.approx(x1[1], 1e-4) == 1.2
    assert robot.is_safe(x1)


def test_battery_storage_plant_step() -> None:
    batt = BatteryStoragePlant(capacity_mwh=100.0, p_max_mw=25.0)
    x0 = np.array([0.5, 30.0])
    a = np.array([10.0])  # Charge 10 MW
    x1 = batt.step(x0, a, np.zeros(1))
    assert x1[0] > x0[0]  # SoC increases
    assert batt.is_safe(x1)


def test_theorem_a_feasibility() -> None:
    robot = DoubleIntegratorRobot(dt=0.1, p_max=10.0, u_max=3.0)

    # 1. Clear safe zone (far from obstacle p_max=10)
    safe_box = [Interval(2.0, 3.0), Interval(0.5, 1.5)]
    res_safe = TheoremEngine.theorem_a_feasibility(robot, safe_box)
    assert res_safe["feasible"] is True
    assert res_safe["admissible_interval"] is not None

    # 2. Critical edge zone where fallback IS still safe (viable set boundary)
    # p in [8.5, 9.0], v in [1.0, 1.5]. With u=-3.0, p_next = 9.0 + 0.15 - 0.015 = 9.135 <= 10.0
    viable_edge_box = [Interval(8.5, 9.0), Interval(1.0, 1.5)]
    res_edge = TheoremEngine.theorem_a_feasibility(robot, viable_edge_box)
    assert res_edge["fallback_safe"] is True

    # 3. Non-viable zone (too close with too high velocity: even full braking fails)
    non_viable_box = [Interval(9.9, 10.0), Interval(2.5, 3.0)]
    res_non_viable = TheoremEngine.theorem_a_feasibility(robot, non_viable_box)
    assert res_non_viable["feasible"] is False
    assert res_non_viable["fallback_safe"] is False


def test_theorem_b_bayes_lower_bound() -> None:
    robot = DoubleIntegratorRobot(dt=0.1, p_max=10.0, u_max=3.0)

    # Two states that are indistinguishable:
    # State 1: p=9.9, v=2.5 (requires strong braking u <= -2.0 to not hit p_max)
    # State 2: p=0.0, v=0.0 (stationary safe, but if forced forward, safe)
    # Now consider a third state that is in a hazard condition
    x1 = np.array([9.95, 2.8])  # Needs maximum brake
    x2 = np.array([10.05, 0.5])  # Already slightly beyond boundary
    res = TheoremEngine.theorem_b_bayes_lower_bound(
        robot, [x1, x2], [0.5, 0.5], action_grid_points=50
    )
    # Since x2 is already unsafe, min_risk must be at least 0.5
    assert res["bayes_lower_bound_risk"] >= 0.5
    assert res["is_strictly_positive"] is True


def test_theorem_c_trajectory_achievability() -> None:
    robot = DoubleIntegratorRobot(dt=0.1, p_max=10.0, u_max=3.0)

    def aggressive_policy(obs: np.ndarray) -> np.ndarray:
        # Aggressive forward driver
        return np.array([3.0])

    # Start close to obstacle so aggressive policy causes collision within horizon
    initial_state = np.array([8.0, 2.0])
    res = TheoremEngine.theorem_c_trajectory_achievability(
        plant=robot,
        initial_true_state=initial_state,
        nominal_policy=aggressive_policy,
        noise_std=0.2,
        alpha_per_step=0.05,
        horizon_t=10,
        n_episodes=50,
        seed=123,
    )
    # Nominal policy pushes car into wall, so nominal violation is high
    assert res["nominal_trajectory_violation_rate"] > 0.0
    # ORIUS shield intervenes and enforces safety
    assert res["orius_trajectory_violation_rate"] <= res["theoretical_bound"]
    assert res["bound_satisfied"] is True


def test_theorem_d_safety_utility_tradeoff() -> None:
    batt = BatteryStoragePlant()
    true_state = np.array([0.88, 35.0])  # Near SoC max = 0.90
    nominal_action = np.array([20.0])   # Desired large charge

    def economic_cost(x: np.ndarray, a: np.ndarray) -> float:
        # Penalize deviation from nominal profit
        return float((a[0] - 20.0) ** 2)

    half_widths = [0.005, 0.02, 0.05]
    tradeoffs = TheoremEngine.theorem_d_safety_utility_tradeoff(
        batt, true_state, nominal_action, half_widths, economic_cost
    )
    assert len(tradeoffs) == 3
    # As uncertainty width increases, utility loss should increase or remain non-negative
    losses = [t["utility_loss"] for t in tradeoffs]
    assert all(loss >= 0.0 for loss in losses)
    assert losses[-1] >= losses[0]


def test_theorem_e_adapter_invariance() -> None:
    # 1. Battery Adapter passes all 4 obligations
    res_bess = TheoremEngine.theorem_e_adapter_invariance_audit(
        adapter_name="BatteryStorageAdapter",
        has_sound_uncertainty_set=True,
        has_computable_safe_set=True,
        has_verified_fallback=True,
        has_causal_certificates=True,
    )
    assert res_bess["passed_universal_transfer"] is True

    # 2. Defective Adapter with missing fallback fails
    res_bad = TheoremEngine.theorem_e_adapter_invariance_audit(
        adapter_name="UnsafeAdapter",
        has_sound_uncertainty_set=True,
        has_computable_safe_set=True,
        has_verified_fallback=False,
        has_causal_certificates=True,
    )
    assert res_bad["passed_universal_transfer"] is False
    assert res_bad["obligations"]["O3_verified_fail_closed_fallback"] is False
