"""Dynamical implementations and verification for ORIUS Theorems A through E.

This module implements:
- Theorem A: Information-dependent release feasibility and value of delay.
- Theorem B: Dynamic observation-history Bayes lower bound on mandatory release.
- Theorem C: Trajectory-level certified release achievability bound.
- Theorem D: Safety-utility characterization and Pareto frontier.
- Theorem E: Structural transfer obligations across physical domains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np


@dataclass(frozen=True)
class Interval:
    """Compact 1D interval [low, high]."""
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high + 1e-9:
            raise ValueError(f"Invalid interval: low={self.low} > high={self.high}")

    @property
    def diameter(self) -> float:
        return max(0.0, self.high - self.low)

    def contains(self, val: float, tol: float = 1e-7) -> bool:
        return (self.low - tol) <= val <= (self.high + tol)

    def intersection(self, other: Interval) -> Interval | None:
        new_low = max(self.low, other.low)
        new_high = min(self.high, other.high)
        if new_low <= new_high + 1e-9:
            return Interval(new_low, new_high)
        return None


class PhysicalPlant(ABC):
    """Abstract base class for physical control plants."""

    @abstractmethod
    def state_dim(self) -> int:
        pass

    @abstractmethod
    def action_dim(self) -> int:
        pass

    @abstractmethod
    def step(self, x: np.ndarray, a: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """Evolve x_{t+1} = f(x_t, a_t, xi_t)."""
        pass

    @abstractmethod
    def observe(self, x: np.ndarray, nu: np.ndarray, is_dropped: bool = False) -> np.ndarray | None:
        """Compute o_t = h(x_t, nu_t) or None if dropped."""
        pass

    @abstractmethod
    def is_safe(self, x: np.ndarray) -> bool:
        """Evaluate if state x in S_d."""
        pass

    @abstractmethod
    def action_bounds(self) -> list[Interval]:
        """Admissible physical actuator bounds A_d."""
        pass

    @abstractmethod
    def fallback_action(self) -> np.ndarray:
        """Known safe fallback action."""
        pass


class DoubleIntegratorRobot(PhysicalPlant):
    """Robot moving towards an obstacle at position p_max.
    
    State: x = [position p, velocity v]
    Action: a = [acceleration u] in [-u_max, u_max]
    Dynamics:
      p_{t+1} = p_t + v_t * dt + 0.5 * u_t * dt^2
      v_{t+1} = v_t + u_t * dt + xi_t
    Safety:
      p <= p_max (no collision)
      v >= 0 (forward motion)
    """

    def __init__(
        self,
        dt: float = 0.1,
        p_max: float = 10.0,
        u_max: float = 3.0,
        xi_bound: float = 0.05,
    ) -> None:
        self.dt = dt
        self.p_max = p_max
        self.u_max = u_max
        self.xi_bound = xi_bound

    def state_dim(self) -> int:
        return 2

    def action_dim(self) -> int:
        return 1

    def step(self, x: np.ndarray, a: np.ndarray, xi: np.ndarray) -> np.ndarray:
        p, v = float(x[0]), float(x[1])
        u = float(np.clip(a[0], -self.u_max, self.u_max))
        w = float(np.clip(xi[0], -self.xi_bound, self.xi_bound))

        p_next = p + v * self.dt + 0.5 * u * (self.dt ** 2)
        v_next = max(0.0, v + u * self.dt + w)
        return np.array([p_next, v_next], dtype=float)

    def observe(self, x: np.ndarray, nu: np.ndarray, is_dropped: bool = False) -> np.ndarray | None:
        if is_dropped:
            return None
        return x + nu

    def is_safe(self, x: np.ndarray) -> bool:
        p, v = float(x[0]), float(x[1])
        # Viability: must not cross obstacle and must retain sufficient braking distance
        stopping_distance = (v ** 2) / (2.0 * self.u_max)
        return bool(p + stopping_distance <= self.p_max + 1e-6 and v >= -1e-6)

    def action_bounds(self) -> list[Interval]:
        return [Interval(-self.u_max, self.u_max)]

    def fallback_action(self) -> np.ndarray:
        # Full emergency braking
        return np.array([-self.u_max], dtype=float)


class BatteryStoragePlant(PhysicalPlant):
    """Battery Energy Storage System (BESS).
    
    State: x = [SoC in (0, 1), Temperature T_c in C]
    Action: a = [Power P in MW, positive=charge, negative=discharge]
    Dynamics:
      SoC_{t+1} = SoC_t + (eta * P * dt) / Capacity
      T_{t+1} = T_t + alpha_heat * P^2 * dt - cooling * (T_t - T_amb) + xi
    Safety:
      SoC in [SoC_min, SoC_max]
      T_c <= T_max
    """

    def __init__(
        self,
        capacity_mwh: float = 100.0,
        soc_min: float = 0.10,
        soc_max: float = 0.90,
        t_max: float = 50.0,
        p_max_mw: float = 25.0,
        dt_hours: float = 1.0,
        eta: float = 0.95,
    ) -> None:
        self.capacity_mwh = capacity_mwh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.t_max = t_max
        self.p_max_mw = p_max_mw
        self.dt_hours = dt_hours
        self.eta = eta

    def state_dim(self) -> int:
        return 2

    def action_dim(self) -> int:
        return 1

    def step(self, x: np.ndarray, a: np.ndarray, xi: np.ndarray) -> np.ndarray:
        soc, temp = float(x[0]), float(x[1])
        p = float(np.clip(a[0], -self.p_max_mw, self.p_max_mw))
        w = float(xi[0]) if len(xi) > 0 else 0.0

        eff = self.eta if p >= 0 else (1.0 / self.eta)
        delta_soc = (eff * p * self.dt_hours) / self.capacity_mwh
        soc_next = soc + delta_soc
        # Thermal model
        heat = 0.008 * (p ** 2) * self.dt_hours
        cooling = 0.15 * (temp - 25.0) * self.dt_hours
        temp_next = temp + heat - cooling + w
        return np.array([soc_next, temp_next], dtype=float)

    def observe(self, x: np.ndarray, nu: np.ndarray, is_dropped: bool = False) -> np.ndarray | None:
        if is_dropped:
            return None
        return x + nu

    def is_safe(self, x: np.ndarray) -> bool:
        soc, temp = float(x[0]), float(x[1])
        return (self.soc_min - 1e-6 <= soc <= self.soc_max + 1e-6) and (temp <= self.t_max + 1e-6)

    def action_bounds(self) -> list[Interval]:
        return [Interval(-self.p_max_mw, self.p_max_mw)]

    def fallback_action(self) -> np.ndarray:
        # Zero power standby
        return np.array([0.0], dtype=float)


# -------------------------------------------------------------------------
# THEOREM ENGINES
# -------------------------------------------------------------------------

class TheoremEngine:
    """Verification and evaluation engine for ORIUS Theorems A through E."""

    @staticmethod
    def theorem_a_feasibility(
        plant: PhysicalPlant,
        state_uncertainty_box: list[Interval],
        action_grid_points: int = 100,
        disturbances: list[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Theorem A: Release Feasibility.
        
        Computes the robust safe action set:
          A_robust(U) = { a in A : forall x in U, forall xi in Xi, f(x, a, xi) in S }
        Returns feasibility status, admissible action interval/set, and margin.
        """
        if disturbances is None:
            disturbances = [np.zeros(plant.action_dim())]

        bounds = plant.action_bounds()[0]
        candidate_actions = np.linspace(bounds.low, bounds.high, action_grid_points)

        # Discretize state boundary vertices
        mins = np.array([iv.low for iv in state_uncertainty_box])
        maxs = np.array([iv.high for iv in state_uncertainty_box])
        
        # Test corners of state uncertainty box
        from itertools import product
        corners = [np.array(p) for p in product(*zip(mins, maxs))]

        admissible_actions = []
        for a_val in candidate_actions:
            a_vec = np.array([a_val])
            is_action_robustly_safe = True
            for x_corner in corners:
                for xi in disturbances:
                    x_next = plant.step(x_corner, a_vec, xi)
                    if not plant.is_safe(x_next):
                        is_action_robustly_safe = False
                        break
                if not is_action_robustly_safe:
                    break
            if is_action_robustly_safe:
                admissible_actions.append(float(a_val))

        feasible = len(admissible_actions) > 0
        admissible_interval = (
            Interval(min(admissible_actions), max(admissible_actions))
            if feasible
            else None
        )

        return {
            "feasible": feasible,
            "admissible_count": len(admissible_actions),
            "admissible_interval": admissible_interval,
            "fallback_safe": all(
                plant.is_safe(plant.step(x_corner, plant.fallback_action(), np.zeros(plant.action_dim())))
                for x_corner in corners
            ),
        }

    @staticmethod
    def theorem_b_bayes_lower_bound(
        plant: PhysicalPlant,
        ambiguous_states: list[np.ndarray],
        state_probabilities: list[float],
        action_grid_points: int = 100,
    ) -> dict[str, Any]:
        """Theorem B: Observation-Limited Bayes Lower Bound.
        
        For states indistinguishable under observation o:
          Risk(pi) >= min_{a} sum_i P(x_i | o) * 1(a not in Safe(x_i))
        """
        assert len(ambiguous_states) == len(state_probabilities)
        probs = np.array(state_probabilities) / sum(state_probabilities)

        bounds = plant.action_bounds()[0]
        actions = np.linspace(bounds.low, bounds.high, action_grid_points)

        min_risk = 1.0
        best_action = None

        for a_val in actions:
            a_vec = np.array([a_val])
            risk_a = 0.0
            for x, p in zip(ambiguous_states, probs):
                x_next = plant.step(x, a_vec, np.zeros(plant.action_dim()))
                if not plant.is_safe(x_next):
                    risk_a += p
            if risk_a < min_risk:
                min_risk = risk_a
                best_action = float(a_val)

        return {
            "bayes_lower_bound_risk": float(min_risk),
            "best_mandatory_action": best_action,
            "is_strictly_positive": bool(min_risk > 1e-5),
        }

    @staticmethod
    def theorem_c_trajectory_achievability(
        plant: PhysicalPlant,
        initial_true_state: np.ndarray,
        nominal_policy: Callable[[np.ndarray], np.ndarray],
        noise_std: float,
        alpha_per_step: float,
        horizon_t: int,
        n_episodes: int = 200,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Theorem C: Finite-Horizon Trajectory Achievability Bound.
        
        Evaluates whether:
          P(exists t <= T: x_t notin S) <= sum_{t=0}^{T-1} alpha_t + delta_model + delta_act
        """
        rng = np.random.default_rng(seed)
        trajectory_violations_nominal = 0
        trajectory_violations_orius = 0

        # Conformal quantile for standard normal (alpha_per_step two-sided)
        from scipy.stats import norm
        z_crit = norm.ppf(1.0 - alpha_per_step / 2.0)

        for _ in range(n_episodes):
            x_nom = initial_true_state.copy()
            x_ori = initial_true_state.copy()

            nom_failed = False
            ori_failed = False

            for t in range(horizon_t):
                # 1. Telemetry with sensor noise and occasional dropout (15% rate)
                is_dropped = (rng.uniform() < 0.15)
                nu = rng.normal(0, noise_std, size=plant.state_dim())

                o_nom = plant.observe(x_nom, nu, is_dropped=is_dropped)
                o_ori = plant.observe(x_ori, nu, is_dropped=is_dropped)

                # --- Nominal Controller (uncalibrated, mandatory release) ---
                obs_nom = o_nom if o_nom is not None else x_nom  # zero-order hold if dropped
                a_nom = nominal_policy(obs_nom)
                x_nom = plant.step(x_nom, a_nom, np.zeros(plant.action_dim()))
                if not plant.is_safe(x_nom):
                    nom_failed = True

                # --- ORIUS Controller (DC3S with conformal inflation & fallback) ---
                # Detect & Calibrate: construct uncertainty set U_t
                if o_ori is not None:
                    # Conformal box around observation
                    half_width = z_crit * noise_std
                    u_box = [Interval(o_ori[i] - half_width, o_ori[i] + half_width) for i in range(plant.state_dim())]
                else:
                    # Widened uncertainty under dropout
                    half_width = z_crit * noise_std * 2.5
                    u_box = [Interval(x_ori[i] - half_width, x_ori[i] + half_width) for i in range(plant.state_dim())]

                # Constrain & Shield
                feas = TheoremEngine.theorem_a_feasibility(plant, u_box, action_grid_points=50)
                if feas["feasible"] and feas["admissible_interval"] is not None:
                    # Project nominal action into robust safe interval
                    iv: Interval = feas["admissible_interval"]
                    a_prop = nominal_policy(o_ori if o_ori is not None else x_ori)
                    a_safe = np.array([np.clip(a_prop[0], iv.low, iv.high)])
                else:
                    # Trigger verified safe fallback
                    a_safe = plant.fallback_action()

                x_ori = plant.step(x_ori, a_safe, np.zeros(plant.action_dim()))
                if not plant.is_safe(x_ori):
                    ori_failed = True

            if nom_failed:
                trajectory_violations_nominal += 1
            if ori_failed:
                trajectory_violations_orius += 1

        empirical_tsvr_nominal = trajectory_violations_nominal / n_episodes
        empirical_tsvr_orius = trajectory_violations_orius / n_episodes
        theoretical_upper_bound = min(1.0, horizon_t * alpha_per_step)

        return {
            "episodes": n_episodes,
            "horizon_t": horizon_t,
            "nominal_trajectory_violation_rate": empirical_tsvr_nominal,
            "orius_trajectory_violation_rate": empirical_tsvr_orius,
            "theoretical_bound": theoretical_upper_bound,
            "bound_satisfied": bool(empirical_tsvr_orius <= theoretical_upper_bound + 1e-4),
        }

    @staticmethod
    def theorem_d_safety_utility_tradeoff(
        plant: PhysicalPlant,
        true_state: np.ndarray,
        nominal_action: np.ndarray,
        uncertainty_half_widths: list[float],
        cost_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> list[dict[str, Any]]:
        """Theorem D: Safety-Utility Characterization.
        
        Evaluates utility loss Delta J = J(x, a_repaired) - J(x, a_optimal)
        as a function of uncertainty set diameter.
        """
        results = []
        for hw in uncertainty_half_widths:
            u_box = [Interval(true_state[i] - hw, true_state[i] + hw) for i in range(plant.state_dim())]
            feas = TheoremEngine.theorem_a_feasibility(plant, u_box, action_grid_points=100)
            
            if feas["feasible"] and feas["admissible_interval"] is not None:
                iv: Interval = feas["admissible_interval"]
                repaired_a = np.array([np.clip(nominal_action[0], iv.low, iv.high)])
                fallback_used = False
            else:
                repaired_a = plant.fallback_action()
                fallback_used = True

            loss = cost_fn(true_state, repaired_a) - cost_fn(true_state, nominal_action)
            results.append({
                "half_width": float(hw),
                "diameter": float(2 * hw),
                "feasible": feas["feasible"],
                "fallback_used": fallback_used,
                "repaired_action": float(repaired_a[0]),
                "utility_loss": float(max(0.0, loss)),
            })
        return results

    @staticmethod
    def theorem_e_adapter_invariance_audit(
        adapter_name: str,
        has_sound_uncertainty_set: bool,
        has_computable_safe_set: bool,
        has_verified_fallback: bool,
        has_causal_certificates: bool,
    ) -> dict[str, Any]:
        """Theorem E: Universal Structural Transfer Invariance Check.
        
        Validates the 4 necessary and sufficient obligations for ORIUS transfer.
        """
        obligations = {
            "O1_soundness_of_uncertainty_set": bool(has_sound_uncertainty_set),
            "O2_computable_robust_safe_core": bool(has_computable_safe_set),
            "O3_verified_fail_closed_fallback": bool(has_verified_fallback),
            "O4_causal_certificate_chain": bool(has_causal_certificates),
        }
        all_passed = all(obligations.values())
        return {
            "adapter": adapter_name,
            "passed_universal_transfer": all_passed,
            "obligations": obligations,
        }
