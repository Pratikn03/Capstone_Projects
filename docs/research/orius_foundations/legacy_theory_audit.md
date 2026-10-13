# Audit of Legacy Observation-Ambiguity, Minimax, and Sensor-Necessity Results
**Issue OR-004 Milestone Audit**

---

## 1. Executive Summary

This audit assesses the pre-existing theoretical code located in:
1. `src/orius/universal_theory/observation_ambiguity.py`
2. `src/orius/universal_theory/minimax_boundary.py`
3. `src/orius/universal_theory/sensor_necessity.py`

### Key Assessment
The legacy modules implemented **discrete, finite-state combinatorial evaluations** where states $x \in \mathcal{X}$ and actions $a \in \mathcal{A}$ were treated as arbitrary hashable objects, and safe sets were supplied via pre-tabulated lookups (`Mapping[State, Iterable[Action]]`). While mathematically valid as combinatorial toy models, they fell short of the 9.8 foundational standard because:
1. They lacked physical state dynamics ($x_{k+1} = f(x_k, u_k)$).
2. They assumed static one-step safe sets rather than recursive dynamic viability kernels $\mathcal{V}$.
3. Their bounds evaluated pre-supplied formulas rather than deriving non-asymptotic trajectory-level risk certificates under continuous observation noise.

---

## 2. Component-by-Component Comparison

| Dimension | Legacy Implementation (`legacy/*`) | Upgraded Foundational Engine (`src/orius/universal_theory/dynamical_theorems.py`) |
| :--- | :--- | :--- |
| **State Space $\mathcal{X}$** | Discrete finite set: `{s_1, s_2, \dots, s_k}` | Continuous dynamical space: $\mathcal{X} \subseteq \mathbb{R}^n$ (e.g. $[p, v]^\top$, $[\text{SoC}, T]^\top$, $[f, v]^\top$) |
| **Action Space $\mathcal{U}$** | Discrete finite choices: `{a_1, a_2}` | Continuous bounded manifold: $\mathcal{U} \subseteq \mathbb{R}^m$ (e.g. $[-u_{\max}, u_{\max}]$) |
| **Dynamics** | None (static one-step mapping) | Continuous differential / difference equations: $x_{t+1} = f(x_t, u_t) + w_t$ |
| **Safety Characterization** | Pre-computed static membership ($a \in \text{Safe}(s)$) | Dynamic viability invariance: $\mathcal{V} \subseteq \mathcal{S}_d$, stopping distance barriers, and Lyapunov/droop envelopes |
| **Minimax Bound** | Combinatorial intersection test ($S(x_1) \cap S(x_2) = \emptyset$) | Theorem B: Analytical continuous Radon-Nikodym / Bayes risk lower bound $\mathcal{R}^* \ge \min(\pi_1, \pi_2)$ |
| **Trajectory Risk** | Static coverage evaluate $\alpha$ | Theorem C: Non-asymptotic trajectory union bound $\Pr(\exists t \le T: x_t \notin \mathcal{S}_d) \le \sum \alpha_t + \delta_{\mathrm{model}} + \delta_{\mathrm{act}}$ |
| **Formal Mechanization** | Definitional tautologies in earlier Lean sketches | Lean 4 machine-verified proofs in `formal/Orius/TheoremsAE.lean` without `sorry` or definitional circularity |

---

## 3. Disposition and Integration

1. **Retain Legacy Modules as Discrete Toy Witnesses**: `observation_ambiguity.py` and `sensor_necessity.py` remain preserved for backwards compatibility and unit test sanity checks over discrete toy graphs.
2. **Promote `dynamical_theorems.py` as Authoritative**: All foundational claims, empirical benchmarks, and flagship paper submissions cite the continuous dynamical formulation and its Lean 4 mechanization.
