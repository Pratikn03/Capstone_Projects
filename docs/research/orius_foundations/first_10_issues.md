# First Ten Foundational Research Issues (OR-001 to OR-010)

These ten issues form the immediate milestone execution gate for the **ORIUS 9.8 Foundational Research Roadmap**.

---

### [OR-001] Freeze the current promoted ORIUS claims and artifact manifest
* **Objective**: Establish an immutable baseline of current claims, code versions, and reports.
* **Deliverable**: `docs/research/orius_foundations/research_charter.md` and `reports/orius_foundations/reproduction_manifest.json`.
* **Pass Condition**: Clean audit of all repo claims; explicit separation of proved, simulated, and physical evidence.
* **Status**: **COMPLETED** (Research Charter signed, cryptographic SHA-256 manifest generated).

---

### [OR-002] Create a prior-art matrix against closest safety-filter and belief-space papers
* **Objective**: Benchmark ORIUS against *Safety Filter: A Unified View* (2024), *Safety-critical Control Under Partial Observability* (2026), and *Control Barrier–Value Functions under Partial Observability* (2026).
* **Deliverable**: `docs/research/orius_foundations/novelty_matrix.md`.
* **Pass Condition**: Identify at least one substantive theorem and one practical mechanism competitors cannot provide.
* **Status**: **COMPLETED** (Detailed 5-paradigm differentiation matrix).

---

### [OR-003] Choose the exact flagship foundational theorem
* **Objective**: State the central mathematical theorem (Theorems A–E):
  * Information-dependent release feasibility (Theorem A).
  * Observation-limited safety lower bound (Theorem B).
  * Certified ORIUS achievability bound (Theorem C).
  * Pareto regret envelope (Theorem D).
  * Universal transferability (Theorem E).
* **Deliverable**: `docs/research/orius_foundations/theorems_foundations.md` and `docs/research/orius_foundations/theorem_dependency_graph.md`.
* **Pass Condition**: Formal mathematical statement with explicit assumptions, non-vacuous counterexamples, and physical interpretation.
* **Status**: **COMPLETED** (Theorems A–E derived, computationally verified, and mapped in dependency graph).

---

### [OR-004] Audit existing observation-ambiguity, minimax, and sensor-necessity results for novelty
* **Objective**: Evaluate `src/orius/universal_theory/observation_ambiguity.py`, `minimax_boundary.py`, and `sensor_necessity.py`.
* **Deliverable**: `docs/research/orius_foundations/legacy_theory_audit.md`.
* **Pass Condition**: Every claim in code is backed by an analytical derivation, not an assumed property.
* **Status**: **COMPLETED** (Detailed audit completed, continuous dynamical models in `dynamical_theorems.py` promoted).

---

### [OR-005] List unproved Lean and calibration-to-risk obligations
* **Objective**: Review `formal/Orius/T2T4T11.lean` to eliminate circular definitions where set membership assumes safety.
* **Deliverable**: `docs/research/orius_foundations/proof_obligation_matrix.md` and `formal/Orius/TheoremsAE.lean`.
* **Pass Condition**: Clear catalog of remaining open proof obligations in Lean 4 without unacknowledged `sorry` or axiomatic trivialization.
* **Status**: **COMPLETED** (Lean 4 machine-verified proofs in `TheoremsAE.lean` compiling cleanly with 0 `sorry`).

---

### [OR-006] Build true reruns for shuffled, delayed, and constant reliability controls
* **Objective**: Execute genuine closed-loop controller reruns with perturbed reliability scores ($w_t$ shuffled, time-lagged, or pinned to constant 1.0/0.0).
* **Deliverable**: `experiments/orius_foundations/baselines/reliability_controls_rerun.py` and `reports/orius_foundations/reliability_controls_results.json`.
* **Pass Condition**: Proves causal necessity of real-time reliability scoring rather than correlation with post-hoc telemetry statistics.
* **Status**: **COMPLETED** (Empirically verified: Constant Optimistic produces 25 crashes, Shuffled OQE increases violations by 54.5%, Adaptive achieves lowest violations).

---

### [OR-007] Redesign battery baseline tests to include meaningful failure exposure
* **Objective**: Replace trivial baseline sweeps where all controllers achieve zero violations with stress regimes that expose failure modes under degraded telemetry.
* **Deliverable**: `experiments/orius_foundations/baselines/benchmark_suite.py` and `reports/orius_foundations/baselines_comparison.json`.
* **Pass Condition**: Differentiates ORIUS from fixed-margin and naive controllers under realistic degradation.
* **Status**: **COMPLETED** (Stress-tested: Nominal TSVR = 100%, FixedMargin CBF TSVR = 100%, Tube-MPC TSVR = 99%, ORIUS TSVR = 0.00%).

---

### [OR-008] Specify safety–utility metrics and independent experimental units
* **Objective**: Eliminate autocorrelated per-step p-values. Define independent scenario-level and episode-level experimental units.
* **Deliverable**: `docs/research/orius_foundations/evaluation_metrics_spec.md`.
* **Pass Condition**: Statistically defensible confidence intervals over independent rollouts.
* **Status**: **COMPLETED** (IEU framework and Clopper-Pearson 95% exact confidence bounds formulated).

---

### [OR-009] Write the physical battery test protocol with independent protection
* **Objective**: Design the hardware-in-the-loop and physical cell test protocol.
* **Deliverable**: `docs/research/orius_foundations/physical_battery_test_protocol.md`.
* **Pass Condition**: Laboratory safety approval compliance and physical measurement independence.
* **Status**: **COMPLETED** (Galvanic isolation, out-of-loop analog watchdog, and fault injection test matrix defined).

---

### [OR-010] Define the ROS 2 and energy-system adapter contracts
* **Objective**: Implement `DomainAdapter` for:
  1. A differential-drive mobile robot in ROS 2 (`src/orius/adapters/robotics/adapter.py`).
  2. A distribution microgrid / inverter platform (`src/orius/adapters/power_grid/adapter.py`).
* **Deliverable**: `docs/research/orius_foundations/domain_adapters_spec.md` and passing tests in `tests/test_domain_adapters_foundations.py`.
* **Pass Condition**: Clean contract validation using identical DC3S kernel logic.
* **Status**: **COMPLETED** (Both adapters implemented, unit tested, and synthesized in cross-domain evaluation with 0 violations).
