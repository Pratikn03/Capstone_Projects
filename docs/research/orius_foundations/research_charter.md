# ORIUS Research Charter: Target Quality 9.8 / 10

**Status**: Active Research Constitution  
**Target Venue**: Flagship Physical AI / Control / Robotics Venue (IEEE TAC / Automatica / RSS / NeurIPS)  
**Research Identity**: *ORIUS: Fundamental Limits and Certified Action Release Under Degraded Observation for Physical AI*

---

## 1. Central Claim

> **An autonomous physical system should release an action only when its available information, uncertainty model, and verified control authority support the required safety guarantee. ORIUS characterizes when such release is possible, when it is impossible, and how to enforce it efficiently across different physical systems.**

ORIUS is not merely an empirical safety filter or a heuristic anomaly wrapper. It is a foundational framework characterizing the **information-theoretic and control-theoretic limits of certified physical release under degraded observation**.

---

## 2. The Four Foundational Research Questions

1. **Knowability**: Can the system distinguish physical states that require incompatible safe actions?
   $$\mathcal{X}_{\mathrm{amb}}(H_t) = \{x \in \mathcal{X} : h_d(x, \nu) \text{ is consistent with } H_t\}$$
2. **Possibility**: Under what conditions on dynamics $f_d$, disturbance set $\Xi_d$, and observation history $H_t$ does any acceptably safe action exist?
   $$\mathcal{A}_{\mathrm{robust}}(U_t) = \left\{ a \in \mathcal{A}_d : \forall x \in U_t(H_t), \forall \xi \in \Xi_d, \, f_d(x, a, \xi) \in \mathcal{S}_d \right\} \neq \emptyset$$
3. **Achievability**: Can ORIUS compute, verify, and execute such an action in real-time with an auditable, causal certificate guarantee?
4. **Usefulness**: Can ORIUS guarantee safety without unnecessarily sacrificing task performance (utility, throughput, energy delivery)?

---

## 3. Scope and Decoupling Principles

* **Decoupling from K-Bound**: While K-Bound establishes limits of unsupervised adaptation, ORIUS independently establishes the limits of physical actuation under degraded telemetry. Conceptual synergy is preserved, but all physical theorems and runtime release mechanics are self-contained in ORIUS.
* **Separation of Evidence Tiers**:
  * **Proved**: Formally derived and mechanized in Lean 4 without circularity.
  * **Simulated / Replay**: nuPlan AV replay and software-in-the-loop HIL (explicitly labeled as simulated).
  * **Physically Measured**: Closed-loop trials on real physical hardware with independent measurement paths and safety interlocks.
* **Mandatory Disqualifying Gates**:
  1. No circular theorem proofs in Lean or LaTeX.
  2. No simulated trials labeled as physical hardware.
  3. No safety gains achieved purely via trivial shutdown without safety–utility Pareto analysis.
  4. Baselines must receive equal tuning, information, and computational budgets.
  5. Statistical confidence must be computed over independent episodes, not autocorrelated time steps.
