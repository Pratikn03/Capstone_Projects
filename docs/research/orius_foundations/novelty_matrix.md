# ORIUS Novelty & Prior-Art Defense Matrix

**Target**: Sharp, Defensible Novelty against 2024–2026 Literature  
**Rule**: ORIUS must establish a result that cannot be reduced to any single prior framework.

---

## 1. Prior-Art Differentiation Matrix

| Framework Family | Representative Landmark / Prior Art | Key Mechanism | What ORIUS Proves & Demonstrates Beyond It |
| :--- | :--- | :--- | :--- |
| **Unified Safety Filters** | *Safety Filter: A Unified View* (Annual Reviews in Control, 2024) | Unifies CBF, predictive filters, and projection filters for known/estimated state. | Characterizes **information-dependent feasibility**: when observation ambiguity forces incompatible safe actions, proving Bayes lower bounds on mandatory release and the safety–utility cost of conservative release. |
| **Belief-Space Conformal Safety** | *Safety-critical Control Under Partial Observability* (arXiv, 2026) | Belief-space CBF with conformal bounds tested on space-robotics platform. | Decouples full belief-state estimation from runtime release; establishes **trajectory-level finite-horizon release certificates** ($\Pr(\exists t \le T: x_t \notin \mathcal{S}_d) \le \sum \alpha_t + \delta_{\mathrm{model}} + \delta_{\mathrm{act}}$) with sub-millisecond auditable cryptographic certificates. |
| **Control Barrier–Value Functions** | *Control Barrier–Value Functions under Partial Observability* (arXiv, 2026) | Finite-horizon probabilistic safety with online conformal estimation-error bounds. | Proves **information-acquisition necessity**: exact conditions where acquiring an additional measurement or delaying actuation alters feasibility, coupled with cross-domain typed transfer across distinct plant dynamics. |
| **Robust / Tube MPC** | Mayne et al. (Automatica); Rawlings et al. | Computes nominal trajectory inside an invariant rigid tube with fixed disturbance bounds. | Replaces static/fixed geometric tubes with **dynamically inflated observation-uncertainty sets** that widen under packet drop/jitter and contract under verified telemetry, avoiding permanent conservatism. |
| **Simplex / Runtime Assurance** | Sha (IEEE Computer, 1998; 2001); NASA RTA Framework (2024) | Switches to pre-certified fail-safe backup controller when an analytic safety monitor trips. | Introduces **continuous convex action repair and graceful degradation** before discrete fallback, maintaining partial utility and providing formal certificate TTL expiration laws. |
| **K-Bound** | Niroula et al. (2026) | Indistinguishability and abstention theory for unsupervised model adaptation. | Independent physical control and actuation theory: formulates physical constraint preservation, dynamic viability, and real-time plant actuation contracts rather than feature adaptation. |

---

## 2. The Flagship Novelty Target (Three-Sentence Defense)

1. **Fundamental Limit**: Under partially observed physical dynamics with observation degradation, there exists an intrinsic information-theoretic lower bound on achievable true-state safety risk for any mandatory-release controller operating on observation history $H_t$.
2. **Constructive Achievability**: The ORIUS typed runtime layer (`Detect -> Calibrate -> Constrain -> Shield -> Certify`) achieves an $\alpha$-bounded trajectory-level safety guarantee by dynamically coupling conformal uncertainty inflation to constrained safe-set geometry and feasible fallback.
3. **Empirical Value**: Across three distinct physical plant classes (battery energy storage, power grids, and embodied robotics), ORIUS demonstrates that certificate-backed action release strictly dominates uncalibrated safety filters on the safety–utility Pareto frontier under controlled telemetry failures.
