# Mathematical Foundations: Theorems A through E
**Target Quality: 9.8 / 10**  
*ORIUS: Fundamental Limits and Certified Action Release Under Degraded Observation for Physical AI*

---

## 1. Problem Formulation & Dynamical System Model

Let $d \in \mathcal{D}$ denote a physical domain (e.g., battery energy storage, power grid inverter, autonomous robot).
The physical plant evolves according to discrete-time partially observed nonlinear dynamics with disturbance:

$$x_{t+1} = f_d(x_t, a_t, \xi_t), \quad x_t \in \mathcal{X}_d \subseteq \mathbb{R}^{n_x}, \; a_t \in \mathcal{A}_d \subseteq \mathbb{R}^{n_a}, \; \xi_t \in \Xi_d \subseteq \mathbb{R}^{n_\xi}$$

The true state $x_t$ is hidden. The controller receives degraded telemetry:

$$o_t = h_d(x_t, \nu_t), \quad o_t \in \mathcal{O}_d \cup \{\bot\}, \; \nu_t \in \mathcal{N}_d$$

where $\bot$ represents packet dropout or complete communication blackout, and $\nu_t$ captures measurement noise, quantization, timestamp jitter, or sensor drift.

### Observation History and Uncertainty Sets
At time $t$, the controller has access only to the causal observation history:

$$H_t = \left( (o_0, a_0), (o_1, a_1), \dots, (o_{t-1}, a_{t-1}), o_t \right) \in \mathcal{H}_t$$

Let $U_t(H_t) \subseteq \mathcal{X}_d$ denote the **observation-consistent state uncertainty set** constructed from $H_t$ under a specified calibration confidence level $1 - \alpha_t$:

$$\mathbb{P}\left( x_t \in U_t(H_t) \mid H_t \right) \ge 1 - \alpha_t$$

Let $\mathcal{S}_d \subset \mathcal{X}_d$ denote the physical safety set (e.g., safe battery SoC/temperature limits, grid voltage/frequency bands, or collision-free robot coordinates).

### Robust Feasible Action Set
For any candidate uncertainty set $U \subseteq \mathcal{X}_d$, the robust one-step safe action set is:

$$\mathcal{A}_{\mathrm{robust}}(U) \triangleq \left\{ a \in \mathcal{A}_d : \forall x \in U, \; \forall \xi \in \Xi_d, \; f_d(x, a, \xi) \in \mathcal{S}_d \right\}$$

---

## 2. Theorem A: Information-Dependent Release Feasibility

### Statement
Given plant dynamics $f_d$, disturbance compact set $\Xi_d$, safety set $\mathcal{S}_d$, and observation history $H_t$:

1. **Release Feasibility Condition**: An admissible safe release action exists if and only if the common robust core over $U_t(H_t)$ is non-empty:
   $$\mathcal{A}_{\mathrm{robust}}(U_t(H_t)) = \bigcap_{x \in U_t(H_t)} \left\{ a \in \mathcal{A}_d : \forall \xi \in \Xi_d, \, f_d(x, a, \xi) \in \mathcal{S}_d \right\} \neq \emptyset$$

2. **Dynamic Viability (Infinite-Horizon Recursive Safety)**:
   Let $\mathcal{V}_d^\infty \subseteq \mathcal{S}_d$ be the maximal viable control invariant set under disturbance $\Xi_d$:
   $$\mathcal{V}_d^\infty = \left\{ x \in \mathcal{S}_d : \exists a \in \mathcal{A}_d \text{ s.t. } \forall \xi \in \Xi_d, \, f_d(x, a, \xi) \in \mathcal{V}_d^\infty \right\}$$
   A release action $a_t$ guarantees infinite-horizon safety preservation if and only if:
   $$a_t \in \mathcal{A}_{\mathrm{viable}}(U_t(H_t)) \triangleq \left\{ a \in \mathcal{A}_d : \forall x \in U_t(H_t), \; \forall \xi \in \Xi_d, \; f_d(x, a, \xi) \in \mathcal{V}_d^\infty \right\}$$

3. **Sensor Information Value (Value of Delay / Information Acquisition)**:
   Let $\Delta t$ denote an intentional actuation delay (holding current control or safe standby) to acquire an additional observation $o_{t+1}$. If $U_{t+1}(H_{t+1}) \subset U_t(H_t)$ almost surely and $\mathcal{A}_{\mathrm{viable}}(U_t) = \emptyset$, but $\mathbb{P}(\mathcal{A}_{\mathrm{viable}}(U_{t+1}) \neq \emptyset) > 0$, then holding action for $\Delta t$ strictly expands the viable action feasibility margin compared to mandatory immediate release.

---

## 3. Theorem B: Observation-Limited Safety Lower Bound (Bayes Impossibility)

### Statement
Let $\pi: \mathcal{H}_t \to \mathcal{A}_d$ be any **mandatory-release controller** that maps observation history $H_t$ directly to an action without an abstention or safe-fallback authority.

Let the True-State Violation Rate ($\text{TSVR}$) over horizon $T$ be defined as:
$$\text{TSVR}_T(\pi) \triangleq \mathbb{E}\left[ \frac{1}{T} \sum_{t=0}^{T-1} \mathbf{1}\left( f_d(x_t, \pi(H_t), \xi_t) \notin \mathcal{S}_d \right) \right]$$

Then, for any such policy $\pi$, the safety risk is strictly bounded below by the Bayes observation ambiguity risk:

$$\text{TSVR}_T(\pi) \ge \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}_{H_t} \left[ \min_{a \in \mathcal{A}_d} \mathbb{P}\left( \exists \xi \in \Xi_d : f_d(X_t, a, \xi) \notin \mathcal{S}_d \;\middle|\; H_t \right) \right]$$

### Tightness and Significance
* If the observation map $h_d$ admits an ambiguity class $B(o) = \{ x : h_d(x) = o \}$ such that $\bigcap_{x \in B(o)} \mathcal{A}_{\mathrm{robust}}(\{x\}) = \emptyset$, then **no mandatory-release controller, regardless of neural network architecture, parameter scale, or nominal optimality, can achieve zero true-state violation**.
* This proves **Observation Necessity ("No Free Safety")**: better prediction on observed data cannot bypass the fundamental geometric conflict of state ambiguity.

---

## 4. Theorem C: Certified ORIUS Achievability Bound

### Statement
Let ORIUS execute the five-stage DC3S kernel at each time step $t \in \{0, \dots, T-1\}$:
1. Construct uncertainty set $U_t(H_t)$ with marginal miscoverage probability $\alpha_t$:
   $$\mathbb{P}(x_t \notin U_t(H_t) \mid H_t) \le \alpha_t$$
2. If $\mathcal{A}_{\mathrm{robust}}(U_t) \neq \emptyset$, release repaired action $\tilde{a}_t = \arg\min_{a \in \mathcal{A}_{\mathrm{robust}}(U_t)} \| a - a_t^{\mathrm{nom}} \|_2^2$.
3. If $\mathcal{A}_{\mathrm{robust}}(U_t) = \emptyset$, release verified fallback action $a_t^{\mathrm{fallback}} \in \mathcal{A}_{\mathrm{fallback}}(U_t)$.

Assume:
* **Model Discrepancy Bound**: The nominal dynamics model $\hat{f}_d$ satisfies $\| f_d(x, a, \xi) - \hat{f}_d(x, a) \| \le \epsilon_m$ with probability at least $1 - \delta_{\mathrm{model}}$.
* **Actuator Execution Guarantee**: The physical actuator executes $\tilde{a}_t$ within tolerance $\| a_{\mathrm{actual}} - \tilde{a}_t \| \le \epsilon_a$ with probability at least $1 - \delta_{\mathrm{act}}$.
* **Fallback Viability**: The fallback action satisfies $\hat{f}_d(x, a^{\mathrm{fallback}}) \in \mathcal{S}_d \ominus (\epsilon_m + L_f \epsilon_a)$.

Then the cumulative trajectory-level safety violation probability over horizon $T$ satisfies:

$$\mathbb{P}\left( \exists t \in \{0, \dots, T-1\} : x_{t+1} \notin \mathcal{S}_d \right) \le \sum_{t=0}^{T-1} \alpha_t + \delta_{\mathrm{model}} + \delta_{\mathrm{act}}$$

Under exact conformal calibration ($\alpha_t = \alpha / T$) and verified actuator execution ($\delta_{\mathrm{model}} = \delta_{\mathrm{act}} = 0$):
$$\mathbb{P}\left( \text{Trajectory Violation} \right) \le \alpha$$

---

## 5. Theorem D: Safety–Utility Characterization & Pareto Frontier

### Statement
Let $J_d(x_t, a_t)$ be a continuous, $L_J$-Lipschitz utility/cost objective (e.g., power tracking error, transport speed, energy cost).
Let $a_t^* = \arg\min_{a \in \mathcal{A}_d} J_d(x_t, a)$ be the unconstrained optimal action under perfect state information.
Let $\tilde{a}_t$ be the action released by ORIUS under uncertainty set $U_t(H_t)$.

1. **Cost of Epistemic Conservatism**:
   The instantaneous utility loss $\Delta J_t = J_d(x_t, \tilde{a}_t) - J_d(x_t, a_t^*)$ is bounded by:
   $$\Delta J_t \le L_J \cdot \operatorname{dist}\left( a_t^*, \mathcal{A}_{\mathrm{robust}}(U_t(H_t)) \right) \le L_J \cdot \kappa_d \cdot \operatorname{diam}\left( U_t(H_t) \right)$$
   where $\kappa_d$ is the plant's constraint conditioning constant and $\operatorname{diam}(U_t) = \sup_{x, x' \in U_t} \|x - x'\|$.

2. **Degradation-Sensitive Pareto Dominance**:
   As observation reliability $w_t \to 1$ (high telemetry quality), $\operatorname{diam}(U_t) \to 0$ and $\Delta J_t \to 0$ (near-optimal utility).  
   As $w_t \to 0$ (telemetry blackout), ORIUS smoothly contracts $\mathcal{A}_{\mathrm{robust}}$ toward fallback $a^{\mathrm{fallback}}$, establishing an upper envelope on utility regret while strictly holding $\text{TSVR} \le \alpha$.

---

## 6. Theorem E: Cross-Domain Structural Invariance (Universal Transfer)

### Statement
The ORIUS safety certificate and trajectory bounds transfer without modification across distinct physical domains $d_1, d_2 \in \mathcal{D}$ if and only if each domain's typed adapter satisfies four invariant proof obligations:

1. **Soundness of Uncertainty Set**: $U_t(H_t)$ contains the true latent state $x_t$ with probability $\ge 1 - \alpha_t$.
2. **Robust Safe Core Feasibility**: The adapter defines a computable projection or linear program onto $\mathcal{A}_{\mathrm{robust}}(U_t)$.
3. **Admissible Fail-Closed Fallback**: In the event that $\mathcal{A}_{\mathrm{robust}}(U_t) = \emptyset$, the adapter supplies a verified invariant fallback $a^{\mathrm{fallback}} \in \mathcal{A}_d$ that is guaranteed safe for all remaining states.
4. **Causal Certificate Integrity**: Every emitted release certificate $C_t = \langle t, w_t, U_t, \tilde{a}_t, \text{hash}_{t-1} \rangle$ is non-anticipative and causally dependent strictly on $H_t$.
