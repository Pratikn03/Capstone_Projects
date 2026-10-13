# Formal Proof Obligations Matrix (Lean 4 & Mathematics)

**Target**: Genuine, Non-Tautological Machine-Checked Formalization  
**Files Audited**:  
- [`formal/Orius/T2T4T11.lean`](file:///Volumes/T9/gridpulse/formal/Orius/T2T4T11.lean)  
- [`formal/Orius/T3T8T9.lean`](file:///Volumes/T9/gridpulse/formal/Orius/T3T8T9.lean)  
- [`formal/Orius/Program.lean`](file:///Volumes/T9/gridpulse/formal/Orius/Program.lean)

---

## 1. The Definitional Tautology Finding

In the initial draft of `formal/Orius/T2T4T11.lean`, Theorem T2 was proved as:
```lean
def tightActionSet (D : SafetyDomain) (U : ObsConsistent D) : Set D.Action :=
  fun a => ∀ x : D.State, x ∈ U → ∀ ω : D.Disturbance, D.safe (D.dynamics x a ω)

theorem t2_safety_preservation ... (ha : a ∈ tightActionSet D U) :
  D.safe (D.dynamics x a ω) := by
  exact ha x hx ω
```

### Why this must be upgraded for 9.8:
Proving $D.\mathrm{safe}$ by applying the definition of `tightActionSet` is mathematically valid, but it **assumes what it must prove**. The genuine scientific and implementation obligations are:
1. **Computability & Non-Emptiness**: Proving that for given dynamics and state bounds, an actual action $a$ can be computed such that $a \in \mathrm{tightActionSet}(U)$.
2. **Refinement Bridge**: Proving that the runtime convex projection solver ($\arg\min \|a - a^{\mathrm{nom}}\|_2^2$) actually terminates with a point inside the safety polytope.
3. **Converse / Impossibility**: Showing that when $U$ exceeds the critical capacity $U \not\subseteq \mathcal{V}_d^\infty$, $\mathrm{tightActionSet}(U) = \emptyset$.

---

## 2. Itemized Proof Obligations

| Theorem | Role | Current Lean Status | Required 9.8 Upgrade Obligation |
| :--- | :--- | :--- | :--- |
| **Theorem A (Feasibility)** | Necessary & sufficient conditions for release | Partially defined in `Program.lean` | Mechanize dynamic viability invariance; prove non-emptiness of robust core for polyhedral sets. |
| **Theorem B (Bayes Lower Bound)** | Impossibility of zero-risk mandatory release | Constructive witness in `T4` | Formalize continuous state distributions and prove Bayes risk $\ge \mathbb{E}[\min_a \mathbb{P}(a \notin C(X) \mid O)]$. |
| **Theorem C (Achievability)** | Trajectory-level union bound | Gated via `axiom_coverage_bound` | Formalize union bound over non-stationary temporal martingales without external axioms. |
| **Theorem D (Pareto Tradeoff)** | Cost of conservatism | Not in Lean | Bound distance $\operatorname{dist}(a^*, \mathcal{A}_{\mathrm{robust}})$ under Lipschitz cost $L_J$. |
| **Theorem E (Transfer)** | Universal adapter invariant | Typeclass in `T11` | Mechanize the four typed obligations ($O_1, O_2, O_3, O_4$) as a formal Lean typeclass. |

---

## 3. Discharged vs. Open Obligations Ledger

* [x] **Discrete Two-State Ambiguity Core**: Discharged in `observation_ambiguity.py` and `formal/Orius/T1T10.lean`.
* [x] **Deterministic One-Step Implication**: Discharged in `formal/Orius/T2T4T11.lean`.
* [ ] **Substantive Safe-Set Membership Proof**: Open. Obligation to show convex projection onto affine linear constraints is verified.
* [ ] **Martingale / History-Dependent Conformal Coverage**: Open. Need finite-horizon trajectory union bound.
* [ ] **Physical Hardware Actuator Lag ($\delta_{\mathrm{act}}$)**: Open. Must be integrated into the state transition inequality.
