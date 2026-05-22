namespace Orius

/-!
# Mechanized core for ORIUS T3, T8, T9

Second pass of real Lean 4 mechanizations to follow `Orius.T2T4T11`.  Each
theorem below has a real proof body -- no `sorry`, no `axiom` -- and the
module remains Mathlib-free so `lake build` stays a zero-dependency, no-network
verification step.

What is mechanized vs. what is gated

* **T3 (ORIUS Core Risk Envelope, arithmetic core)**: the discrete-sum
  identity that powers the bound `E[V_T] ≤ α(1−w̄)T` -- namely, that the sum
  of per-step risk budgets `α(1−w_t)` factors as `α · (T − Σ w_t)`.  The
  measure-theoretic step `P[Z_t = 1 | H_t] ≤ α(1−w_t)` itself is gated to
  the calibration discharge (A5) and a Mathlib port.  Reviewers can read
  this as: "given the per-step risk budgets, the *bookkeeping* of the bound
  is mechanically checked."
* **T8 (Graceful Degradation Dominance)**: monotonicity of the runtime
  inflation `α(w_t, s) = 1 + κ_r(1−w_t) + κ_s·s` in `(1−w_t)`.  Provable
  in pure `Nat` arithmetic with no probability or analysis.
* **T9 (Mandatory-Release Impossibility, empty-safe-core form)**: the clean
  structural impossibility we promoted T9 to in the paper -- if the common
  safe core for an observation is empty, every observation-only
  mandatory-release controller has a witness state for which the released
  action is unsafe.  One-line constructive proof.
-/

/-!
## T3 - arithmetic core of the risk envelope

The paper's T3 states `E[V_T] ≤ α(1 − w̄)T`.  Unfolding `w̄ = T⁻¹ ∑ w_t`,
this is equivalent to `E[V_T] ≤ α · (T − ∑ w_t)`.  Without measure theory we
mechanize the bookkeeping: the sum of per-step budgets `α(1 − w_t)` equals
`α · (T − ∑ w_t)`.  We model `(1 − w_t)` as a `Nat` "deficit" `d_t` and the
budget multiplier `α` as a `Nat` scale factor.  In a Mathlib port these
become `ENNReal` and the identity below lifts to the integrated bound.
-/

/-- Per-step risk budget `α · d_t` for a list of integer deficits. -/
def riskBudgetList (alpha : Nat) (deficits : List Nat) : List Nat :=
  deficits.map (alpha * ·)

/-- Total risk budget over the episode: sum of per-step `α · d_t`. -/
def riskEnvelope (alpha : Nat) (deficits : List Nat) : Nat :=
  (riskBudgetList alpha deficits).foldr (· + ·) 0

/-- Sum of deficits `∑ d_t`. -/
def deficitTotal (deficits : List Nat) : Nat :=
  deficits.foldr (· + ·) 0

/-- **T3 (Risk Envelope Arithmetic)**: the sum of per-step budgets
factors as `α · (∑ d_t)`.  This is the discrete bookkeeping that the
paper's `α(1 − w̄)T` collapses to once `w̄ = T⁻¹ Σ w_t` is unfolded.

Real proof by induction on the deficit list; no `sorry`. -/
theorem t3_risk_envelope_factors
    (alpha : Nat) (deficits : List Nat) :
    riskEnvelope alpha deficits = alpha * deficitTotal deficits := by
  induction deficits with
  | nil => rfl
  | cons d ds ih =>
    -- LHS: `(α·d) + foldr (+) 0 (map (α*·) ds)` = `α·d + riskEnvelope α ds`.
    -- RHS: `α · (d + foldr (+) 0 ds)` = `α·d + α · deficitTotal ds`.
    -- Use the induction hypothesis on `ds` and distributivity of `*` over `+`.
    show alpha * d + riskEnvelope alpha ds
       = alpha * (d + deficitTotal ds)
    rw [ih, Nat.mul_add]

/-- Corollary: monotonicity of the risk envelope in the deficits.
Increasing any per-step deficit cannot decrease the total budget. -/
theorem t3_risk_envelope_monotone
    (alpha : Nat) (d1 d2 : List Nat)
    (h : deficitTotal d1 ≤ deficitTotal d2) :
    riskEnvelope alpha d1 ≤ riskEnvelope alpha d2 := by
  rw [t3_risk_envelope_factors, t3_risk_envelope_factors]
  exact Nat.mul_le_mul_left alpha h

/-!
## T8 - Graceful Degradation Dominance: monotonicity of the runtime inflation

The runtime inflation formula `α(w_t, s) = 1 + κ_r(1 − w_t) + κ_s · s`
(Eq. 10 in the paper, before clipping) is monotonically *non-decreasing*
in the degradation level `(1 − w_t)` and in the shift score `s` -- which is
exactly the property the paper labels "graceful degradation dominance".

We model the formula in fixed-point `Nat` arithmetic: `unit = 100` so a
unit increase in `(1 − w_t)` corresponds to a unit increase in the deficit
counter.
-/

/-- Runtime inflation in fixed-point `Nat` units.  `baseline = 100` is the
nominal inflation of `1.0`; `kappa_r = 70` corresponds to the paper's
reliability penalty `0.7`; `kappa_s = 50` corresponds to the shift-score
penalty `0.5`. -/
def runtimeInflation (deficit shiftScore : Nat) : Nat :=
  100 + 70 * deficit + 50 * shiftScore

/-- **T8 (Graceful Degradation Dominance, deficit axis)**: the runtime
inflation is monotonically non-decreasing in `(1 − w_t)`.

Real proof using `Nat.mul_le_mul_left` plus left-additivity of `+`. -/
theorem t8_graceful_degradation_monotone_in_deficit
    (d1 d2 shiftScore : Nat) (h : d1 ≤ d2) :
    runtimeInflation d1 shiftScore ≤ runtimeInflation d2 shiftScore := by
  show 100 + 70 * d1 + 50 * shiftScore
     ≤ 100 + 70 * d2 + 50 * shiftScore
  exact Nat.add_le_add_right
    (Nat.add_le_add_left (Nat.mul_le_mul_left 70 h) 100) (50 * shiftScore)

/-- Monotonicity in the shift score (companion to the deficit-axis result;
both axes contribute additively, so the dominance is separable). -/
theorem t8_graceful_degradation_monotone_in_shift
    (deficit s1 s2 : Nat) (h : s1 ≤ s2) :
    runtimeInflation deficit s1 ≤ runtimeInflation deficit s2 := by
  show 100 + 70 * deficit + 50 * s1
     ≤ 100 + 70 * deficit + 50 * s2
  exact Nat.add_le_add_left (Nat.mul_le_mul_left 50 h) (100 + 70 * deficit)

/-- Joint monotonicity: degrading either axis (or both) only widens the
uncertainty set. -/
theorem t8_graceful_degradation_joint_monotone
    (d1 d2 s1 s2 : Nat) (hd : d1 ≤ d2) (hs : s1 ≤ s2) :
    runtimeInflation d1 s1 ≤ runtimeInflation d2 s2 := by
  exact Nat.le_trans
    (t8_graceful_degradation_monotone_in_deficit d1 d2 s1 hd)
    (t8_graceful_degradation_monotone_in_shift d2 s1 s2 hs)

/-- The runtime inflation is bounded below by the nominal `100` (i.e.\
$1.0\times$); this matches contract predicate **CP1** (Inflation factor
$\alpha \ge 1.0$). -/
theorem t8_inflation_at_least_unity (deficit shiftScore : Nat) :
    100 ≤ runtimeInflation deficit shiftScore := by
  show 100 ≤ 100 + 70 * deficit + 50 * shiftScore
  exact Nat.le_trans (Nat.le_add_right 100 (70 * deficit))
                     (Nat.le_add_right (100 + 70 * deficit) (50 * shiftScore))

/-!
## T9 - Empty-Safe-Core Impossibility (matches paper Proposition T9)

The paper's revised T9 (Proposition VI.x in `detailed_theory_bridge.tex`)
states: if the common safe core `K(o) := ⋂_{x ∈ B(o)} C(x)` is empty, no
observation-only mandatory-release controller can guarantee true-state
safety for all `x ∈ B(o)`.

Mechanized below as a one-line constructive proof.  The hypothesis
`h_empty` is precisely the empty-safe-core assumption expressed
constructively: for every candidate action `a`, there exists a state `x`
in the ambiguity class for which `a` is unsafe.
-/

/-- **T9 (Empty-Safe-Core Impossibility)**: for any observation-only
mandatory-release controller `π`, an unsafe-state witness exists in the
ambiguity class of the controller's chosen action.

The hypothesis `h_empty` is the empty-safe-core assumption.  The conclusion
is the constructive witness used by the paper's proof. -/
theorem t9_empty_safe_core_impossibility
    {Obs State Action : Type}
    (observe : State → Obs) (safeForAction : State → Action → Prop)
    (o : Obs)
    (h_empty : ∀ a : Action, ∃ x : State, observe x = o ∧ ¬ safeForAction x a)
    (π : Obs → Action) :
    ∃ x : State, observe x = o ∧ ¬ safeForAction x (π o) := by
  exact h_empty (π o)

/-- The contrapositive: a universally safe observation-only controller
implies that the common safe core is non-empty.  This is the form the
paper uses to argue that ORIUS's intervention surface is necessary. -/
theorem t9_universal_safety_requires_nonempty_core
    {Obs State Action : Type}
    (observe : State → Obs) (safeForAction : State → Action → Prop)
    (o : Obs)
    (π : Obs → Action)
    (h_all_safe : ∀ x : State, observe x = o → safeForAction x (π o)) :
    ¬ ∃ x : State, observe x = o ∧ ¬ safeForAction x (π o) := by
  intro ⟨x, hobs, hunsafe⟩
  exact hunsafe (h_all_safe x hobs)

/-!
## What this module adds vs. the existing stubs

The existing `Orius.T1T10` and `Orius.T9T10` modules carry placeholder
kernels that effectively assert tautologies of the form `(P → Q) → (P → Q)`.
Those are retained for backward compatibility but should be considered
*structural anchors* rather than mechanized proofs; the canonical
mechanizations for the corresponding theorem identifiers are now:

|  | Canonical Lean home | Proof character |
|--|---------------------|-----------------|
| T2 | `Orius.T2T4T11.t2_safety_preservation` | one-line, true by construction |
| T3 | `Orius.T3T8T9.t3_risk_envelope_factors` | arithmetic sum identity |
| T4 | `Orius.T2T4T11.t4_no_free_safety_witness` | constructive existential |
| T7 | `Orius.T1T10.t7_piecewise_fallback_kernel` | explicit case split (already real) |
| T8 | `Orius.T3T8T9.t8_graceful_degradation_*` | three real monotonicity proofs |
| T9 | `Orius.T3T8T9.t9_empty_safe_core_impossibility` | one-line constructive proof |
| T11 (det) | `Orius.T2T4T11.t11_typed_transfer_deterministic` | reduces to T2 |
| T11 (prob shape) | `Orius.T2T4T11.t11_probabilistic_lift_shape` | transitivity of `≤` |

Theorems still gated to a future Mathlib port: T5 (definition only,
trivial), T6 (sub-Gaussian first-passage), T10 (Le Cam two-point), and the
measure-theoretic completion of T3 and T11's probabilistic lift.
-/

end Orius
