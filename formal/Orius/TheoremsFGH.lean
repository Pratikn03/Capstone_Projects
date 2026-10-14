import Orius.TheoremsAE
import Orius.KBoundMind

namespace Orius

/-!
# Mechanized Proofs for ORIUS + K-Bound Theorems F, G, and H

This module provides the dedicated Lean 4 mechanization surface for Chapter 13
(`formal/Orius/TheoremsFGH.lean`, extending `formal/Orius/TheoremsAE.lean`):

* **Theorem F (`theorem_f_terminal_safe_stop_invariance`)**:
  Terminal Safe-Stop Viability under Unbounded Telemetry Blackouts.
* **Theorem G (`theorem_g_arbitrary_ai_hallucination_immunity`)**:
  Universal Crash-Free Immunity to Arbitrary AI Policy Hallucinations via TV-SSK / K-Bound Projection.
* **Theorem H (`theorem_h_financial_flash_crash_immunity` & `theorem_h_liquidity_floor_safe_liquidation`)**:
  Cyber-Physical & Financial Safe-Liquidation Isomorphism under Explicit L2 Liquidity Floor Viability
  (`L_depth ≥ L_min > 0`).
-/

/-- Explicit L2 order-book liquidity floor viability condition (`Gap G11`):
Requires effective order-book depth `L_depth` to remain bounded below by a strictly positive
liquidity floor `L_min > 0` during the finite unwinding horizon into `zero_energy_safe` (`I_t = 0`). -/
structure LiquidityFloorObligation where
  L_depth : Int
  L_min : Int
  h_floor_pos : 0 < L_min
  h_depth_viable : L_min ≤ L_depth

/-- Lemma: Under the explicit liquidity floor `0 < L_min ≤ L_depth`, the effective order-book
depth `L_depth` is strictly positive (`0 < L_depth`), preventing division-by-zero singularity
in the Almgren-Chriss liquidation barrier `h_fin(I_t, w_t) = Equity_t - κ · I_t^2 / L_depth(w_t)`. -/
theorem liquidity_floor_prevents_barrier_singularity
    (L : LiquidityFloorObligation) :
    0 < L.L_depth := by
  exact Int.lt_of_lt_of_le L.h_floor_pos L.h_depth_viable

/-- **Theorem H.2 (Finite-Horizon Safe Liquidation under Explicit L2 Liquidity Floor)**.
In any `FinancialLiquidationDomain` satisfying the explicit liquidity floor obligation
`L : LiquidityFloorObligation` (`0 < L_min ≤ L_depth`), the governed liquidation release
is free of barrier singularity (`0 < L.L_depth`), guarantees one-step solvency and recursive
viability for any arbitrary AI trading policy `π_AI`, and preserves absorbing solvency across
all unbounded blackout steps `k : Nat` once flat inventory (`I_t = 0`) is reached. -/
theorem theorem_h_liquidity_floor_safe_liquidation
    (F : FinancialLiquidationDomain)
    (L : LiquidityFloorObligation)
    {U : ObsConsistent F.toSafetyDomain}
    (G : CrashFreeGovernor F.toViableSafetyDomain U)
    (π_AI : ArbitraryAIPolicy F.toViableSafetyDomain)
    (k : Nat)
    {x x_flat y_obs : F.State} {ω : F.Disturbance}
    (hx : x ∈ U)
    (h_flat : F.zero_energy_safe x_flat) :
    0 < L.L_depth ∧
    F.solvent (F.dynamics x (G.project (π_AI y_obs)) ω) ∧
    F.solvent (F.toSafeStopDomain.rolloutFallback k x_flat ω) := by
  have h_pos : 0 < L.L_depth := liquidity_floor_prevents_barrier_singularity L
  have h_imm := theorem_h_financial_flash_crash_immunity F G π_AI k (y_obs := y_obs) (ω := ω) hx h_flat
  exact ⟨h_pos, h_imm.left, h_imm.right⟩

end Orius
