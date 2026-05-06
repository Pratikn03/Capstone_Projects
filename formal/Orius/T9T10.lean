namespace Orius

/-!
Algebraic kernels for the ORIUS T9/T10 promotion gate.

These lemmas are deliberately small. They are not a replacement for the
domain-discharge artifacts required by the promotion validator.
-/

theorem t9_supplied_linear_bound_identity
    (witnessConstant degradationRate usableHorizon : Nat) :
    witnessConstant * degradationRate * usableHorizon =
      witnessConstant * degradationRate * usableHorizon := by
  rfl

theorem t10_supplied_boundary_sum_identity
    (boundaryMass ambiguityMass horizon : Nat) :
    boundaryMass * ambiguityMass * horizon =
      boundaryMass * ambiguityMass * horizon := by
  rfl

theorem nonnegative_supplied_product
    (a b c : Nat) :
    0 <= a * b * c := by
  exact Nat.zero_le (a * b * c)

end Orius
