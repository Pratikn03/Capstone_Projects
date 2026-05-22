# Active Theorem Remediation Plan

This file is generated from `reports/publication/theorem_registry.yml` and converts the reconciled theorem registry into concrete follow-up work.

## Immediate registry rules

- Treat `reports/publication/theorem_registry.yml` as the only hand-edited theorem inventory.
- Treat `reports/publication/theorem_surface_register.csv` as generated inventory only.
- Keep theorem-facing assumptions synchronized with Appendix B and fail validation on unknown IDs.

## Theorem-by-theorem actions

- T1 (paper_rigorous, high): add assumption - Keep A11 explicit and battery-scoped unless a stronger dispatch-model derivation is written.
- T2 (proof_runtime_linked, critical): strengthen proof - Keep the absorbed margin explicit in theorem text, repair metadata, and true-state invariance checks.
- T3a (paper_rigorous, critical): strengthen proof - Keep the reliability-score disclaimer explicit and avoid reintroducing weighted-exchangeability overclaims.
- T3b (paper_rigorous, medium): maintain split - Keep T3b corollary-only and avoid collapsing it back into a single overloaded T3 row.
- T4 (paper_rigorous, high): keep scope explicit - Preserve the fixed-margin mandatory-release definition and the admissible degraded-gap hypothesis; avoid broadening the theorem into a claim that every ambiguity class or every controller is unsafe.
- T5 (paper_rigorous, medium): keep finite-horizon scope - Preserve the reachable-tube containment hypothesis and do not convert T5 into an unconditional future-step probability law.
- T6 (machine_checked_ready, high): keep flagship closed-form - Keep the no-delta legacy semantics out of the theorem-facing API and preserve the explicit first-passage side conditions.
- T7 (proof_runtime_linked, medium): keep battery-specific - Preserve the piecewise hold-or-safe-landing scope and do not generalize T7 into a cross-domain transfer theorem.
- T8 (paper_rigorous, medium): keep useful-work gate - Preserve the useful-work threshold so immediate shutdown cannot satisfy T8 merely by being safe.
- T9 (paper_rigorous, high): keep mandatory-release scope - Keep T9 tied to empty-safe-core witnesses and do not restate it as a universal impossibility for policies allowed to abstain or fail closed; domain discharge is evidence, not an extra hidden assumption.
- T10 (paper_rigorous, high): keep two-state scope - Do not describe T10 as a full minimax frontier; the scoped minimax row is T_minimax and still lacks a global optimality claim; domain discharge is evidence, not an extra hidden assumption.
- T11 (paper_rigorous, critical): keep forward-only - Do not blur the supporting mini-harness into the active T11 transfer theorem.
- T10_T11_ObservationAmbiguitySandwich (proof_runtime_linked, high): keep scoped lower-upper comparison - Keep the corollary phrased as a covered lower/upper safety sandwich, not as a global optimality theorem.
- T11_AV_BrakeHold (proof_runtime_linked, high): fail closed - Missing T11 status, failed obligations, invalid certificate, or false postcondition must make the witness fail.
- T11_HC_FailSafeRelease (proof_runtime_linked, high): fail closed - Missing T11 status, failed obligations, invalid certificate, or false postcondition must make the witness fail.
- T6_AV_FallbackValidity (proof_runtime_linked, high): fail closed - Non-fail-safe fallback or failed postcondition must invalidate the witness.
- T6_HC_FallbackValidity (proof_runtime_linked, high): fail closed - Non-fail-safe fallback or failed postcondition must invalidate the witness.
- T_EQ_Battery_RuntimeArtifactPackage (artifact_runtime_linked, high): fail closed - Missing runtime-native comparator, ablation, negative-control, utility, or reproducibility evidence fails the equal-domain gate.
- T_EQ_AV_RuntimeArtifactPackage (artifact_runtime_linked, high): fail closed - Missing runtime-native comparator, ablation, negative-control, utility, or reproducibility evidence fails the equal-domain gate.
- T_EQ_HC_RuntimeArtifactPackage (artifact_runtime_linked, high): fail closed - Missing runtime-native comparator, ablation, negative-control, utility, or reproducibility evidence fails the equal-domain gate.
- L1 (paper_rigorous, medium): keep lemma scope - Keep L1 as a monotonicity lemma and do not revive the old rate-distortion converse wording.
- L2 (paper_rigorous, medium): keep lemma scope - Keep L2 as set antitonicity and do not restore the old capacity-proxy bridge as a defended theorem.
- L3 (paper_rigorous, medium): keep lemma scope - Keep L3 as the runtime intervention threshold and do not revive the old critical-capacity theorem wording.
- L4 (paper_rigorous, medium): keep lemma scope - Keep L4 as a runtime-law sandwich and preserve the lower-class and coverage-miss claim boundaries.
- T11_Byzantine (paper_rigorous, medium): keep Byzantine budget scope - Preserve the b<n/2 condition and honest-interval hypothesis in all manuscript-facing statements.
- T_stale_decay (paper_rigorous, medium): keep bounded-drift scope - Preserve the bounded-drift stale-hold radius statement and avoid deriving physical sensing laws from design schedules.
- T_minimax (paper_rigorous, medium): keep scoped minimax wording - Do not use the word optimal unless a matching upper bound is added for the same policy and distribution class.
- T_sensor_converse (paper_rigorous, medium): keep adapter-semantics scope - Preserve the missing-coordinate and disjoint-safe-core hypotheses; do not claim every low-quality sensor is universally necessary.
- T_trajectory_PAC (paper_rigorous, high): keep narrowed - Any future martingale strengthening must appear as a new theorem, not as a silent replacement.
