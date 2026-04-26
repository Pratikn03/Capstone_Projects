# Active Theorem Audit

This file is generated from `reports/publication/theorem_registry.yml` and is the reconciled active audit surface for the live theorem program.

## Summary

- Active theorem rows: 29
- Rigor counts: {'paper_rigorous': 6, 'proof_runtime_linked': 7, 'explicit_definition': 1, 'machine_checked_ready': 1, 'narrowed_supporting_surface': 3, 'scoped_extension': 6, 'artifact_runtime_linked': 3, 'stylized_surrogate': 1, 'proxy_bridge': 1}
- Code correspondence counts: {'partial': 10, 'matches': 19}
- Defense-tier counts: {'flagship_defended': 8, 'supporting_defended': 12, 'draft_non_defended': 9}
- Flagship gate ready: True
- Flagship defended IDs: ['T1', 'T2', 'T3a', 'T4', 'T6', 'T7', 'T11', 'T_trajectory_PAC']
- Supporting defended IDs: ['T3b', 'T8', 'T10_T11_ObservationAmbiguitySandwich', 'T11_AV_BrakeHold', 'T11_HC_FailSafeRelease', 'T6_AV_FallbackValidity', 'T6_HC_FallbackValidity', 'T_EQ_Battery_RuntimeArtifactPackage', 'T_EQ_AV_RuntimeArtifactPackage', 'T_EQ_HC_RuntimeArtifactPackage', 'T11_Byzantine', 'T_stale_decay']
- Draft / non-defended IDs: ['T5', 'T9', 'T10', 'L1', 'L2', 'L3', 'L4', 'T_minimax', 'T_sensor_converse']

## Namespace Drift

### src/orius/dc3s/coverage_theorem.py and tests/test_conditional_coverage.py

- Issue: Legacy auxiliary coverage helpers previously reused active T9/T10 numbering.
- Impact: The code namespace could silently contradict the manuscript theorem namespace.
- Status: resolved
- Remediation: Keep those helpers explicitly auxiliary and route theorem-facing claims through the YAML registry.

### src/orius/universal/contract.py, tests/test_universal_contract.py, and tests/test_unification.py

- Issue: The five-invariant reference harness was described too broadly relative to active T11.
- Impact: Supporting adapter checks could be mistaken for the full transfer theorem.
- Status: resolved
- Remediation: Treat the mini-harness as supporting evidence only; active T11 flows through the typed four-obligation surface.

### reports/publication/theorem_surface_register.csv

- Issue: The legacy register mixed raw inventory and defended-surface interpretation.
- Impact: Downstream tooling could select broader or legacy rows by accident.
- Status: resolved
- Remediation: Keep the register as YAML-generated inventory; use active_theorem_audit.* as the reconciled defended surface.

## Per-Theorem Audit

### T1: OASG Existence

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: phenomenon_existence
- Scope note: Defended as the battery witness-row OASG existence theorem under explicit arbitrage reachability and controller-fault independence assumptions.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:802
- Proof location: appendices/app_c_full_proofs.tex:36
- Assumptions used: ['A1', 'A2', 'A4', 'A11', 'A12']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ["{'Lemma': 'Observation gap under dropout'}", "{'Lemma': 'Boundary proximity under arbitrage'}"]
- Weakest step: Boundary reachability remains domain-specific and is discharged here by explicit arbitrage reachability rather than by a generic dispatch argument.
- Rigor rating: paper_rigorous
- Code correspondence: partial - Runtime and benchmark code expose observed-vs-true disagreement directly, but the arbitrage reachability step remains a theorem-level assumption rather than a runtime proof object.
- Severity if broken: high
- Remediation class: add assumption - Keep A11 explicit and battery-scoped unless a stronger dispatch-model derivation is written.
- Legacy aliases: []
- Code anchors:
  - src/orius/orius_bench/metrics_engine.py:91 (`compute_oasg`) - Counts observation-action safety gaps.
  - src/orius/cpsbench_iot/runner.py:1040 (`run_single`) - Exposes true and observed trajectories on the battery witness row.
- Test anchors:
  - tests/test_oasg_metrics.py:26 (`test_signature_equals_exposure_times_severity`) - Metric-level witness for OASG exposure.

### T2: Safety Preservation

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V3
- Program role: one_step_runtime_guarantee
- Scope note: Defended as a one-step true-state postcondition whose tightened margin already absorbs the one-step model-error allowance.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1041
- Proof location: appendices/app_c_full_proofs.tex:101
- Assumptions used: ['A1', 'A2', 'A3', 'A4', 'A5', 'A7']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ["{'Proposition': 'Inflated set contains the current state'}", "{'Proposition': 'Tightened feasibility implies true feasibility'}"]
- Weakest step: The absorbed tightening must match the runtime FTIT/shield implementation exactly; any separation between theory margin and repair margin breaks the V3 story.
- Rigor rating: proof_runtime_linked
- Code correspondence: matches - The battery tightening, repair, guarantee check, and runtime assertion surface all carry the absorbed margin explicitly.
- Severity if broken: critical
- Remediation class: strengthen proof - Keep the absorbed margin explicit in theorem text, repair metadata, and true-state invariance checks.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/safety_filter_theory.py:35 (`tightened_soc_bounds`) - Computes the absorbed tightening margin.
  - src/orius/dc3s/guarantee_checks.py:62 (`check_soc_invariance`) - Checks the true-state one-step postcondition with model-error slack.
  - src/orius/certos/runtime.py:180 (`CertOSRuntime.validate_and_step`) - Enforces the runtime theorem hook for one-step safety.
- Test anchors:
  - tests/test_dc3s_guarantee_checks.py:25 (`test_guarantee_checks_pass_for_safe_action`) - Safe-action witness.
  - tests/test_t2_absorbed_tightening.py:32 (`test_t2_absorbed_tightening_randomized`) - Randomized absorbed-margin regression.

### T3a: ORIUS Core Envelope Derivation

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: runtime_risk_budget_derivation
- Scope note: Defended as the per-step envelope derivation under the explicit battery assumptions and the narrowed reliability-score interpretation.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1276
- Proof location: appendices/app_c_full_proofs.tex:169
- Assumptions used: ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'A9']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ["{'Lemma': 'Aggregation under a predictable risk budget'}", 'Theorem T2 one-step safety preservation']
- Weakest step: The theorem is only as good as the runtime-to-risk bridge; the repo now keeps that bridge explicit and battery-scoped instead of hiding it inside generic conformal language.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The runtime now emits an explicit T3a theorem-contract summary that separates executable envelope checks from the declared battery calibration bridge.
- Severity if broken: critical
- Remediation class: strengthen proof - Keep the reliability-score disclaimer explicit and avoid reintroducing weighted-exchangeability overclaims.
- Legacy aliases: ['T3']
- Code anchors:
  - src/orius/universal_theory/kernel.py:211 (`execute_universal_step`) - Attaches theorem-contract summaries to the typed runtime result.
  - src/orius/universal_theory/risk_bounds.py:140 (`build_t3a_contract_summary`) - Canonical theorem-contract summary for the narrowed T3a surface.
  - src/orius/dc3s/coverage_theorem.py:44 (`compute_expected_violation_bound`) - Backward-compatible wrapper.
- Test anchors:
  - tests/test_dc3s_coverage_theorem.py:59 (`test_compute_expected_violation_bound`) - Envelope algebra regression.
  - tests/test_universal_theory.py:37 (`test_run_universal_step_returns_structured_result`) - Runtime theorem-contract regression.

### T3b: ORIUS Core Aggregation Corollary

- Surface kind: corollary
- Defense tier: supporting_defended
- Proof tier: V1
- Program role: runtime_risk_budget_aggregation
- Scope note: Derived corollary that converts the predictable per-step budget into the episode-average envelope.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1371
- Proof location: appendices/app_c_full_proofs.tex:169
- Assumptions used: ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'A9']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ['T3a', "{'Lemma': 'Aggregation under a predictable risk budget'}"]
- Weakest step: This row is bookkeeping rather than new theory; the actual burden sits in T3a.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The corollary is exactly the aggregation performed by the episode envelope helper.
- Severity if broken: medium
- Remediation class: maintain split - Keep T3b corollary-only and avoid collapsing it back into a single overloaded T3 row.
- Legacy aliases: ['T3']
- Code anchors:
  - src/orius/universal_theory/risk_bounds.py:109 (`compute_episode_risk_bound`) - Computes the corollary bound directly.
- Test anchors:
  - tests/test_active_theorem_audit.py:195 (`test_defense_tiers_match_the_rebuilt_core`) - Audit regression for the T3 split.

### T4: Observation Necessity / No Free Safety

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: bounded_necessity_witness
- Scope note: Defended as the observation-necessity witness for the fixed-margin quality-ignorant controller class on the battery row with explicit arbitrage reachability.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1609
- Proof location: appendices/app_c_full_proofs.tex:231
- Assumptions used: ['A1', 'A2', 'A4', 'A11']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ["{'Definition': 'Quality-ignorant controller'}", "{'Lemma': 'Admissible fault sequence existence'}", "{'Lemma': 'No margin compensation for quality-ignorant controllers'}"]
- Weakest step: The constructive witness is intentionally class-scoped; widening the controller class would require a new proof.
- Rigor rating: paper_rigorous
- Code correspondence: partial - The repo contains the necessary baseline and witness pieces, but the theorem remains constructive rather than fully executable.
- Severity if broken: high
- Remediation class: keep scope explicit - Preserve the fixed-margin quality-ignorant controller definition and avoid broadening the theorem into a claim that every ambiguity class is unsafe.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/supporting_results.py:187 (`verify_no_margin_compensation`) - Executable witness for fixed-margin insufficiency.
  - src/orius/cpsbench_iot/scenarios.py:171 (`generate_episode`) - Admissible degraded-observation episode generator.
- Test anchors:
  - tests/test_unification.py:65 (`test_unification_argument_w_t_never_drops_below_one`) - Supporting quality-ignorant limitation check.

### T5: Certificate Validity Horizon Definition

- Surface kind: definition
- Defense tier: draft_non_defended
- Proof tier: V1
- Program role: temporal_validity_definition
- Scope note: Kept as the constructive maximal forward-tube containment definition and excluded from defended theorem counts.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1811
- Proof location: appendices/app_c_full_proofs.tex:284
- Assumptions used: ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'A9']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ["{'Definition': 'Safety certificate'}", "{'Definition': 'Forward tube'}"]
- Weakest step: This row is intentionally definitional; future-step probability language is kept out of the defended theorem count.
- Rigor rating: explicit_definition
- Code correspondence: matches - The helper computes the maximal deterministic containment horizon directly.
- Severity if broken: medium
- Remediation class: keep retiered - Do not promote T5 back into defended headline theorem counts without a separate temporal concentration proof.
- Legacy aliases: ['T5']
- Code anchors:
  - src/orius/universal_theory/battery_instantiation.py:73 (`certificate_validity_horizon`) - Deterministic maximal-horizon helper.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:62 (`test_certificate_validity_horizon_matches_expiration_bound_for_zero_action`) - Deterministic containment regression.

### T6: Certificate Expiration Bound

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V2_linked
- Program role: temporal_expiration_bound
- Scope note: Defended as the confidence-aware closed-form battery expiration theorem with explicit delta dependence and first-passage side conditions.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1843
- Proof location: appendices/app_c_full_proofs.tex:304
- Assumptions used: ['A4', 'A7', 'A9']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ["{'Definition': 'Forward tube'}", 'Reflection-principle / sub-Gaussian first-passage bound']
- Weakest step: The defended row depends on keeping the delta-aware helper canonical and not silently widening the theorem surface back to legacy no-delta semantics.
- Rigor rating: machine_checked_ready
- Code correspondence: matches - The exported helper carries the closed-form first-passage formula, explicit side conditions, and fail-closed theorem metadata that forbids legacy no-delta semantics.
- Severity if broken: high
- Remediation class: keep flagship closed-form - Keep the no-delta legacy semantics out of the theorem-facing API and preserve the explicit first-passage side conditions.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/battery_instantiation.py:134 (`certificate_expiration_bound`) - Canonical theorem-facing T6 helper.
  - src/orius/dc3s/half_life.py:217 (`compute_conservative_horizon`) - Supplemental first-passage helper.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:35 (`test_certificate_expiration_bound_uses_delta_aware_formula`) - Direct theorem formula regression.
  - tests/test_t6_expiration_bound.py:69 (`test_t6_empirical_subgaussian_bound`) - Empirical confidence check.

### T7: Feasible Fallback Existence

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V2_linked
- Program role: piecewise_fallback_existence
- Scope note: Defended as a battery-specific piecewise fallback theorem: safe hold on the interior, safe landing near the boundary, and fail-closed infeasibility otherwise.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1937
- Proof location: appendices/app_c_full_proofs.tex:351
- Assumptions used: ['A1', 'A4', 'A8']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ['Constructive battery safe hold', 'Boundary-aware safe-landing recovery action']
- Weakest step: The theorem is battery-specific and only defended for the piecewise hold-or-safe-landing fallback surface implemented by the runtime.
- Rigor rating: proof_runtime_linked
- Code correspondence: matches - The validator now certifies the piecewise T7 surface directly: interior hold, boundary recovery by safe landing, and fail-closed infeasibility.
- Severity if broken: medium
- Remediation class: keep battery-specific - Preserve the piecewise hold-or-safe-landing scope and do not generalize T7 into a cross-domain transfer theorem.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/battery_instantiation.py:308 (`validate_battery_fallback`) - Piecewise theorem-facing T7 validator.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:66 (`test_certify_fallback_existence_passes_for_interior_soc`) - Positive interior witness.
  - tests/test_dc3s_temporal_theorems.py:79 (`test_certify_fallback_existence_recovers_near_lower_boundary`) - Lower-boundary recovery witness.
  - tests/test_dc3s_temporal_theorems.py:101 (`test_certify_fallback_existence_recovers_near_upper_boundary`) - Upper-boundary recovery witness.
  - tests/test_dc3s_temporal_theorems.py:123 (`test_certify_fallback_existence_fails_closed_when_boundary_recovery_is_infeasible`) - Fail-closed infeasibility witness.

### T8: Graceful Degradation Dominance

- Surface kind: theorem
- Defense tier: supporting_defended
- Proof tier: V1
- Program role: graceful_degradation_extension
- Scope note: Supporting comparison theorem at the sequence level; the strict absorbing-landing claim is not part of the defended headline core.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1967
- Proof location: appendices/app_c_full_proofs.tex:370
- Assumptions used: ['A1', 'A2', 'A3', 'A4', 'A5', 'A8']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ['Sequence comparison helper']
- Weakest step: The helper proves comparison for supplied paired sequences under a shared fault trace, not a full absorbing-landing dynamics theorem.
- Rigor rating: narrowed_supporting_surface
- Code correspondence: matches - The executable helper matches the narrowed statement exactly.
- Severity if broken: medium
- Remediation class: keep helper-scoped - Keep strict improvement phrasing tied to supplied paired sequences under the same admissible fault trace.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/battery_instantiation.py:416 (`evaluate_graceful_degradation_dominance`) - Sequence-level dominance checker.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:144 (`test_graceful_degradation_dominance_detects_strict_domination`) - Comparison helper regression.

### T9: Universal Impossibility, T9

- Surface kind: theorem
- Defense tier: draft_non_defended
- Proof tier: V0
- Program role: impossibility_extension
- Scope note: Visible only as a scoped impossibility extension under the explicit geometric-mixing assumption A10b.
- Statement location: chapters/ch37_universality_completeness.tex:112
- Proof location: appendices/app_c_full_proofs.tex:412
- Assumptions used: ['A10b', 'A11']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ['T4', 'Azuma-Hoeffding concentration for separated windows']
- Weakest step: The witness constant is still domain-scoped rather than universally instantiated.
- Rigor rating: scoped_extension
- Code correspondence: partial - The helper exposes the algebra once the witness constant is provided.
- Severity if broken: high
- Remediation class: keep scoped - Keep T9 out of defended headline counts until the witness constant and mixing bridge are fully discharged.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/theoretical_guarantees.py:599 (`compute_universal_impossibility_bound`) - Scaling witness helper.
- Test anchors:
  - tests/test_theoretical_guarantees_hypothesis.py:23 (`test_t9_impossibility_bound_matches_linear_scaling`) - Formula-only regression.

### T10: Boundary-indistinguishability lower bound, T10

- Surface kind: theorem
- Defense tier: draft_non_defended
- Proof tier: V0
- Program role: information_lower_bound_extension
- Scope note: Visible only as a scoped lower-bound frontier under the explicit TV bridge A13 and boundary-mass assumptions.
- Statement location: chapters/ch37_universality_completeness.tex:209
- Proof location: appendices/app_c_full_proofs.tex:453
- Assumptions used: ['A13']
- Typed obligations: []
- Unresolved assumptions: ['Unsafe-side boundary-mass sequence p_t is supplied explicitly.']
- Dependencies: ['Le Cam two-point lemma']
- Weakest step: The theorem stays scoped because the unsafe-side error bridge is not promoted beyond the explicit boundary-mass model.
- Rigor rating: scoped_extension
- Code correspondence: partial - The helper implements the stylized lower-bound algebra without overstating it as a universal converse.
- Severity if broken: high
- Remediation class: keep scoped - Preserve the explicit A13 + p_t boundary assumptions in all theorem-facing surfaces.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/theoretical_guarantees.py:638 (`compute_stylized_frontier_lower_bound`) - Scoped lower-bound helper.
- Test anchors:
  - tests/test_theoretical_guarantees_hypothesis.py:50 (`test_t10_frontier_lower_bound_matches_sum_formula`) - Formula-only regression.

### T11: Typed structural transfer theorem, T11

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: typed_transfer_theorem
- Scope note: Defended as the forward four-obligation one-step transfer theorem; the converse remains a separate structural failure proposition.
- Statement location: chapters/ch37_universality_completeness.tex:341
- Proof location: appendices/app_c_full_proofs.tex:501
- Assumptions used: []
- Typed obligations: ['Coverage obligation for the observation-consistent state set.', 'Soundness of the tightened safe-action set.', 'Repair membership in the tightened safe-action set.', 'Fallback admissibility when the tightened set is empty.']
- Unresolved assumptions: []
- Dependencies: ["{'Definition': 'Universal adapter contract'}", "{'Proposition': 'Failure of any transfer obligation breaks the reference proof pattern'}"]
- Weakest step: The theorem is forward-only at the active surface; stronger episode claims require an explicit domain risk budget.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The authoritative typed contract now emits a four-obligation theorem-contract summary over runtime artifacts; the five-invariant mini-harness remains supporting-only.
- Severity if broken: critical
- Remediation class: keep forward-only - Do not blur the supporting mini-harness into the active T11 transfer theorem.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/contracts.py:561 (`ContractVerifier.validate_runtime_step`) - Authoritative typed theorem surface.
  - src/orius/universal_theory/contracts.py:752 (`ContractVerifier.build_transfer_theorem_summary`) - Four-obligation theorem-contract summary over runtime artifacts.
  - src/orius/dc3s/theoretical_guarantees.py:683 (`evaluate_structural_transfer`) - Four-obligation executable witness.
  - src/orius/universal/contract.py:281 (`ContractVerifier.check`) - Supporting five-invariant mini-harness only.
- Test anchors:
  - tests/test_theoretical_guarantees_hypothesis.py:76 (`test_t11_transfer_requires_all_four_obligations`) - Active theorem regression.
  - tests/test_universal_contract.py:75 (`test_passes_all_five_invariants`) - Supporting harness regression.

### T10_T11_ObservationAmbiguitySandwich: T10_T11_ObservationAmbiguitySandwich: Covered Observation-Ambiguity Optimality

- Surface kind: corollary
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: supporting_observation_ambiguity_optimality
- Scope note: Supporting optimality corollary under covered observation ambiguity; it does not assert unrestricted global optimality for all physical AI systems.
- Statement location: chapters/ch37_universality_completeness.tex:389
- Proof location: appendices/app_c_full_proofs.tex:536
- Assumptions used: []
- Typed obligations: ['Observation ambiguity classes are explicit.', 'The common safe core is computed as the intersection of state-conditioned safe action sets.', 'ORIUS releases only actions safe for every state in its covered uncertainty set.', 'Probabilistic coverage is reported as an alpha-bound rather than unconditional zero violation.']
- Unresolved assumptions: []
- Dependencies: ['T10', 'T11', 'Common safe-core witness']
- Weakest step: The lower bound is a Bayes ambiguity-risk bound, not the false claim that differing safe sets alone imply unavoidable violation.
- Rigor rating: proof_runtime_linked
- Code correspondence: matches - The executable witness computes common safe cores, observation-only Bayes lower bounds, and covered ORIUS release upper bounds with explicit alpha-bound semantics.
- Severity if broken: high
- Remediation class: keep scoped optimality - Keep the theorem phrased as safety-optimal under covered observation ambiguity, not as a global optimality theorem for every physical-AI system.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/observation_ambiguity.py:169 (`build_observation_ambiguity_contract_summary`) - Publication-facing executable witness for the lower/upper sandwich.
  - src/orius/universal_theory/observation_ambiguity.py:55 (`observation_only_bayes_lower_bound`) - Computes the observation-only Bayes lower bound.
  - src/orius/universal_theory/observation_ambiguity.py:133 (`verify_covered_orius_release`) - Certifies the covered ORIUS upper-bound side.
- Test anchors:
  - tests/test_observation_ambiguity_optimality.py:21 (`test_different_safe_sets_do_not_imply_unavoidable_violation`) - Guards against the false unsafe-ambiguity overclaim.
  - tests/test_observation_ambiguity_optimality.py:52 (`test_orius_covered_release_certifies_zero_violation`) - Covered-release upper-bound regression.
  - tests/test_observation_ambiguity_optimality.py:66 (`test_probabilistic_coverage_is_alpha_bounded_not_unconditional_zero`) - Alpha-bound semantics regression.

### T11_AV_BrakeHold: T11_AV_BrakeHold: AV Brake-Hold Runtime Lemma

- Surface kind: runtime_linked_lemma
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: bounded_domain_transfer_lemma
- Scope note: Supporting AV runtime lemma under forward-only T11; bounded to promoted ORIUS replay rows and the longitudinal brake-hold postcondition.
- Statement location: appendices/app_c_full_proofs.tex:570
- Proof location: appendices/app_c_full_proofs.tex:570
- Assumptions used: []
- Typed obligations: ['T11 coverage obligation is runtime-linked.', 'T11 sound safe-action set obligation is runtime-linked.', 'T11 repair-membership obligation is runtime-linked.', 'T11 fallback-admissibility obligation is runtime-linked.', 'True brake-hold runtime postcondition passes for the AV replay row.']
- Unresolved assumptions: []
- Dependencies: ['T11', 'AV promoted replay runtime trace', 'Domain runtime contract witness CSV']
- Weakest step: Row-local replay witness only; it does not prove complete autonomous-driving closure.
- Rigor rating: proof_runtime_linked
- Code correspondence: matches - Runtime traces emit T11 status, certificate validity, and the AV brake-hold postcondition into domain contract witnesses.
- Severity if broken: high
- Remediation class: fail closed - Missing T11 status, failed obligations, invalid certificate, or false postcondition must make the witness fail.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/domain_runtime_contracts.py:122 (`DomainRuntimeContractWitness`) - Bounded runtime witness type.
  - src/orius/av_waymo/runtime.py:1491 (`_run_episode`) - Emits AV T11 and brake-hold witness fields.
  - scripts/build_domain_runtime_contract_artifacts.py:154 (`build_domain_runtime_contract_artifacts`) - Builds publication-facing witness artifacts.
- Test anchors:
  - tests/test_domain_runtime_contract_witnesses.py:30 (`test_passing_av_row_links_t11_certificate_and_postcondition`) - Passing AV witness regression.
  - tests/test_domain_runtime_contract_witnesses.py:51 (`test_failing_av_postcondition_fails_closed_even_with_certificate`) - Failing AV witness regression.

### T11_HC_FailSafeRelease: T11_HC_FailSafeRelease: Healthcare Fail-Safe Release Runtime Lemma

- Surface kind: runtime_linked_lemma
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: bounded_domain_transfer_lemma
- Scope note: Supporting healthcare runtime lemma under forward-only T11; bounded to promoted MIMIC monitoring rows and the fail-safe alert-release postcondition.
- Statement location: appendices/app_c_full_proofs.tex:590
- Proof location: appendices/app_c_full_proofs.tex:590
- Assumptions used: []
- Typed obligations: ['T11 coverage obligation is runtime-linked.', 'T11 sound safe-action set obligation is runtime-linked.', 'T11 repair-membership obligation is runtime-linked.', 'T11 fallback-admissibility obligation is runtime-linked.', 'Healthcare true fail-safe alert-release runtime postcondition passes.']
- Unresolved assumptions: []
- Dependencies: ['T11', 'Healthcare promoted monitoring runtime trace', 'Domain runtime contract witness CSV']
- Weakest step: Row-local monitoring witness only; it does not prove regulated clinical deployment closure.
- Rigor rating: proof_runtime_linked
- Code correspondence: matches - Runtime traces emit T11 status, certificate validity, and the healthcare fail-safe release postcondition into domain contract witnesses.
- Severity if broken: high
- Remediation class: fail closed - Missing T11 status, failed obligations, invalid certificate, or false postcondition must make the witness fail.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/domain_runtime_contracts.py:122 (`DomainRuntimeContractWitness`) - Bounded runtime witness type.
  - scripts/build_healthcare_runtime_artifacts.py:584 (`_run_orius_episode`) - Emits healthcare T11 and fail-safe witness fields.
  - scripts/build_domain_runtime_contract_artifacts.py:154 (`build_domain_runtime_contract_artifacts`) - Builds publication-facing witness artifacts.
- Test anchors:
  - tests/test_domain_runtime_contract_witnesses.py:72 (`test_passing_healthcare_row_links_t11_certificate_and_postcondition`) - Passing healthcare witness regression.
  - tests/test_domain_runtime_contract_witnesses.py:92 (`test_missing_t11_status_fails_closed`) - Missing T11 status regression.

### T6_AV_FallbackValidity: T6_AV_FallbackValidity: AV One-Step Fallback Certificate Lemma

- Surface kind: runtime_linked_lemma
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: bounded_domain_certificate_validity
- Scope note: Supporting AV certificate-validity lemma under T6/T11; degraded full-brake fallback releases are one-step valid only.
- Statement location: appendices/app_c_full_proofs.tex:609
- Proof location: appendices/app_c_full_proofs.tex:609
- Assumptions used: []
- Typed obligations: ['T6 first-passage validity semantics for positive-margin hold certificates.', 'T11 runtime witness is linked.', 'Runtime witness: AV full-brake fallback action is emitted for degraded fallback rows.', 'Runtime witness: AV true brake-hold postcondition passes.']
- Unresolved assumptions: []
- Dependencies: ['T6', 'T11', 'T11_AV_BrakeHold']
- Weakest step: One-step fallback validity only; degraded observation never grants multi-step validity.
- Rigor rating: proof_runtime_linked
- Code correspondence: matches - Shared domain validity helper assigns H_t=1 only to AV fail-safe fallback certificates.
- Severity if broken: high
- Remediation class: fail closed - Non-fail-safe fallback or failed postcondition must invalidate the witness.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/domain_validity.py:120 (`domain_certificate_validity_semantics`) - Bounded fallback and T6-style hold validity helper.
  - src/orius/av_waymo/runtime.py:1117 (`WaymoAVDomainAdapter.emit_certificate`) - AV certificate emitter.
- Test anchors:
  - tests/test_domain_runtime_contract_witnesses.py:109 (`test_av_fallback_certificate_validity_is_one_step_only`) - AV fallback validity regression.

### T6_HC_FallbackValidity: T6_HC_FallbackValidity: Healthcare One-Step Fallback Certificate Lemma

- Surface kind: runtime_linked_lemma
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: bounded_domain_certificate_validity
- Scope note: Supporting healthcare certificate-validity lemma under T6/T11; degraded max-alert fallback releases are one-step valid only.
- Statement location: appendices/app_c_full_proofs.tex:626
- Proof location: appendices/app_c_full_proofs.tex:626
- Assumptions used: []
- Typed obligations: ['T6 first-passage validity semantics for positive-margin hold certificates.', 'T11 runtime witness is linked.', 'Healthcare max-alert fallback action is emitted for degraded fallback rows.', 'Healthcare true fail-safe alert-release postcondition passes.']
- Unresolved assumptions: []
- Dependencies: ['T6', 'T11', 'T11_HC_FailSafeRelease']
- Weakest step: One-step fallback validity only; degraded observation never grants multi-step validity.
- Rigor rating: proof_runtime_linked
- Code correspondence: matches - Shared domain validity helper assigns H_t=1 only to healthcare max-alert fallback certificates.
- Severity if broken: high
- Remediation class: fail closed - Non-fail-safe fallback or failed postcondition must invalidate the witness.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/domain_validity.py:120 (`domain_certificate_validity_semantics`) - Bounded fallback and T6-style hold validity helper.
  - src/orius/universal_framework/healthcare_adapter.py:372 (`HealthcareDomainAdapter.emit_certificate`) - Healthcare certificate emitter.
- Test anchors:
  - tests/test_domain_runtime_contract_witnesses.py:128 (`test_healthcare_fallback_certificate_validity_is_one_step_only`) - Healthcare fallback validity regression.

### T_EQ_Battery_RuntimeArtifactPackage: T_EQ_Battery_RuntimeArtifactPackage: Battery Equal Artifact Discipline Package

- Surface kind: artifact_discipline_gate
- Defense tier: supporting_defended
- Proof tier: V1_artifact_runtime_linked
- Program role: equal_domain_artifact_discipline
- Scope note: Supporting artifact-discipline row; it checks battery theorem, runtime, comparator, ablation, negative-control, utility, and reproducibility evidence without changing the flagship theorem tier.
- Statement location: appendices/app_c_full_proofs.tex:642
- Proof location: appendices/app_c_full_proofs.tex:642
- Assumptions used: []
- Typed obligations: ['Battery locked witness runtime trace exists.', 'Battery comparator, ablation, and negative-control rows are runtime-denominator rows.', 'Battery ORIUS useful work exceeds the degenerate safe-fallback comparator.']
- Unresolved assumptions: []
- Dependencies: ['Battery runtime trace', 'Equal domain artifact discipline gate']
- Weakest step: Artifact-discipline equivalence only; it does not add a new battery theorem.
- Rigor rating: artifact_runtime_linked
- Code correspondence: matches - Equal-domain builder reads battery runtime traces and locked witness surfaces into strict per-domain artifact gates.
- Severity if broken: high
- Remediation class: fail closed - Missing runtime-native comparator, ablation, negative-control, utility, or reproducibility evidence fails the equal-domain gate.
- Legacy aliases: []
- Code anchors:
  - scripts/build_equal_domain_artifact_discipline.py:728 (`build_equal_domain_artifact_discipline`) - Builds the equal artifact-discipline gate.
  - scripts/validate_equal_domain_artifact_discipline.py:80 (`main`) - Validates fail-closed equal-domain evidence.
- Test anchors:
  - tests/test_equal_domain_artifact_discipline.py:68 (`test_equal_domain_artifact_discipline_gates_pass_for_all_domains`) - Battery equal-discipline regression.

### T_EQ_AV_RuntimeArtifactPackage: T_EQ_AV_RuntimeArtifactPackage: AV Equal Artifact Discipline Package

- Surface kind: artifact_discipline_gate
- Defense tier: supporting_defended
- Proof tier: V1_artifact_runtime_linked
- Program role: equal_domain_artifact_discipline
- Scope note: Supporting artifact-discipline row for the bounded AV replay contract; it does not assert full autonomous-driving field closure.
- Statement location: appendices/app_c_full_proofs.tex:661
- Proof location: appendices/app_c_full_proofs.tex:661
- Assumptions used: []
- Typed obligations: ['Vehicle T11 and T6 runtime lemmas are linked.', 'Vehicle comparator, ablation, and negative-control rows are runtime-denominator rows.', 'Vehicle ORIUS useful work exceeds always-brake useful work.']
- Unresolved assumptions: []
- Dependencies: ['T11_AV_BrakeHold', 'T6_AV_FallbackValidity', 'Equal domain artifact discipline gate']
- Weakest step: Artifact-discipline equivalence only; it remains bounded to the promoted AV replay contract.
- Rigor rating: artifact_runtime_linked
- Code correspondence: matches - AV runtime generation emits runtime-native comparator artifacts consumed by the equal-domain gate.
- Severity if broken: high
- Remediation class: fail closed - Missing runtime-native comparator, ablation, negative-control, utility, or reproducibility evidence fails the equal-domain gate.
- Legacy aliases: []
- Code anchors:
  - src/orius/av_waymo/runtime.py:1814 (`run_runtime_dry_run`) - Emits AV runtime and comparator artifacts.
  - scripts/build_equal_domain_artifact_discipline.py:728 (`build_equal_domain_artifact_discipline`) - Builds the equal artifact-discipline gate.
- Test anchors:
  - tests/test_equal_domain_artifact_discipline.py:68 (`test_equal_domain_artifact_discipline_gates_pass_for_all_domains`) - AV equal-discipline regression.
  - tests/test_av_waymo_dry_run.py:130 (`test_waymo_dry_run_smoke`) - AV runtime comparator smoke regression.

### T_EQ_HC_RuntimeArtifactPackage: T_EQ_HC_RuntimeArtifactPackage: Healthcare Equal Artifact Discipline Package

- Surface kind: artifact_discipline_gate
- Defense tier: supporting_defended
- Proof tier: V1_artifact_runtime_linked
- Program role: equal_domain_artifact_discipline
- Scope note: Supporting artifact-discipline row for the bounded healthcare monitoring contract; it does not assert regulated clinical deployment readiness.
- Statement location: appendices/app_c_full_proofs.tex:679
- Proof location: appendices/app_c_full_proofs.tex:679
- Assumptions used: []
- Typed obligations: ['Healthcare T11 and T6 runtime lemmas are linked.', 'Healthcare comparator, ablation, and negative-control rows are runtime-denominator rows.', 'Healthcare ORIUS useful work exceeds always-alert useful work.']
- Unresolved assumptions: []
- Dependencies: ['T11_HC_FailSafeRelease', 'T6_HC_FallbackValidity', 'Equal domain artifact discipline gate']
- Weakest step: Artifact-discipline equivalence only; it remains bounded to the promoted healthcare monitoring contract.
- Rigor rating: artifact_runtime_linked
- Code correspondence: matches - Healthcare runtime generation emits runtime-native comparator artifacts consumed by the equal-domain gate.
- Severity if broken: high
- Remediation class: fail closed - Missing runtime-native comparator, ablation, negative-control, utility, or reproducibility evidence fails the equal-domain gate.
- Legacy aliases: []
- Code anchors:
  - scripts/build_healthcare_runtime_artifacts.py:846 (`build_healthcare_runtime_artifacts`) - Emits healthcare runtime and comparator artifacts.
  - scripts/build_equal_domain_artifact_discipline.py:728 (`build_equal_domain_artifact_discipline`) - Builds the equal artifact-discipline gate.
- Test anchors:
  - tests/test_equal_domain_artifact_discipline.py:68 (`test_equal_domain_artifact_discipline_gates_pass_for_all_domains`) - Healthcare equal-discipline regression.
  - tests/test_healthcare_runtime_artifacts.py:39 (`test_build_healthcare_runtime_artifacts_emits_domain_native_runtime_surfaces`) - Healthcare runtime comparator smoke regression.

### L1: L1: Rate-Distortion Safety Law

- Surface kind: scoped_proxy_law
- Defense tier: draft_non_defended
- Proof tier: V0
- Program role: stylized_law_extension
- Scope note: Scoped surrogate law only.
- Statement location: appendices/app_c_full_proofs.tex:703
- Proof location: appendices/app_c_full_proofs.tex:703
- Assumptions used: []
- Typed obligations: []
- Unresolved assumptions: ['Stylized binary-source surrogate lower envelope.']
- Dependencies: ['Scoped law-helper surrogate definition']
- Weakest step: Surrogate rather than converse.
- Rigor rating: stylized_surrogate
- Code correspondence: partial - Arithmetic helper only.
- Severity if broken: medium
- Remediation class: future work - Keep scoped.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/orius_law.py:26 (`rate_distortion_safety_law`) - Stylized helper.
- Test anchors:
  - tests/test_orius_law.py:23 (`test_positive_loss_below_capacity`) - Arithmetic regression.

### L2: L2: Capacity Bridge

- Surface kind: scoped_proxy_law
- Defense tier: draft_non_defended
- Proof tier: V0
- Program role: stylized_bridge_extension
- Scope note: Proxy bridge only.
- Statement location: appendices/app_c_full_proofs.tex:742
- Proof location: appendices/app_c_full_proofs.tex:742
- Assumptions used: []
- Typed obligations: []
- Unresolved assumptions: ['Domain-scoped proxy constant kappa_d is externally calibrated.']
- Dependencies: ['Scoped kappa_d calibration']
- Weakest step: Proxy bridge rather than derived theorem.
- Rigor rating: proxy_bridge
- Code correspondence: partial - Proxy helper only.
- Severity if broken: medium
- Remediation class: future work - Keep scoped.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/orius_law.py:94 (`capacity_bridge`) - Proxy bridge helper.
- Test anchors:
  - tests/test_orius_law.py:58 (`test_inconsistency_detected`) - Proxy consistency regression.

### L3: L3: Critical Capacity

- Surface kind: scoped_proxy_law
- Defense tier: draft_non_defended
- Proof tier: V0
- Program role: stylized_threshold_extension
- Scope note: Threshold calculator only.
- Statement location: appendices/app_c_full_proofs.tex:776
- Proof location: appendices/app_c_full_proofs.tex:776
- Assumptions used: []
- Typed obligations: []
- Unresolved assumptions: ['Stylized L2 capacity-proxy bridge.']
- Dependencies: ['L2', 'T3a']
- Weakest step: Derived threshold, not independent converse.
- Rigor rating: scoped_extension
- Code correspondence: partial - Threshold helper only.
- Severity if broken: medium
- Remediation class: future work - Keep scoped.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/orius_law.py:162 (`critical_capacity`) - Threshold helper.
- Test anchors:
  - tests/test_orius_law.py:69 (`test_critical_capacity_exists`) - Arithmetic regression.

### L4: L4: Achievability-Converse Sandwich

- Surface kind: scoped_proxy_law
- Defense tier: draft_non_defended
- Proof tier: V0
- Program role: stylized_sandwich_extension
- Scope note: Stylized sandwich only.
- Statement location: appendices/app_c_full_proofs.tex:815
- Proof location: appendices/app_c_full_proofs.tex:815
- Assumptions used: []
- Typed obligations: []
- Unresolved assumptions: ['Executable T3-style upper envelope.']
- Dependencies: ['T3a', 'L1', 'L2']
- Weakest step: Lower side remains proxy-based.
- Rigor rating: scoped_extension
- Code correspondence: partial - Stylized sandwich helper only.
- Severity if broken: medium
- Remediation class: future work - Keep scoped.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/orius_law.py:227 (`achievability_converse_sandwich`) - Stylized sandwich helper.
- Test anchors:
  - tests/test_orius_law.py:89 (`test_bounds_consistent`) - Arithmetic regression.

### T11_Byzantine: T11_Byzantine: Byzantine Trimmed-Mean Bound

- Surface kind: theorem
- Defense tier: supporting_defended
- Proof tier: V1
- Program role: robustness_extension
- Scope note: Supporting bounded Byzantine robustness theorem under the stated trimmed-mean assumptions.
- Statement location: appendices/app_c_full_proofs.tex:863
- Proof location: appendices/app_c_full_proofs.tex:863
- Assumptions used: ['A9']
- Typed obligations: ['Byzantine fraction f is strictly below 1/3.', 'Trim fraction beta is at least f.', 'Tail-contamination model places adversarial readings in the trimmed tails.']
- Unresolved assumptions: []
- Dependencies: ['Trimmed-mean robustness argument']
- Weakest step: Simplified bounded-fraction model.
- Rigor rating: narrowed_supporting_surface
- Code correspondence: matches - Helper matches appendix scope.
- Severity if broken: medium
- Remediation class: strengthen proof - Keep bounded-fraction scope explicit.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/theoretical_guarantees.py:1257 (`prove_byzantine_bound`) - Executable witness.
- Test anchors:
  - tests/test_theoretical_guarantees.py:210 (`test_bound_holds_for_small_f`) - Witness regression.

### T_stale_decay: T_stale_decay: Stale-Decay Schedule

- Surface kind: theorem
- Defense tier: supporting_defended
- Proof tier: V1
- Program role: policy_schedule_extension
- Scope note: Supporting design-schedule theorem for stale-data degradation handling.
- Statement location: appendices/app_c_full_proofs.tex:921
- Proof location: appendices/app_c_full_proofs.tex:921
- Assumptions used: ['A6']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ['Exponential decay schedule', 'T3a upper envelope']
- Weakest step: Design schedule rather than physical sensing law.
- Rigor rating: narrowed_supporting_surface
- Code correspondence: matches - Helper matches the stated schedule.
- Severity if broken: medium
- Remediation class: keep scoped - Preserve design-schedule wording.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/theoretical_guarantees.py:1374 (`stale_decay_bound`) - Decay witness.
- Test anchors:
  - tests/test_theoretical_guarantees.py:242 (`test_reaches_epsilon`) - Schedule regression.

### T_minimax: T_minimax: Minimax OASG Tradeoff

- Surface kind: scoped_extension
- Defense tier: draft_non_defended
- Proof tier: V0
- Program role: stylized_minimax_extension
- Scope note: Visible only as a stylized minimax extension.
- Statement location: appendices/app_c_full_proofs.tex:979
- Proof location: appendices/app_c_full_proofs.tex:979
- Assumptions used: []
- Typed obligations: []
- Unresolved assumptions: ['T3a/L1/L2 stylized lower-envelope bridge.']
- Dependencies: ['Stylized lower-envelope witness']
- Weakest step: Lower side still proxy-based.
- Rigor rating: scoped_extension
- Code correspondence: partial - Stylized helper only.
- Severity if broken: medium
- Remediation class: future work - Keep out of defended counts.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/theoretical_guarantees.py:753 (`compute_tight_impossibility_bound`) - Stylized lower-envelope witness.
- Test anchors:
  - tests/test_theoretical_guarantees.py:291 (`test_lower_bound_uses_alpha_and_w_bar`) - Formula regression.

### T_sensor_converse: T_sensor_converse: Sensor Quality Converse

- Surface kind: scoped_extension
- Defense tier: draft_non_defended
- Proof tier: V0
- Program role: stylized_sensor_threshold_extension
- Scope note: Visible only as a proxy threshold extension.
- Statement location: appendices/app_c_full_proofs.tex:1030
- Proof location: appendices/app_c_full_proofs.tex:1030
- Assumptions used: []
- Typed obligations: []
- Unresolved assumptions: ['T3a/L2/L3 proxy threshold bridge.']
- Dependencies: ['Stylized inverse threshold helper']
- Weakest step: Proxy threshold rather than defended converse.
- Rigor rating: scoped_extension
- Code correspondence: partial - Helper exposes the threshold honestly.
- Severity if broken: medium
- Remediation class: future work - Keep out of defended counts.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/theoretical_guarantees.py:857 (`sensor_quality_converse`) - Stylized threshold helper.
- Test anchors:
  - tests/test_theoretical_guarantees.py:321 (`test_converse_holds_when_w_sufficient`) - Threshold regression.

### T_trajectory_PAC: T_trajectory_PAC: PAC Trajectory Certificate

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: trajectory_certificate
- Scope note: Defended as the implemented Bonferroni/union-bound trajectory certificate and nothing stronger.
- Statement location: appendices/app_c_full_proofs.tex:1067
- Proof location: appendices/app_c_full_proofs.tex:1067
- Assumptions used: ['A1', 'A4', 'A5', 'A9']
- Typed obligations: ['Bonferroni/union-bound aggregation over the horizon.']
- Unresolved assumptions: []
- Dependencies: ['T3a', 'Finite-sample conformal correction']
- Weakest step: The certificate is conservative by design because it stacks finite-sample and worst-case reliability corrections.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The helper, tests, and appendix all use the same union-bound semantics.
- Severity if broken: high
- Remediation class: keep narrowed - Any future martingale strengthening must appear as a new theorem, not as a silent replacement.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/risk_bounds.py:551 (`pac_trajectory_safety_certificate`) - Canonical certificate helper.
- Test anchors:
  - tests/test_theoretical_guarantees.py:380 (`test_martingale_flag_set_correctly`) - Guards the narrowed semantics.
