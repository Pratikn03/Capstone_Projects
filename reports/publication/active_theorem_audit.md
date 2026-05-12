# Active Theorem Audit

This file is generated from `reports/publication/theorem_registry.yml` as a historical traceability surface.
The current proof-strength and promotion authority is `reports/publication/theorem_promotion_matrix.json` plus `reports/publication/theorem_result_cards/*.json`, validated by `scripts/validate_theorem_promotion.py`.

## Summary

- Authority role: historical_traceability
- Current promotion authority: {'matrix': 'reports/publication/theorem_promotion_matrix.json', 'result_cards': 'reports/publication/theorem_result_cards/', 'validator': 'scripts/validate_theorem_promotion.py'}
- Active theorem rows: 29
- Rigor counts: {'paper_rigorous': 18, 'proof_runtime_linked': 7, 'machine_checked_ready': 1, 'artifact_runtime_linked': 3}
- Code correspondence counts: {'matches': 29}
- Defense-tier counts: {'flagship_defended': 16, 'supporting_defended': 13, 'draft_non_defended': 0}
- Flagship gate ready: True
- Flagship defended IDs: ['T1', 'T2', 'T3a', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T11_Byzantine', 'T_stale_decay', 'T_minimax', 'T_sensor_converse', 'T_trajectory_PAC']
- Supporting defended IDs: ['T3b', 'T10_T11_ObservationAmbiguitySandwich', 'T11_AV_BrakeHold', 'T11_HC_FailSafeRelease', 'T6_AV_FallbackValidity', 'T6_HC_FallbackValidity', 'T_EQ_Battery_RuntimeArtifactPackage', 'T_EQ_AV_RuntimeArtifactPackage', 'T_EQ_HC_RuntimeArtifactPackage', 'L1', 'L2', 'L3', 'L4']
- Draft / non-defended IDs: []

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
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:837
- Proof location: appendices/app_c_full_proofs.tex:46
- Assumptions used: ['A1', 'A2', 'A4', 'A11', 'A12']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ["{'Lemma': 'Observation gap under dropout'}", "{'Lemma': 'Boundary proximity under arbitrage'}"]
- Weakest step: Boundary reachability remains domain-specific; the active code witness now encodes the battery-row observed-safe/true-unsafe margin construction rather than treating it as an unanchored prose step.
- Rigor rating: paper_rigorous
- Code correspondence: matches - Runtime metrics count OASG events and the supporting-results witness constructs the observed-safe/true-unsafe degraded-observation margin used by the proof.
- Severity if broken: high
- Remediation class: add assumption - Keep A11 explicit and battery-scoped unless a stronger dispatch-model derivation is written.
- Legacy aliases: []
- Code anchors:
  - src/orius/orius_bench/metrics_engine.py:93 (`compute_oasg`) - Counts observation-action safety gaps.
  - src/orius/dc3s/supporting_results.py:209 (`verify_oasg_existence_witness`) - Executable observed-safe/true-unsafe margin witness for T1.
  - src/orius/cpsbench_iot/runner.py:1071 (`run_single`) - Exposes true and observed trajectories on the battery witness row.
- Test anchors:
  - tests/test_oasg_metrics.py:27 (`test_signature_equals_exposure_times_severity`) - Metric-level witness for OASG exposure.
  - tests/test_oasg_metrics.py:66 (`test_t1_executable_oasg_witness_has_observed_safe_true_unsafe_release`) - Constructive T1 witness regression.

### T2: Safety Preservation

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V3
- Program role: one_step_runtime_guarantee
- Scope note: Defended as a one-step true-state postcondition whose tightened margin already absorbs the one-step model-error allowance.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1079
- Proof location: appendices/app_c_full_proofs.tex:111
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
  - src/orius/dc3s/safety_filter_theory.py:37 (`tightened_soc_bounds`) - Computes the absorbed tightening margin.
  - src/orius/dc3s/guarantee_checks.py:64 (`check_soc_invariance`) - Checks the true-state one-step postcondition with model-error slack.
  - src/orius/certos/runtime.py:182 (`CertOSRuntime.validate_and_step`) - Enforces the runtime theorem hook for one-step safety.
- Test anchors:
  - tests/test_dc3s_guarantee_checks.py:26 (`test_guarantee_checks_pass_for_safe_action`) - Safe-action witness.
  - tests/test_t2_absorbed_tightening.py:32 (`test_t2_absorbed_tightening_randomized`) - Randomized absorbed-margin regression.

### T3a: ORIUS Core Envelope Derivation

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: runtime_risk_budget_derivation
- Scope note: Defended as the per-step envelope derivation under the explicit battery assumptions and the narrowed reliability-score interpretation.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1314
- Proof location: appendices/app_c_full_proofs.tex:179
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
  - src/orius/universal_theory/kernel.py:214 (`execute_universal_step`) - Attaches theorem-contract summaries to the typed runtime result.
  - src/orius/universal_theory/risk_bounds.py:141 (`build_t3a_contract_summary`) - Canonical theorem-contract summary for the narrowed T3a surface.
  - src/orius/dc3s/coverage_theorem.py:46 (`compute_expected_violation_bound`) - Backward-compatible wrapper.
- Test anchors:
  - tests/test_dc3s_coverage_theorem.py:59 (`test_compute_expected_violation_bound`) - Envelope algebra regression.
  - tests/test_universal_theory.py:38 (`test_run_universal_step_returns_structured_result`) - Runtime theorem-contract regression.

### T3b: ORIUS Core Aggregation Corollary

- Surface kind: corollary
- Defense tier: supporting_defended
- Proof tier: V1
- Program role: runtime_risk_budget_aggregation
- Scope note: Derived corollary that converts the predictable per-step budget into the episode-average envelope.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1409
- Proof location: appendices/app_c_full_proofs.tex:179
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
  - src/orius/universal_theory/risk_bounds.py:110 (`compute_episode_risk_bound`) - Computes the corollary bound directly.
- Test anchors:
  - tests/test_active_theorem_audit.py:252 (`test_defense_tiers_match_the_rebuilt_core`) - Audit regression for the T3 split.

### T4: Observation Necessity / No Free Safety

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: bounded_necessity_witness
- Scope note: Defended as the observation-necessity witness for the fixed-margin quality-ignorant controller class on the battery row with explicit arbitrage reachability.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1647
- Proof location: appendices/app_c_full_proofs.tex:241
- Assumptions used: ['A1', 'A2', 'A4', 'A11']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ["{'Definition': 'Quality-ignorant controller'}", "{'Lemma': 'Admissible fault sequence existence'}", "{'Lemma': 'No margin compensation for quality-ignorant controllers'}"]
- Weakest step: The constructive witness is intentionally class-scoped; widening the controller class would require a new proof.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The fixed-margin, quality-ignorant counterexample is now executable as a release witness and remains class-scoped.
- Severity if broken: high
- Remediation class: keep scope explicit - Preserve the fixed-margin quality-ignorant controller definition and avoid broadening the theorem into a claim that every ambiguity class is unsafe.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/supporting_results.py:182 (`verify_no_margin_compensation`) - Executable witness for fixed-margin insufficiency.
  - src/orius/dc3s/supporting_results.py:249 (`verify_quality_ignorant_release_counterexample`) - Constructive fixed-margin mandatory-release counterexample.
  - src/orius/cpsbench_iot/scenarios.py:179 (`generate_episode`) - Admissible degraded-observation episode generator.
- Test anchors:
  - tests/test_unification.py:67 (`test_unification_argument_w_t_never_drops_below_one`) - Supporting quality-ignorant limitation check.
  - tests/test_unification.py:80 (`test_t4_executable_quality_ignorant_counterexample`) - Executable T4 fixed-margin counterexample regression.

### T5: Finite-Horizon Certificate Validity

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: finite_horizon_certificate_validity
- Scope note: Defended as a finite-horizon runtime certificate theorem; if the reachable tube remains inside the safe set, releases governed by the certificate remain safe until expiry or an invalidating event.
- Statement location: appendices/proofs/T5_certificate_validity.tex:1
- Proof location: appendices/proofs/T5_certificate_validity.tex:12
- Assumptions used: ['A1', 'A2', 'A3', 'A4', 'A5']
- Typed obligations: []
- Unresolved assumptions: []
- Dependencies: ['Forward reachable tube containment', 'Certificate invalidation semantics']
- Weakest step: The theorem is finite-horizon and runtime-object scoped; it does not claim universal plant-time safety beyond the certified tube or after invalidating evidence.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The temporal-theorem helpers compute forward tubes, maximal contained horizons, fail-closed expiry, and invalidating events.
- Severity if broken: medium
- Remediation class: keep finite-horizon scope - Preserve the reachable-tube containment hypothesis and do not convert T5 into an unconditional future-step probability law.
- Legacy aliases: ['T5']
- Code anchors:
  - src/orius/dc3s/temporal_theorems.py:23 (`certificate_validity_horizon`) - Finite-horizon certificate helper.
  - src/orius/dc3s/temporal_theorems.py:44 (`forward_reachable_tube`) - Generic reachable-tube helper.
  - src/orius/dc3s/temporal_theorems.py:92 (`certificate_invalidating_event`) - Runtime invalidation event helper.
- Test anchors:
  - tests/test_T5_certificate_validity.py:17 (`test_certificate_horizon_positive_under_clean_telemetry`) - Positive clean-telemetry horizon regression.
  - tests/test_certos_horizon_expiry.py:21 (`test_release_fails_closed_when_horizon_less_than_one`) - Fail-closed horizon regression.
  - tests/test_certificate_invalidating_events.py:4 (`test_fresh_contradictory_evidence_invalidates_certificate`) - Contradictory evidence invalidation regression.

### T6: Certificate Expiration Bound

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V2_linked
- Program role: temporal_expiration_bound
- Scope note: Defended as the confidence-aware closed-form battery expiration theorem with explicit delta dependence and first-passage side conditions.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1884
- Proof location: appendices/app_c_full_proofs.tex:299
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
  - src/orius/universal_theory/battery_instantiation.py:135 (`certificate_expiration_bound`) - Canonical theorem-facing T6 helper.
  - src/orius/dc3s/half_life.py:227 (`compute_conservative_horizon`) - Supplemental first-passage helper.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:34 (`test_certificate_expiration_bound_uses_delta_aware_formula`) - Direct theorem formula regression.
  - tests/test_t6_expiration_bound.py:69 (`test_t6_empirical_subgaussian_bound`) - Empirical confidence check.

### T7: Feasible Fallback Existence

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V2_linked
- Program role: piecewise_fallback_existence
- Scope note: Defended as a battery-specific piecewise fallback theorem: safe hold on the interior, safe landing near the boundary, and fail-closed infeasibility otherwise.
- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1981
- Proof location: appendices/app_c_full_proofs.tex:346
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
  - src/orius/universal_theory/battery_instantiation.py:315 (`validate_battery_fallback`) - Piecewise theorem-facing T7 validator.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:67 (`test_certify_fallback_existence_passes_for_interior_soc`) - Positive interior witness.
  - tests/test_dc3s_temporal_theorems.py:80 (`test_certify_fallback_existence_recovers_near_lower_boundary`) - Lower-boundary recovery witness.
  - tests/test_dc3s_temporal_theorems.py:102 (`test_certify_fallback_existence_recovers_near_upper_boundary`) - Upper-boundary recovery witness.
  - tests/test_dc3s_temporal_theorems.py:124 (`test_certify_fallback_existence_fails_closed_when_boundary_recovery_is_infeasible`) - Fail-closed infeasibility witness.

### T8: Graceful Degradation Dominance with Useful Work

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: graceful_degradation_useful_work
- Scope note: Defended as paired-trace dominance; ORIUS weakly reduces true-state violations versus blind persistence while preserving a declared fraction of useful work.
- Statement location: appendices/proofs/T8_graceful_dominance.tex:1
- Proof location: appendices/proofs/T8_graceful_dominance.tex:9
- Assumptions used: ['A1', 'A2', 'A3', 'A4', 'A5', 'A8']
- Typed obligations: ['Paired graceful and uncontrolled policies are evaluated on the same admissible fault trace.', 'The useful-work threshold lambda is declared before evaluating dominance.']
- Unresolved assumptions: []
- Dependencies: ['Sequence comparison helper', 'Useful-work lower-bound check']
- Weakest step: The theorem proves a two-objective partial order on paired traces; it does not claim that shutdown is operationally equivalent to graceful degradation.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The policy frontier helper compares Blind, Shutdown, Ramp, and ORIUS policies with safety and useful-work gates.
- Severity if broken: medium
- Remediation class: keep useful-work gate - Preserve the useful-work threshold so immediate shutdown cannot satisfy T8 merely by being safe.
- Legacy aliases: []
- Code anchors:
  - src/orius/benchmarks/graceful_degradation.py:99 (`graceful_dominance_with_useful_work`) - T8 safety and useful-work dominance checker.
  - src/orius/benchmarks/graceful_degradation.py:125 (`evaluate_policy_frontier`) - Multi-policy safety-work frontier evaluator.
- Test anchors:
  - tests/test_T8_graceful_dominance.py:11 (`test_orius_weakly_dominates_blind_persistence_with_useful_work`) - Useful-work dominance regression.
  - tests/test_graceful_policy_useful_work.py:15 (`test_shutdown_has_zero_work_and_orius_preserves_nontrivial_work`) - Degenerate shutdown guard.

### T9: Impossibility of Quality-Ignorant Mandatory Release

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: impossibility_extension
- Scope note: Defended as a mandatory-release impossibility theorem; when an observation ambiguity class has empty common safe core, no observation-only controller can guarantee true-state safety for every latent state in that class; domain discharge is evidence, not an extra hidden assumption.
- Statement location: appendices/proofs/T9_no_free_safety.tex:1
- Proof location: appendices/proofs/T9_no_free_safety.tex:11
- Assumptions used: []
- Typed obligations: ['Mandatory release policy depends only on the observation.', 'The observation ambiguity class has empty common safe core.']
- Unresolved assumptions: []
- Dependencies: ['T4', 'Common safe-core witness']
- Weakest step: The impossibility applies only to mandatory observation-only release; optional abstention, fallback, uncertainty expansion, or denial of release are outside the lower-bound class.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The ambiguity helpers build ambiguity classes, compute common safe cores, and find mandatory-release counterexamples.
- Severity if broken: high
- Remediation class: keep mandatory-release scope - Keep T9 tied to empty-safe-core witnesses and do not restate it as a universal impossibility for policies allowed to abstain or fail closed; domain discharge is evidence, not an extra hidden assumption.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/ambiguity.py:18 (`compute_common_safe_core`) - Common safe-core helper.
  - src/orius/universal_theory/ambiguity.py:36 (`find_mandatory_release_counterexample`) - Mandatory-release counterexample helper.
- Test anchors:
  - tests/test_T9_mandatory_release_impossibility.py:9 (`test_empty_safe_core_counterexample_exists`) - Empty-safe-core counterexample regression.

### T10: Boundary-Indistinguishability Lower Bound

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: information_lower_bound_extension
- Scope note: Defended as a two-state boundary lower bound; observation-only mandatory release has worst-case risk at least (1-epsilon)/2 when two boundary states have disjoint safe-action sets and observation laws within TV epsilon; domain discharge is evidence, not an extra hidden assumption.
- Statement location: appendices/proofs/T10_boundary_lower_bound.tex:1
- Proof location: appendices/proofs/T10_boundary_lower_bound.tex:11
- Assumptions used: []
- Typed obligations: ['Two latent boundary states induce observation distributions within the stated TV radius.', 'The two state-conditioned safe-action sets are disjoint.']
- Unresolved assumptions: []
- Dependencies: ['Le Cam two-point lemma']
- Weakest step: The result is a two-hypothesis lower bound, not a sharp global frontier for arbitrary policies or observation models.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The boundary-indistinguishability helper estimates total variation and evaluates the exact two-state lower-bound curve.
- Severity if broken: high
- Remediation class: keep two-state scope - Do not describe T10 as a full minimax frontier; the scoped minimax row is T_minimax and still lacks a global optimality claim; domain discharge is evidence, not an extra hidden assumption.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/boundary_indistinguishability.py:14 (`two_state_lower_bound`) - Two-state lower-bound helper.
  - src/orius/universal_theory/boundary_indistinguishability.py:8 (`estimate_total_variation`) - Total-variation estimator.
- Test anchors:
  - tests/test_T10_boundary_lower_bound.py:7 (`test_tv_extremes`) - TV endpoint lower-bound regression.

### T11: Typed structural transfer theorem, T11

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: typed_transfer_theorem
- Scope note: Defended as the forward four-obligation one-step transfer theorem; the converse remains a separate structural failure proposition.
- Statement location: chapters/ch37_universality_completeness.tex:386
- Proof location: appendices/app_c_full_proofs.tex:388
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
  - src/orius/universal_theory/contracts.py:630 (`ContractVerifier.validate_runtime_step`) - Authoritative typed theorem surface.
  - src/orius/universal_theory/contracts.py:822 (`ContractVerifier.build_transfer_theorem_summary`) - Four-obligation theorem-contract summary over runtime artifacts.
  - src/orius/dc3s/theoretical_guarantees.py:722 (`evaluate_structural_transfer`) - Four-obligation executable witness.
  - src/orius/universal/contract.py:287 (`ContractVerifier.check`) - Supporting five-invariant mini-harness only.
- Test anchors:
  - tests/test_theoretical_guarantees_hypothesis.py:76 (`test_t11_transfer_requires_all_four_obligations`) - Active theorem regression.
  - tests/test_universal_contract.py:77 (`test_passes_all_five_invariants`) - Supporting harness regression.

### T10_T11_ObservationAmbiguitySandwich: T10_T11_ObservationAmbiguitySandwich: Covered Observation-Ambiguity Optimality

- Surface kind: corollary
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: supporting_observation_ambiguity_optimality
- Scope note: Supporting optimality corollary under covered observation ambiguity; contract-universal, not unrestricted-global; it does not assert unrestricted global optimality for all physical AI systems.
- Statement location: chapters/ch37_universality_completeness.tex:434
- Proof location: appendices/app_c_full_proofs.tex:423
- Assumptions used: []
- Typed obligations: ['Observation ambiguity classes are explicit.', 'The common safe core is computed as the intersection of state-conditioned safe action sets.', 'ORIUS releases only actions safe for every state in its covered uncertainty set.', 'Probabilistic coverage is reported as an alpha-bound rather than unconditional zero violation.', 'Coverage, safe-set correctness, repair membership, fallback admissibility, and adapter validity are typed proof obligations rather than empirical metrics.']
- Unresolved assumptions: []
- Dependencies: ['T10', 'T11', 'Common safe-core witness']
- Weakest step: The lower bound is a Bayes ambiguity-risk bound, not the false claim that differing safe sets alone imply unavoidable violation.
- Rigor rating: proof_runtime_linked
- Code correspondence: matches - The executable witness computes common safe cores, observation-only Bayes lower bounds, and covered ORIUS release upper bounds with explicit alpha-bound semantics.
- Severity if broken: high
- Remediation class: keep scoped optimality - Keep the theorem phrased as safety-optimal under covered observation ambiguity and contract-universal, not unrestricted-global.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/observation_ambiguity.py:169 (`build_observation_ambiguity_contract_summary`) - Publication-facing executable witness for the lower/upper sandwich.
  - src/orius/universal_theory/observation_ambiguity.py:55 (`observation_only_bayes_lower_bound`) - Computes the observation-only Bayes lower bound.
  - src/orius/universal_theory/observation_ambiguity.py:133 (`verify_covered_orius_release`) - Certifies the covered ORIUS upper-bound side.
- Test anchors:
  - tests/test_observation_ambiguity_optimality.py:20 (`test_different_safe_sets_do_not_imply_unavoidable_violation`) - Guards against the false unsafe-ambiguity overclaim.
  - tests/test_observation_ambiguity_optimality.py:51 (`test_orius_covered_release_certifies_zero_violation`) - Covered-release upper-bound regression.
  - tests/test_observation_ambiguity_optimality.py:65 (`test_probabilistic_coverage_is_alpha_bounded_not_unconditional_zero`) - Alpha-bound semantics regression.

### T11_AV_BrakeHold: T11_AV_BrakeHold: AV Brake-Hold Runtime Lemma

- Surface kind: runtime_linked_lemma
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: bounded_domain_transfer_lemma
- Scope note: Supporting AV runtime lemma under forward-only T11; bounded to promoted ORIUS replay rows and the longitudinal brake-hold postcondition.
- Statement location: appendices/app_c_full_proofs.tex:457
- Proof location: appendices/app_c_full_proofs.tex:457
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
  - src/orius/universal_theory/domain_runtime_contracts.py:199 (`DomainRuntimeContractWitness`) - Bounded runtime witness type.
  - src/orius/av_waymo/runtime.py:1633 (`_run_episode`) - Emits AV T11 and brake-hold witness fields.
  - scripts/build_domain_runtime_contract_artifacts.py:202 (`build_domain_runtime_contract_artifacts`) - Builds publication-facing witness artifacts.
- Test anchors:
  - tests/test_domain_runtime_contract_witnesses.py:30 (`test_passing_av_row_links_t11_certificate_and_postcondition`) - Passing AV witness regression.
  - tests/test_domain_runtime_contract_witnesses.py:51 (`test_failing_av_postcondition_fails_closed_even_with_certificate`) - Failing AV witness regression.

### T11_HC_FailSafeRelease: T11_HC_FailSafeRelease: Healthcare Fail-Safe Release Runtime Lemma

- Surface kind: runtime_linked_lemma
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: bounded_domain_transfer_lemma
- Scope note: Supporting healthcare runtime lemma under forward-only T11; bounded to promoted MIMIC monitoring rows and the fail-safe alert-release postcondition.
- Statement location: appendices/app_c_full_proofs.tex:477
- Proof location: appendices/app_c_full_proofs.tex:477
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
  - src/orius/universal_theory/domain_runtime_contracts.py:199 (`DomainRuntimeContractWitness`) - Bounded runtime witness type.
  - scripts/build_healthcare_runtime_artifacts.py:616 (`_run_orius_episode`) - Emits healthcare T11 and fail-safe witness fields.
  - scripts/build_domain_runtime_contract_artifacts.py:202 (`build_domain_runtime_contract_artifacts`) - Builds publication-facing witness artifacts.
- Test anchors:
  - tests/test_domain_runtime_contract_witnesses.py:72 (`test_passing_healthcare_row_links_t11_certificate_and_postcondition`) - Passing healthcare witness regression.
  - tests/test_domain_runtime_contract_witnesses.py:113 (`test_missing_t11_status_fails_closed`) - Missing T11 status regression.

### T6_AV_FallbackValidity: T6_AV_FallbackValidity: AV One-Step Fallback Certificate Lemma

- Surface kind: runtime_linked_lemma
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: bounded_domain_certificate_validity
- Scope note: Supporting AV certificate-validity lemma under T6/T11; degraded full-brake fallback releases are one-step valid only.
- Statement location: appendices/app_c_full_proofs.tex:496
- Proof location: appendices/app_c_full_proofs.tex:496
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
  - src/orius/universal_theory/domain_validity.py:122 (`domain_certificate_validity_semantics`) - Bounded fallback and T6-style hold validity helper.
  - src/orius/av_waymo/runtime.py:1247 (`WaymoAVDomainAdapter.emit_certificate`) - AV certificate emitter.
- Test anchors:
  - tests/test_domain_runtime_contract_witnesses.py:130 (`test_av_fallback_certificate_validity_is_one_step_only`) - AV fallback validity regression.

### T6_HC_FallbackValidity: T6_HC_FallbackValidity: Healthcare One-Step Fallback Certificate Lemma

- Surface kind: runtime_linked_lemma
- Defense tier: supporting_defended
- Proof tier: V1_runtime_linked
- Program role: bounded_domain_certificate_validity
- Scope note: Supporting healthcare certificate-validity lemma under T6/T11; degraded max-alert fallback releases are one-step valid only.
- Statement location: appendices/app_c_full_proofs.tex:513
- Proof location: appendices/app_c_full_proofs.tex:513
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
  - src/orius/universal_theory/domain_validity.py:122 (`domain_certificate_validity_semantics`) - Bounded fallback and T6-style hold validity helper.
  - src/orius/universal_framework/healthcare_adapter.py:406 (`HealthcareDomainAdapter.emit_certificate`) - Healthcare certificate emitter.
- Test anchors:
  - tests/test_domain_runtime_contract_witnesses.py:149 (`test_healthcare_fallback_certificate_validity_is_one_step_only`) - Healthcare fallback validity regression.

### T_EQ_Battery_RuntimeArtifactPackage: T_EQ_Battery_RuntimeArtifactPackage: Battery Equal Artifact Discipline Package

- Surface kind: artifact_discipline_gate
- Defense tier: supporting_defended
- Proof tier: V1_artifact_runtime_linked
- Program role: equal_domain_artifact_discipline
- Scope note: Supporting artifact-discipline row; it checks battery theorem, runtime, comparator, ablation, negative-control, utility, and reproducibility evidence without changing the flagship theorem tier.
- Statement location: appendices/app_c_full_proofs.tex:529
- Proof location: appendices/app_c_full_proofs.tex:529
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
  - scripts/build_equal_domain_artifact_discipline.py:768 (`build_equal_domain_artifact_discipline`) - Builds the equal artifact-discipline gate.
  - scripts/validate_equal_domain_artifact_discipline.py:82 (`main`) - Validates fail-closed equal-domain evidence.
- Test anchors:
  - tests/test_equal_domain_artifact_discipline.py:70 (`test_equal_domain_artifact_discipline_gates_pass_for_all_domains`) - Battery equal-discipline regression.

### T_EQ_AV_RuntimeArtifactPackage: T_EQ_AV_RuntimeArtifactPackage: AV Equal Artifact Discipline Package

- Surface kind: artifact_discipline_gate
- Defense tier: supporting_defended
- Proof tier: V1_artifact_runtime_linked
- Program role: equal_domain_artifact_discipline
- Scope note: Supporting artifact-discipline row for the bounded AV replay contract; it does not assert full autonomous-driving field closure.
- Statement location: appendices/app_c_full_proofs.tex:548
- Proof location: appendices/app_c_full_proofs.tex:548
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
  - src/orius/av_waymo/runtime.py:1986 (`run_runtime_dry_run`) - Emits AV runtime and comparator artifacts.
  - scripts/build_equal_domain_artifact_discipline.py:768 (`build_equal_domain_artifact_discipline`) - Builds the equal artifact-discipline gate.
- Test anchors:
  - tests/test_equal_domain_artifact_discipline.py:70 (`test_equal_domain_artifact_discipline_gates_pass_for_all_domains`) - AV equal-discipline regression.
  - tests/test_av_waymo_dry_run.py:155 (`test_waymo_dry_run_smoke`) - AV runtime comparator smoke regression.

### T_EQ_HC_RuntimeArtifactPackage: T_EQ_HC_RuntimeArtifactPackage: Healthcare Equal Artifact Discipline Package

- Surface kind: artifact_discipline_gate
- Defense tier: supporting_defended
- Proof tier: V1_artifact_runtime_linked
- Program role: equal_domain_artifact_discipline
- Scope note: Supporting artifact-discipline row for the bounded healthcare monitoring contract; it does not assert regulated clinical deployment readiness.
- Statement location: appendices/app_c_full_proofs.tex:566
- Proof location: appendices/app_c_full_proofs.tex:566
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
  - scripts/build_healthcare_runtime_artifacts.py:884 (`build_healthcare_runtime_artifacts`) - Emits healthcare runtime and comparator artifacts.
  - scripts/build_equal_domain_artifact_discipline.py:768 (`build_equal_domain_artifact_discipline`) - Builds the equal artifact-discipline gate.
- Test anchors:
  - tests/test_equal_domain_artifact_discipline.py:70 (`test_equal_domain_artifact_discipline_gates_pass_for_all_domains`) - Healthcare equal-discipline regression.
  - tests/test_healthcare_runtime_artifacts.py:59 (`test_build_healthcare_runtime_artifacts_emits_domain_native_runtime_surfaces`) - Healthcare runtime comparator smoke regression.

### L1: L1: Reliability-Monotone Inflation

- Surface kind: lemma
- Defense tier: supporting_defended
- Proof tier: V1
- Program role: runtime_monotonicity_law
- Scope note: Flagship lemma in the runtime monotonicity law suite: reliability-inflated margin q_t/(w_t+epsilon) is nonincreasing in reliability.
- Statement location: appendices/proofs/L1_reliability_inflation.tex:1
- Proof location: appendices/proofs/L1_reliability_inflation.tex:5
- Assumptions used: []
- Typed obligations: ['q_t is positive.', 'epsilon is positive.', 'w_t lies in [0,1].']
- Unresolved assumptions: []
- Dependencies: ['Reliability-inflated runtime margin definition']
- Weakest step: The lemma is a runtime proxy monotonicity statement, not a physical law for all sensors.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The runtime law helper directly evaluates the monotonicity relation.
- Severity if broken: medium
- Remediation class: keep lemma scope - Keep L1 as a monotonicity lemma and do not revive the old rate-distortion converse wording.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/runtime_laws.py:6 (`verify_inflation_monotonicity`) - Runtime monotonicity helper.
- Test anchors:
  - tests/test_runtime_law_suite.py:9 (`test_runtime_laws_smoke`) - Runtime law suite regression.

### L2: L2: Safe-Set Antitonicity

- Surface kind: lemma
- Defense tier: supporting_defended
- Proof tier: V1
- Program role: runtime_safe_set_law
- Scope note: Flagship lemma in the runtime monotonicity law suite: enlarging the uncertainty set can only shrink the common safe-action set.
- Statement location: appendices/proofs/L2_safe_set_antitonicity.tex:1
- Proof location: appendices/proofs/L2_safe_set_antitonicity.tex:7
- Assumptions used: []
- Typed obligations: ['The compared uncertainty sets satisfy X_1 subset X_2.', 'Safe-action sets are evaluated by the same domain safe-action correspondence C.']
- Unresolved assumptions: []
- Dependencies: ['Common safe-core definition']
- Weakest step: The lemma is pure set antitonicity; it does not characterize safe-set geometry beyond set inclusion.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The runtime law helper checks the safe-core subset relation.
- Severity if broken: medium
- Remediation class: keep lemma scope - Keep L2 as set antitonicity and do not restore the old capacity-proxy bridge as a defended theorem.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/runtime_laws.py:12 (`verify_safe_set_antitonicity`) - Runtime safe-set antitonicity helper.
- Test anchors:
  - tests/test_runtime_law_suite.py:9 (`test_runtime_laws_smoke`) - Runtime law suite regression.

### L3: L3: Intervention Threshold

- Surface kind: lemma
- Defense tier: supporting_defended
- Proof tier: V1
- Program role: runtime_intervention_law
- Scope note: Flagship lemma in the runtime monotonicity law suite: a candidate action outside the common safe core must be repaired, replaced by fallback, or denied.
- Statement location: appendices/proofs/L3_intervention_threshold.tex:1
- Proof location: appendices/proofs/L3_intervention_threshold.tex:6
- Assumptions used: []
- Typed obligations: ['Candidate action membership is checked against the common safe core.', 'Certified release can repair, fallback, or deny release.']
- Unresolved assumptions: []
- Dependencies: ['L2', 'T3a']
- Weakest step: The lemma specifies a runtime intervention obligation, not an independent converse theorem.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The runtime law helper checks when a candidate lies outside the common safe core.
- Severity if broken: medium
- Remediation class: keep lemma scope - Keep L3 as the runtime intervention threshold and do not revive the old critical-capacity theorem wording.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/runtime_laws.py:18 (`verify_intervention_threshold`) - Runtime intervention threshold helper.
- Test anchors:
  - tests/test_runtime_law_suite.py:9 (`test_runtime_laws_smoke`) - Runtime law suite regression.

### L4: L4: Observation-Ambiguity Safety Sandwich

- Surface kind: lemma
- Defense tier: supporting_defended
- Proof tier: V1
- Program role: runtime_ambiguity_sandwich
- Scope note: Flagship lemma tying the lower side from T9/T10 to the upper side from covered ORIUS release under T2/T3.
- Statement location: appendices/proofs/L4_ambiguity_sandwich.tex:1
- Proof location: appendices/proofs/L4_ambiguity_sandwich.tex:6
- Assumptions used: []
- Typed obligations: ['Lower-side ambiguity risk is supplied by T9 or T10.', 'Upper-side covered-release risk is supplied by T2 or T3.']
- Unresolved assumptions: []
- Dependencies: ['T9', 'T10', 'T2', 'T3a']
- Weakest step: The sandwich depends on the scoped lower-bound class and covered-release upper-bound assumptions; it is not unrestricted global optimality.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The runtime law helper checks compatible lower and upper risk surfaces.
- Severity if broken: medium
- Remediation class: keep lemma scope - Keep L4 as a runtime-law sandwich and preserve the lower-class and coverage-miss claim boundaries.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/runtime_laws.py:22 (`verify_ambiguity_sandwich`) - Runtime ambiguity-sandwich helper.
- Test anchors:
  - tests/test_runtime_law_suite.py:9 (`test_runtime_laws_smoke`) - Runtime law suite regression.

### T11_Byzantine: T11_Byzantine: Byzantine-Robust Reliability Aggregation

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: robustness_extension
- Scope note: Defended robust reliability aggregation theorem for bounded scores with at most b<n/2 Byzantine channels and honest scores in a radius-rho interval.
- Statement location: appendices/proofs/T11Byz_robust_reliability.tex:1
- Proof location: appendices/proofs/T11Byz_robust_reliability.tex:9
- Assumptions used: []
- Typed obligations: ['Reliability sub-scores lie in [0,1].', 'Byzantine budget b is strictly less than n/2.', 'Honest scores lie in an interval of width 2rho centered at the honest score.']
- Unresolved assumptions: []
- Dependencies: ['Trimmed-mean robustness argument']
- Weakest step: The theorem makes no guarantee when b>=n/2 or when honest-channel concentration fails.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The robust OQE helpers implement trimmed mean, median-of-means, adversarial channel injection, and honest-interval error checks.
- Severity if broken: medium
- Remediation class: keep Byzantine budget scope - Preserve the b<n/2 condition and honest-interval hypothesis in all manuscript-facing statements.
- Legacy aliases: []
- Code anchors:
  - src/orius/dc3s/quality.py:561 (`trimmed_mean_reliability`) - Byzantine-budget trimmed reliability helper.
  - src/orius/dc3s/quality.py:610 (`byzantine_reliability_error_bound`) - Honest-interval error-bound checker.
- Test anchors:
  - tests/test_T11Byz_robust_oqe.py:25 (`test_byzantine_error_bound_holds_for_honest_interval`) - Honest-interval bound regression.
  - tests/test_adversarial_reliability_channels.py:14 (`test_robust_aggregators_reduce_extreme_channel_effect`) - Adversarial channel regression.

### T_stale_decay: T_stale_decay: Stale-Hold Uncertainty Growth

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: stale_hold_uncertainty_growth
- Scope note: Defended stale-hold uncertainty theorem: under bounded latent drift, stale observation radius grows conservatively as r_t+Ls.
- Statement location: appendices/proofs/Tstale_uncertainty_growth.tex:1
- Proof location: appendices/proofs/Tstale_uncertainty_growth.tex:7
- Assumptions used: []
- Typed obligations: ['No fresh observation arrives for s steps.', 'Latent state drift is bounded by L per step.', 'Certified release at the stale step is checked over the enlarged set.']
- Unresolved assumptions: []
- Dependencies: ['Triangle inequality', 'T5 certificate horizon check']
- Weakest step: The theorem is bounded-drift containment, not a claim that an exponential reliability decay schedule is a physical law.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The stale-hold helpers compute linear radius growth and certificate-horizon expiry under stale intervals.
- Severity if broken: medium
- Remediation class: keep bounded-drift scope - Preserve the bounded-drift stale-hold radius statement and avoid deriving physical sensing laws from design schedules.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/stale_decay.py:6 (`stale_uncertainty_growth`) - Bounded-drift radius helper.
  - src/orius/universal_theory/stale_decay.py:16 (`stale_certificate_expiry`) - Stale certificate horizon helper.
- Test anchors:
  - tests/test_Tstale_uncertainty_growth.py:4 (`test_stale_growth_linear_and_expiry`) - Stale growth and expiry regression.

### T_minimax: T_minimax: Finite Ambiguity-Class Minimax Lower Bound

- Surface kind: scoped_flagship_theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: finite_ambiguity_minimax_lower_bound
- Scope note: Scoped flagship theorem for finite two-state boundary ambiguity classes; it is a lower bound only and does not claim global minimax optimality.
- Statement location: appendices/proofs/Tminimax_finite_ambiguity.tex:1
- Proof location: appendices/proofs/Tminimax_finite_ambiguity.tex:11
- Assumptions used: []
- Typed obligations: ['The distribution class is finite two-state boundary problems.', 'Observation laws satisfy the stated TV bound.', 'Safe-action sets are disjoint.']
- Unresolved assumptions: []
- Dependencies: ['T10']
- Weakest step: This is a scoped lower bound over a finite ambiguity class; no matching ORIUS upper bound is asserted as global optimality.
- Rigor rating: paper_rigorous
- Code correspondence: matches - The minimax helper delegates to the two-state boundary lower-bound calculator and exposes optional ORIUS coverage upper bounds separately.
- Severity if broken: medium
- Remediation class: keep scoped minimax wording - Do not use the word optimal unless a matching upper bound is added for the same policy and distribution class.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/minimax_boundary.py:8 (`finite_ambiguity_minimax_lower_bound`) - Scoped minimax lower-bound helper.
- Test anchors:
  - tests/test_Tminimax_finite_ambiguity.py:4 (`test_minimax_scoped_lower_bound`) - Scoped minimax lower-bound regression.

### T_sensor_converse: T_sensor_converse: Sensor Necessity Under Adapter Semantics

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: sensor_necessity_adapter_semantics
- Scope note: Defended sensor-necessity theorem under adapter semantics: if an omitted latent coordinate changes safe-action sets to disjoint cores, observation-only mandatory release cannot certify safety without sensing, expansion, fallback, or denial.
- Statement location: appendices/proofs/Tsensor_sensor_necessity.tex:1
- Proof location: appendices/proofs/Tsensor_sensor_necessity.tex:9
- Assumptions used: []
- Typed obligations: ['Safe-action map depends on the omitted latent coordinate.', 'Two states differing only in that coordinate have disjoint safe-action sets.', 'Policy class is observation-only mandatory release.']
- Unresolved assumptions: []
- Dependencies: ['T9']
- Weakest step: The theorem reduces to T9 and is valid only when sensor ablation creates an empty common safe core.
- Rigor rating: paper_rigorous
- Code correspondence: matches - Sensor ablation helpers remove observation keys, recompute safe cores, and identify critical sensor drops.
- Severity if broken: medium
- Remediation class: keep adapter-semantics scope - Preserve the missing-coordinate and disjoint-safe-core hypotheses; do not claim every low-quality sensor is universally necessary.
- Legacy aliases: []
- Code anchors:
  - src/orius/universal_theory/sensor_necessity.py:20 (`critical_sensor_test`) - Empty-core sensor ablation helper.
  - src/orius/universal_theory/sensor_necessity.py:10 (`sensor_ablation`) - Sensor ablation helper.
- Test anchors:
  - tests/test_Tsensor_necessity.py:4 (`test_sensor_drop_and_empty_core_trigger`) - Sensor necessity regression.

### T_trajectory_PAC: T_trajectory_PAC: Finite-Horizon PAC Release Certificate

- Surface kind: theorem
- Defense tier: flagship_defended
- Proof tier: V1
- Program role: trajectory_certificate
- Scope note: Defended as the implemented Bonferroni/union-bound trajectory certificate and nothing stronger.
- Statement location: appendices/proofs/TPAC_trajectory_certificate.tex:1
- Proof location: appendices/proofs/TPAC_trajectory_certificate.tex:14
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
  - src/orius/universal_theory/risk_bounds.py:576 (`trajectory_union_bound_certificate`) - Canonical certificate helper.
- Test anchors:
  - tests/test_TPAC_trajectory_certificate.py:10 (`test_sum_of_budgets_below_delta_passes`) - Union-bound certificate regression.
