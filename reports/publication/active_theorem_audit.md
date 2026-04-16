# Active T1-T11 Theorem Audit

This file is generated from `scripts/_active_theorem_program.py` and is the reconciled active audit surface for the defended T1--T11 program.

## Summary

- Active theorem rows: 11
- Rigor counts: {'plausible-but-informal': 3, 'has-a-hole': 4, 'solid': 1, 'broken': 3}
- Code correspondence counts: {'partial': 8, 'matches': 3}

## Namespace Drift

### src/orius/dc3s/coverage_theorem.py and tests/test_conditional_coverage.py

- Issue: Legacy auxiliary group-coverage and Hoeffding helpers were labeled as Theorem 9 and Theorem 10 even though the active T9/T10 program now means universal impossibility and the boundary-indistinguishability lower bound.
- Impact: Silent theorem-number reuse makes the code surface contradict the active manuscript namespace.
- Status: resolved
- Remediation: Keep these helpers explicitly tagged as legacy auxiliary coverage analyses and never as active T9/T10 witnesses.

### src/orius/universal/contract.py, tests/test_universal_contract.py, and tests/test_unification.py

- Issue: The five-invariant reference harness was described as if satisfying it automatically discharged the full active T11 theorem and the episode-level bound.
- Impact: That story is wider than the live manuscript, which transfers only the one-step theorem unless a separate per-step risk budget is provided.
- Status: resolved
- Remediation: Describe the five invariants as supporting reference-harness checks and route theorem-facing claims through the typed T11 surface instead.

### reports/publication/theorem_surface_register.csv

- Issue: The register is an inventory, not a one-row-per-theorem program: T1-T4 each appear as both general-CPS and main-theorem rows.
- Impact: Consumers can accidentally pick the broader row and overstate the defended theorem surface.
- Status: open
- Remediation: Use the active T1--T11 audit artifact as the single reconciled theorem map and keep the register as raw inventory only.

## Per-Theorem Audit

### T1: OASG Existence

- Statement location: chapters_merged/ch04_theoretical_foundations.tex:800
- Proof location: appendices/app_c_full_proofs.tex:31
- Assumptions used: ['A1', 'A2', 'A4', 'Boundary proximity under arbitrage remains non-trivial on the defended battery witness.']
- Unresolved assumptions: ['Boundary proximity under arbitrage remains non-trivial on the defended battery witness.']
- Dependencies: ['Lemma: Observation gap under dropout', 'Lemma: Boundary proximity under arbitrage']
- Weakest step: Lemma boundary-proximity only argues that non-zero price variance pushes the controller near a boundary; it does not derive the positive boundary-visit fraction from the dispatch model.
- Rigor rating: plausible-but-informal
- Code correspondence: partial - The runtime and metrics code represent observed-vs-true disagreement explicitly, but there is no executable witness for the proof's boundary-reachability step.
- Severity if broken: high
- Remediation class: add assumption - Either register a battery-specific boundary-reachability assumption explicitly or prove it from the dispatch model.
- Code anchors:
  - src/orius/orius_bench/metrics_engine.py:91 (`compute_oasg`) - Counts observation-action safety gaps.
  - src/orius/cpsbench_iot/runner.py:1040 (`run_single`) - Empirical witness path with true and observed trajectories.
- Test anchors:
  - tests/test_orius_bench.py:151 (`test_oasg_counts_observation_action_safety_gap`) - Metric-level witness only; not a proof of existence under optimal arbitrage.

### T2: Safety Preservation

- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1039
- Proof location: appendices/app_c_full_proofs.tex:44
- Assumptions used: ['A1', 'A2', 'A3', 'A4', 'A5', 'A7', 'The tightening already absorbs the one-step model-error allowance.']
- Unresolved assumptions: ['The tightening already absorbs the one-step model-error allowance.']
- Dependencies: ['Proposition: Inflated set contains the current state', 'Proposition: Tightened feasibility implies true feasibility']
- Weakest step: The proof says the model-error allowance is already absorbed by the tightening, but that absorption is asserted in prose rather than derived from the actual FTIT construction.
- Rigor rating: plausible-but-informal
- Code correspondence: matches - The main conditional one-step claim matches the guarantee-check and shield surface, even though the earlier general-CPS statement in the same chapter still contains a broader probabilistic line.
- Severity if broken: high
- Remediation class: strengthen proof - Make the model-error buffer explicit in the theorem statement or add a lemma tying the implemented tightening to the residual bound.
- Namespace drift note: The same chapter still carries a broader general-CPS statement with an unearned probability line; the active audit selects the main conditional theorem row instead.
- Code anchors:
  - src/orius/dc3s/guarantee_checks.py:54 (`check_soc_invariance`) - Implements the one-step battery safety check.
  - src/orius/dc3s/guarantee_checks.py:85 (`evaluate_guarantee_checks`) - Collects the deterministic safety predicates.
- Test anchors:
  - tests/test_dc3s_guarantee_checks.py:25 (`test_guarantee_checks_pass_for_safe_action`) - Checks the implemented one-step safety predicate only.

### T3: ORIUS Core Bound

- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1274
- Proof location: appendices/app_c_full_proofs.tex:98
- Assumptions used: ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'Battery risk-envelope contract: P[Z_t=1 | H_t] <= alpha * (1 - w_t).', 'w_t is a runtime reliability score, not a probability by definition.']
- Unresolved assumptions: ['Battery risk-envelope contract: P[Z_t=1 | H_t] <= alpha * (1 - w_t).', 'w_t is a runtime reliability score, not a probability by definition.']
- Dependencies: ['Lemma: Aggregation under a predictable risk budget', 'Battery-specific theorem-local risk budget contract']
- Weakest step: The bound is only as rigorous as the unproved battery risk-envelope contract; the proof never derives that contract from conformal coverage, detector lag, or the OQE construction.
- Rigor rating: has-a-hole
- Code correspondence: partial - The helpers compute the envelope and expose the missing assumptions honestly, but no code path justifies the battery risk budget from the implemented OQE or calibration stack.
- Severity if broken: critical
- Remediation class: add assumption - Either promote the battery risk-envelope contract to an explicit defended assumption or add a new battery-specific lemma that derives it from the implemented calibration logic.
- Code anchors:
  - src/orius/universal_theory/risk_bounds.py:97 (`compute_episode_risk_bound`) - Implements only the aggregation envelope, not the bridge to the budget contract.
  - src/orius/universal_theory/risk_bounds.py:92 (`risk_envelope_assumptions`) - Makes the missing theorem-local assumptions explicit.
  - src/orius/dc3s/coverage_theorem.py:44 (`compute_expected_violation_bound`) - Backward-compatible wrapper around the narrowed T3 helper.
- Test anchors:
  - tests/test_dc3s_coverage_theorem.py:59 (`test_compute_expected_violation_bound`) - Verifies the algebraic envelope, not the bridge from runtime reliability to residual-risk probability.

### T4: No Free Safety

- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1537
- Proof location: appendices/app_c_full_proofs.tex:155
- Assumptions used: ['A1', 'A2', 'A4', 'Quality-ignorant controllers admit a fixed effective margin m^pi.', 'An admissible concentrated fault window can exceed delta_bnd + m^pi.']
- Unresolved assumptions: ['Quality-ignorant controllers admit a fixed effective margin m^pi.', 'An admissible concentrated fault window can exceed delta_bnd + m^pi.']
- Dependencies: ['Definition: Quality-ignorant controller', 'Lemma: Admissible fault sequence existence', 'Lemma: No margin compensation for quality-ignorant controllers', 'Theorem T1 boundary-proximity witness']
- Weakest step: The proof reduces every quality-ignorant controller to a fixed scalar margin m^pi without proving that the whole controller class admits that abstraction.
- Rigor rating: has-a-hole
- Code correspondence: partial - The repo has empirical fault generators and quality-ignorant baseline reductions, but no executable battery witness that constructs the exact three-phase adversarial sequence used in the proof.
- Severity if broken: high
- Remediation class: strengthen proof - Define the quality-ignorant controller class more tightly and prove that the fault witness works for that class rather than for an informal fixed-margin caricature.
- Code anchors:
  - src/orius/cpsbench_iot/scenarios.py:171 (`generate_episode`) - Supplies admissible degraded-observation episodes.
  - src/orius/cpsbench_iot/runner.py:1040 (`run_single`) - Runs quality-ignorant baselines against the witness battery plant.
- Test anchors:
  - tests/test_unification.py:65 (`test_unification_argument_w_t_never_drops_below_one`) - Supporting quality-ignorant limitation check, not the theorem's adversarial battery witness.

### T5: Certificate validity horizon

- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1739
- Proof location: appendices/app_c_full_proofs.tex:184
- Assumptions used: ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'Future drift remains summarized by the sub-Gaussian sigma_d * sqrt(Delta) tube.']
- Unresolved assumptions: ['Future drift remains summarized by the sub-Gaussian sigma_d * sqrt(Delta) tube.']
- Dependencies: ['Definition: Safety certificate', 'Definition: Forward tube']
- Weakest step: The chapter proof jumps from issuance-time coverage to future-step safety probabilities without an explicit temporal bridge beyond the stylized drift tube.
- Rigor rating: plausible-but-informal
- Code correspondence: matches - The deterministic horizon helper matches the stated maximal containment problem, even though the probabilistic interpretation in the chapter remains compressed.
- Severity if broken: medium
- Remediation class: strengthen proof - Separate the deterministic tube-containment statement from any future-step probability claim, or prove the temporal concentration step explicitly.
- Code anchors:
  - src/orius/universal_theory/battery_instantiation.py:34 (`forward_tube`) - Builds the battery forward tube.
  - src/orius/universal_theory/battery_instantiation.py:66 (`certificate_validity_horizon`) - Searches the maximal horizon whose tube stays inside bounds.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:33 (`test_certificate_validity_horizon_matches_expiration_bound_for_zero_action`) - Checks deterministic containment logic.

### T6: Certificate expiration bound

- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1769
- Proof location: appendices/app_c_full_proofs.tex:203
- Assumptions used: ['A4', 'A6', 'A7', 'The forward-tube boundary crossing is controlled by the drift scale alone.']
- Unresolved assumptions: ['The forward-tube boundary crossing is controlled by the drift scale alone.']
- Dependencies: ['Definition: Forward tube']
- Weakest step: The proof is scalar and conservative, but it silently assumes the action contribution is already frozen inside the issued certificate.
- Rigor rating: solid
- Code correspondence: matches - The implemented lower bound is exactly the analytical floor(delta_bnd^2 / sigma_d^2) expression defended in the text.
- Severity if broken: medium
- Remediation class: strengthen proof - Keep the current statement but spell out that the bound is conditioned on the issued safe action and fixed drift model.
- Code anchors:
  - src/orius/universal_theory/battery_instantiation.py:127 (`certificate_expiration_bound`) - Exact implementation of the analytical lower bound.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:33 (`test_certificate_validity_horizon_matches_expiration_bound_for_zero_action`) - Checks the bound against the validity-horizon helper on the zero-dispatch witness.

### T7: Feasible fallback existence

- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1848
- Proof location: appendices/app_c_full_proofs.tex:231
- Assumptions used: ['A1', 'A3', 'A4', 'A8', 'Current state is interior by at least the one-step model-error margin.']
- Unresolved assumptions: ['Current state is interior by at least the one-step model-error margin.']
- Dependencies: ['Constructive zero-dispatch battery fallback']
- Weakest step: The theorem statement says any currently safe state, but both the appendix proof and the code only support interior states with slack against model error.
- Rigor rating: broken
- Code correspondence: partial - The battery helper validates zero dispatch only for states that remain safe after adding and subtracting the model-error margin. That is strictly narrower than the theorem statement.
- Severity if broken: high
- Remediation class: narrow statement - Replace 'any safe state' with an interior-state condition that matches the implemented fallback validator and the appendix restatement.
- Code anchors:
  - src/orius/universal_theory/battery_instantiation.py:150 (`zero_dispatch_fallback`) - Constructive battery fallback.
  - src/orius/universal_theory/battery_instantiation.py:155 (`validate_battery_fallback`) - Checks fallback preservation only over the interior slack interval.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:54 (`test_certify_fallback_existence_passes_for_interior_soc`) - Positive witness explicitly uses an interior SOC state.
  - tests/test_dc3s_temporal_theorems.py:64 (`test_certify_fallback_existence_fails_near_boundary`) - Negative witness exposes the over-broad theorem statement.

### T8: Graceful degradation dominance

- Statement location: chapters_merged/ch04_theoretical_foundations.tex:1874
- Proof location: appendices/app_c_full_proofs.tex:244
- Assumptions used: ['A1', 'A2', 'A3', 'A4', 'A5', 'A8', 'The graceful landing regime makes the safe interior absorbing after landing.']
- Unresolved assumptions: ['The graceful landing regime makes the safe interior absorbing after landing.']
- Dependencies: ['Theorem T4 witness for the uncontrolled comparator', 'Graceful landing absorbs future excursions after the landing phase']
- Weakest step: The strict-improvement claim depends on an absorbing safe-landing regime that is asserted by chapter narrative rather than discharged as a theorem-grade dynamics argument.
- Rigor rating: has-a-hole
- Code correspondence: partial - The helper compares two violation sequences and can detect strict domination, but it does not prove that the implemented graceful controller always generates the favorable sequence under the same admissible faults.
- Severity if broken: medium
- Remediation class: strengthen proof - Either prove the absorbing-landing argument formally or narrow the theorem to the comparison helper's sequence-level claim.
- Code anchors:
  - src/orius/universal_theory/battery_instantiation.py:182 (`evaluate_graceful_degradation_dominance`) - Sequence-level dominance checker.
- Test anchors:
  - tests/test_dc3s_temporal_theorems.py:73 (`test_graceful_degradation_dominance_detects_strict_domination`) - Checks the comparison helper only.

### T9: Universal Impossibility, T9

- Statement location: chapters/ch37_universality_completeness.tex:109
- Proof location: appendices/app_c_full_proofs.tex:270
- Assumptions used: ['A4', "A6': phi-mixing fault process with geometric decay.", 'A positive witness constant c > 0 is available from the boundary-reachability argument.']
- Unresolved assumptions: ["A6': phi-mixing fault process with geometric decay.", 'A positive witness constant c > 0 is available from the boundary-reachability argument.']
- Dependencies: ['Theorem T4 no-free-safety witness', 'Azuma-Hoeffding concentration for separated windows', 'Window buffering argument based on phi-mixing']
- Weakest step: The proof imports both the phi-mixing assumption A6' and the witness constant c > 0 without deriving either one on the active theorem surface.
- Rigor rating: broken
- Code correspondence: partial - The helper reproduces the claimed c * d * T scaling algebra once c is supplied externally, but it does not instantiate the witness constant or the window-separation argument from runtime objects.
- Severity if broken: critical
- Remediation class: downgrade register status - Treat T9 as a proof sketch or structural conjecture until the missing phi-mixing assumption and witness constant are registered and discharged.
- Code anchors:
  - src/orius/dc3s/theoretical_guarantees.py:587 (`compute_universal_impossibility_bound`) - Encodes the claimed scaling only after c and effective horizon are handed in.
- Test anchors:
  - tests/test_theoretical_guarantees.py:18 (`test_expected_lower_bound_matches_linear_formula`) - Algebraic witness test only.
  - tests/test_theoretical_guarantees_hypothesis.py:23 (`test_t9_impossibility_bound_matches_linear_scaling`) - Property test for the helper's formula, not for the proof obligations.

### T10: Boundary-indistinguishability lower bound, T10

- Statement location: chapters/ch37_universality_completeness.tex:206
- Proof location: appendices/app_c_full_proofs.tex:310
- Assumptions used: ['Controller acts only through the degraded observation channel.', 'TV(P_0,t, P_1,t) <= w_t for the boundary subproblem.', 'Unsafe-side prior mass satisfies P(H_1,t) >= p_t.', 'The unsafe-side decision error controls true-state violations.']
- Unresolved assumptions: ['Controller acts only through the degraded observation channel.', 'TV(P_0,t, P_1,t) <= w_t for the boundary subproblem.', 'Unsafe-side prior mass satisfies P(H_1,t) >= p_t.', 'The unsafe-side decision error controls true-state violations.']
- Dependencies: ['Le Cam two-point lemma', 'Boundary-mass sequence p_t']
- Weakest step: Le Cam only lower-bounds the maximum of the two hypothesis error rates, but the proof immediately treats that as a lower bound on the unsafe-side error probability alone.
- Rigor rating: broken
- Code correspondence: partial - The helper and frontier utilities reproduce the stylized lower-bound algebra, but they accept boundary mass as an exogenous input and do not repair the invalid one-sided Le Cam step.
- Severity if broken: critical
- Remediation class: narrow statement - Retitle T10 as a stylized lower-bound construction unless a valid unsafe-side error lower bound and a defended boundary-mass model are supplied.
- Code anchors:
  - src/orius/dc3s/theoretical_guarantees.py:623 (`compute_stylized_frontier_lower_bound`) - Implements the stylized half-sum formula directly.
  - src/orius/universal/reliability_safety_frontier.py:46 (`compute_frontier`) - Chapter-facing plotting proxy for the scoped lower-bound surface.
- Test anchors:
  - tests/test_theoretical_guarantees.py:63 (`test_general_formula_matches_half_sum_p_one_minus_w`) - Formula check only.
  - tests/test_theoretical_guarantees_hypothesis.py:50 (`test_t10_frontier_lower_bound_matches_half_sum_formula`) - Property test for the stylized algebra only.

### T11: Typed structural transfer theorem, T11

- Statement location: chapters/ch37_universality_completeness.tex:338
- Proof location: appendices/app_c_full_proofs.tex:354
- Assumptions used: ['Coverage obligation for the observation-consistent state set.', 'Soundness of the tightened safe-action set.', 'Repair membership in the tightened safe-action set.', 'Fallback admissibility when the tightened set is empty.']
- Unresolved assumptions: ['Coverage obligation for the observation-consistent state set.', 'Soundness of the tightened safe-action set.', 'Repair membership in the tightened safe-action set.', 'Fallback admissibility when the tightened set is empty.']
- Dependencies: ['Definition: Universal adapter contract', 'Corollary: Episode aggregation under explicit per-step budgets', 'Proposition: Failure of any transfer obligation breaks the reference proof pattern']
- Weakest step: The forward theorem is a clean one-step transfer under four obligations, but the converse is only a proof-pattern failure argument and the supporting mini-harness still tends to overstate what is actually transferred.
- Rigor rating: has-a-hole
- Code correspondence: partial - The authoritative typed contract surface exists, but the executable checks still mix structural method checks, mini-harness invariants, and runtime consistency checks instead of discharging the four theorem obligations directly.
- Severity if broken: high
- Remediation class: narrow statement - Keep T11 as a four-obligation one-step transfer theorem, demote the five-invariant mini-harness to supporting evidence, and avoid implying that passing the mini-harness automatically yields the episode bound.
- Code anchors:
  - src/orius/universal_theory/contracts.py:514 (`ContractVerifier.validate_runtime_step`) - Authoritative typed runtime contract surface.
  - src/orius/dc3s/theoretical_guarantees.py:666 (`evaluate_structural_transfer`) - Executable four-obligation witness for the narrowed theorem statement.
  - src/orius/universal/contract.py:281 (`ContractVerifier.check`) - Supporting five-invariant reference harness only.
- Test anchors:
  - tests/test_theoretical_guarantees_hypothesis.py:76 (`test_t11_transfer_requires_all_four_obligations`) - Covers the active four-obligation theorem surface.
  - tests/test_universal_theory.py:25 (`test_contract_verifier_accepts_canonical_domain_instantiations`) - Checks the typed runtime contract on canonical adapters.
  - tests/test_universal_contract.py:75 (`test_passes_all_five_invariants`) - Supporting mini-harness test, not the full active theorem.
