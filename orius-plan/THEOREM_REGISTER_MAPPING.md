# Canonical Theorem Register and Cross-Source Mapping

This document defines the canonical battery-8 theorem register and maps artifact zip and paper theorem labels to it. Use this file when aligning the artifact zip, thesis, and paper.

## Canonical Register (T1–T8)

Source of truth: thesis Appendix M (`appendices/app_m_verified_theorems_and_gap_audit.tex`).

| ID | Name | One-line statement |
|----|------|--------------------|
| T1 | OASG Existence | There exist scenarios where an action appears safe in observed state but is unsafe in true physical state. |
| T2 | One-Step Safety Preservation | If the repaired action lies inside the tightened safe set and the current state lies inside the inflated certificate, then the next battery state remains safe. |
| T3 | ORIUS Core Bound | The expected battery violation count satisfies $\mathbb{E}[V] \le \alpha(1-\bar{w})T$. |
| T4 | No Free Safety | Any controller maintaining zero true-state violations must accept at least some intervention rate under admissible fault sequences. |
| T5 | Certificate Validity Horizon | A battery certificate remains valid for the largest integer horizon whose forward tube remains inside the battery SOC bounds. |
| T6 | Certificate Expiration Bound | The battery certificate horizon is lower-bounded by the issuance-time distance to the nearest SOC boundary divided by the drift scale. |
| T7 | Feasible Fallback Existence | There exists a battery fallback action that preserves safety from an interior SOC state under bounded model error. |
| T8 | Graceful Degradation Dominance | The battery graceful-degradation controller incurs no more violations than the uncontrolled controller under the same admissible fault sequence. |

## Artifact Zip Mapping

The artifact zip `ORIUS_COMPLETE_MATHEMATICAL_PROOFS.md` uses different theorem names. Map as follows:

| Zip theorem | Canonical ID | Notes |
|-------------|--------------|-------|
| T1 OASG Illusion | T1 | Aligned |
| T2 Certificate Half-Life Bound | T5, T6 | Partial; zip T2 spans both certificate validity and expiration |
| T3 Conformal Reachability Coverage | T3 | Aligned |
| T4 Graceful Degradation Under Sensor Loss | T8 | Aligned |
| T5 Multi-Domain Transfer Bound | — | Out-of-scope for battery-8; multi-domain extension |
| T6 Computational Complexity of DC3S | — | Out-of-scope for battery-8; multi-domain extension |
| T7 Optimal Control Under Uncertainty | — | Out-of-scope for battery-8; multi-domain extension |
| T8 System-Level Safety Convergence | — | Out-of-scope for battery-8; multi-domain extension |

**Note:** Zip T5–T8 are multi-domain extensions. The battery-8 register (T1–T8 above) is the authoritative set for thesis/paper alignment.

## Paper Mapping

The paper (`paper/paper.tex`) uses LaTeX labels. Map as follows:

| Paper label | Canonical ID | Notes |
|-------------|--------------|-------|
| thm:rac_coverage (Marginal coverage preservation) | T3 | ORIUS Core Bound |
| thm:safety_margin (Safety-margin monotonicity) | T2 | One-Step Safety Preservation |
| cor:zero_violation (Zero-violation sufficiency) | T2 | Corollary of safety-margin theorem |
| thm:conditional_coverage (Group-conditional Mondrian) | auxiliary | Diagnostic tool; not a core battery theorem |

## Evidence and Traceability

- **Code anchors:** `src/orius/dc3s/`, `src/orius/cpsbench_iot/`
- **Locked evidence:** `reports/publication/dc3s_main_table_ci.csv`, `reports/publication/graceful_degradation_trace.csv`, etc.
- **Full map:** `orius-plan/theorem_to_evidence_map.md`
