# ORIUS Claim Ledger — Three-Bucket Discipline

> Generated from frozen evidence.
> Every claim in the thesis and IEEE paper must trace to one of these three buckets.
> If a statement cannot be traced to a locked artifact, it must be rewritten or deleted.

## Freeze Reference

- **Evidence Freeze Certificate**: `reports/battery_av/overall/evidence_freeze_certificate.json`
- **Battery Manifest**: `reports/battery_av/battery/artifact_manifest.json` (28 artifacts, 28/28 SHA256 match)
- **AV Manifest**: `reports/orius_av/full_corpus/artifact_manifest.json` (40 artifacts, 40/40 SHA256 match)
- **Release Summary**: `reports/battery_av/overall/release_summary.json`
- **Closure Override**: `reports/battery_av/overall/publication_closure_override.json`

---

## Bucket A — Fully Evidenced (Locked Artifact ↔ Claim)

These claims have a direct, traceable link to a frozen artifact with verified SHA256 hash.

| ID | Claim | Artifact Source | Domain |
|----|-------|----------------|--------|
| A1 | ORIUS achieves TSVR = 0 on Battery across all 4 controllers, 3 fault families, 288 steps | `reports/battery_av/battery/runtime_summary.csv`, `release_summary.json` | Battery |
| A2 | Battery CertOS chain is valid: 288 certificates, 0 failures, 100% audit completeness | `reports/battery_av/battery/certos_verification_summary.json` | Battery |
| A3 | Battery has 20 counterexamples and 0 OASG cases | `reports/battery_av/battery/observed_true_counterexamples.csv` | Battery |
| A4 | AV achieves 9,348 runtime traces across 228 scenarios from 7 Waymo shards | `reports/orius_av/full_corpus/runtime_traces.csv`, `release_summary.json` | AV |
| A5 | AV CertOS chain is valid: 9,348 certificates, 0 failures, 100% audit completeness | `reports/orius_av/full_corpus/certos_verification_summary.json` | AV |
| A6 | AV has 20 counterexamples and 480 OASG cases | `reports/orius_av/full_corpus/oasg_domain_summary.csv` | AV |
| A7 | AV TSVR reduced from 0.660 (baseline) to 0.628 (ORIUS), 4.8% relative improvement | `reports/orius_av/full_corpus/runtime_summary.csv` | AV |
| A8 | AV intervention rate is 90.2% | `reports/orius_av/full_corpus/runtime_summary.csv` | AV |
| A9 | Shift-aware mechanism: ego_speed validity = 0.885 (nominal), relative_gap validity = 0.888 (nominal) | `reports/orius_av/full_corpus/shift_aware_adaptive_summary.json` | AV |
| A10 | Shift-aware mechanism: battery domain reports 0 shift alerts, validity = 1.0 | `reports/battery_av/battery/shift_aware_adaptive_summary.json` | Battery |
| A11 | LP baseline achieves TSVR 3.9% (DE) and 4.2% (US) under drift_combo — confirms T1 empirically | `reports/battery_av/battery/runtime_summary.csv` | Battery |
| A12 | DC3S achieves TSVR = 0.0% in both markets — confirms T2, T3 empirically | `reports/battery_av/battery/runtime_summary.csv` | Battery |
| A13 | Both domains validated with the same kernel binary, same contract stack, same CertOS chain | `release_summary.json`, both `artifact_manifest.json` | Cross-domain |
| A14 | 68 artifacts total, all SHA256-verified, zero mismatches, zero missing | `evidence_freeze_certificate.json` | Cross-domain |
| A15 | Battery tier = reference (witness), AV tier = proof_validated (bounded contract) | `publication_closure_override.json` | Cross-domain |

## Bucket B — Bounded / Qualified Claims

These claims are supported by evidence but carry caveats. Every instance in prose must include the stated qualification.

| ID | Claim | Qualification | Where Qualified |
|----|-------|--------------|----------------|
| B1 | ORIUS kernel is domain-agnostic via typed adapter | Proven only for 2 domains (Battery, AV). Other domains require future validation. | `detailed_limitations.tex` §L1 |
| B2 | AV TSVR improvement (0.660 → 0.628) | Residual TSVR floor at 0.628 — complex multi-agent interactions at 4s horizon cause true states outside widened interval. Not zero. | `detailed_limitations.tex` §L2 |
| B3 | Conformal coverage guarantee holds | Marginal coverage only, not conditional per-input. Shift-aware mechanism mitigates but does not eliminate subgroup gaps. | `detailed_limitations.tex` §L3 |
| B4 | Fallback action is always safe (A4) | Holds for Battery (zero dispatch) and AV (max braking). Non-trivial in coupled-dynamics domains. | `detailed_limitations.tex` §L4 |
| B5 | Shift-aware mechanism detects distribution shift | Bounded to drift rate within adaptation window (A6). Adversarial/sudden-onset shift not tested. | `detailed_limitations.tex` §L7 |
| B6 | 11 theorems proven with formal proofs | Proofs assume A1–A8. Empirical confirmation restricted to Battery + AV. T9 (Universal Impossibility) is purely formal. | `detailed_appendix.tex`, `detailed_theory.tex` |
| B7 | AV validated under bounded longitudinal contract (TTC + predictive-entry-barrier) | Not a claim of full autonomous-driving closure. Lateral control, lane-change, and multi-modal prediction excluded. | `publication_closure_override.json` |
| B8 | Adapter interface enables domain transfer (T11) | Structural transfer proven; functional equivalence across domains not demonstrated beyond Battery + AV. | `detailed_limitations.tex` §L1 |

## Bucket C — Not Claimed (Explicitly Excluded)

These items are **not** claimed in the submission. Any prose that implies them must be removed.

| ID | What is NOT Claimed | Why | Status |
|----|-------------------|-----|--------|
| C1 | Industrial domain is proof_validated | No locked pipeline artifacts in `reports/`. Only architectural instantiation exists. | Remove from README domain matrix |
| C2 | Healthcare domain is proof_validated | No locked pipeline artifacts in `reports/`. Only architectural instantiation exists. | Remove from README domain matrix |
| C3 | Navigation domain is validated | shadow_synthetic tier only; KITTI runtime missing. | Already gated in closure matrix |
| C4 | Aerospace domain is validated | experimental tier only; real-flight runtime missing. | Already gated in closure matrix |
| C5 | Real-time deployment readiness | AV averages ~350ms/step; 10Hz requires <100ms. Explicit limitation. | `detailed_limitations.tex` §L6 |
| C6 | Adversarial robustness | Only fault-injection families tested. Intentional adversarial attacks not covered. | `detailed_limitations.tex` §FW7 |
| C7 | Cross-sensor fusion | Single-sensor pipeline only. Lidar–camera fusion not implemented. | `detailed_limitations.tex` §L5 |
| C8 | Conditional (per-input) conformal guarantees | Only marginal coverage proven. | `detailed_limitations.tex` §L3 |
| C9 | Equal maturity across all 6 domains | Battery = reference, AV = bounded proof-validated. Others are not validated. | Claim ledger, README, intro |

---

## Audit Rules

1. **Every number in the thesis/paper must trace to a Bucket A artifact path.**
2. **Every qualified claim must include its Bucket B qualification in the same paragraph or a forward reference to Limitations.**
3. **No statement anywhere may imply a Bucket C item.** If found, rewrite or delete.
4. **Battery is always "reference witness row"** — not the primary validation.
5. **AV is always "bounded proof-validated row"** — never "full AV closure."
6. **The remaining 4 domains appear only in Future Work**, never in Results or Contributions.
