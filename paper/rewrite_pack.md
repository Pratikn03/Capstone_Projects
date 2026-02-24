# GridPulse Full Thesis Rewrite Pack (Markdown Master)

## How to Use This Pack
Use this file as the write/edit blueprint for `paper/PAPER_DRAFT.md`.
Each section below gives:
- Required facts to include.
- Required tables/figures to include.
- Optional detail you can add if space permits.
- Claims that require external citation before publication.

Canonical values and run IDs come from `paper/metrics_manifest.json`.

---

## Final Heading Tree (Decision-Complete)

1. Title Page
2. Abstract
3. Keywords
4. Introduction
5. System Overview
6. Data Assets and Scope
7. Methods
8. Experimental Protocol
9. Results
10. Metrics Source of Truth and Run IDs
11. Claim Traceability
12. Discussion
13. Threats to Validity
14. Operational Safety and Failure Modes
15. Limitations
16. Future Work
17. Conclusion
18. References
19. Appendices

### Recommended Appendix Tree
A. Replication Checklist
B. Artifact Inventory
C. Additional Forecast Diagnostics
D. Additional Robustness Diagnostics
E. Governance and Compliance Notes

---

## Section-by-Section Content Bank

## 1) Title Page
### Required facts
- Project name: GridPulse.
- Problem framing: forecasting + carbon-aware battery dispatch.
- Author and affiliation.
- Manuscript date.

### Required tables/figures
- None.

### Optional details
- Version tag for manuscript (e.g., "Dataset-scoped latest lock: 2026-02-17").

### Needs citation
- None.

## 2) Abstract
### Required facts
- One-line system definition: end-to-end forecast-to-dispatch pipeline.
- Explicit dataset-scoped metric policy.
- DE impact values: 7.11%, 0.30%, 6.13%.
- US impact values: 0.11%, 0.13%, 0.00%.
- DE stochastic values (run `20260217_165756`): EVPI_robust 2.32, EVPI_det -30.40, VSS 2,708.61.
- US stochastic values (run `20260217_182305`): EVPI_robust 10,279,851.74, EVPI_det 24,915,503.93, VSS 297,092.71.
- One limitation sentence about cross-region value variability.

### Required tables/figures
- None in abstract.

### Optional details
- Mention operational reproducibility and run governance.

### Needs citation
- None.

## 3) Keywords
### Required facts
- Include ML, energy forecasting, robust dispatch, conformal uncertainty, battery optimization.

### Required tables/figures
- None.

### Optional details
- Add "stochastic programming" and "MLOps monitoring".

### Needs citation
- None.

## 4) Introduction
### Required facts
- Why prediction-only systems are insufficient for grid operations.
- Decision loop framing: Forecast -> Optimize -> Dispatch -> Measure -> Monitor.
- Explicit statement that manuscript is evidence-locked to repository artifacts.

### Required tables/figures
- Optional architecture figure later; no mandatory table here.

### Optional details
- Competitive/operational framing and reproducibility demands.

### Needs citation
- Background claims about renewable variability and dispatch practice.

## 5) System Overview
### Required facts
- Main stack components:
  - Data pipeline.
  - Forecasting models (GBM, LSTM, TCN).
  - Conformal/FACI uncertainty.
  - Deterministic and robust dispatch.
  - Monitoring and retraining hooks.
  - API/dashboard serving.

### Required tables/figures
- One architecture diagram (source: `reports/figures/architecture.png` or `reports/figures/architecture.svg`).

### Optional details
- Module-to-module artifact handoffs.

### Needs citation
- None if architecture description is internal.

## 6) Data Assets and Scope
### Required facts
- DE profile lock:
  - Rows 17,377; columns 98; features 94; range 2018-10-07 to 2020-09-30 UTC.
- US profile lock:
  - Rows 13,638; columns 118; features 114; range 2024-07-01 to 2026-01-20 UTC.
- Clarify distinction between raw columns and engineered features.
- Clarify why legacy 92,382-row claims are out-of-scope for this locked manuscript snapshot.

### Required tables/figures
- Dataset profile table (build from `data/dashboard/de_stats.json` and `data/dashboard/us_stats.json`).

### Optional details
- Target summary stats from dashboard stats files.

### Needs citation
- External provider descriptions only (OPSD/EIA references).

## 7) Methods
### Required facts
- Forecasting objective and horizon setup.
- Uncertainty method (split conformal + FACI behavior).
- Optimization objective components: cost, carbon, degradation, constraints.
- Robust formulation intent and stochastic evaluation definitions (EVPI, VSS).

### Required tables/figures
- Optional equations block.
- Optional optimization variable/constraint table.

### Optional details
- Feature groups and rationale.
- Deployment-oriented implementation details.

### Needs citation
- Method-origin literature (conformal, robust optimization, model classes).

## 8) Experimental Protocol
### Required facts
- Time-aware split usage and evaluation process.
- Evaluation outputs are sourced from frozen artifacts, not ad hoc notebook numbers.
- Explicit run IDs for stochastic metrics by dataset.

### Required tables/figures
- Protocol summary table with artifacts and paths.

### Optional details
- Hardware/runtime context.

### Needs citation
- None for internal protocol.

## 9) Results
### Required facts
- Keep this section split into 5.1-5.5 style blocks:
  1. Forecast quality (choose one canonical source family; recommended dashboard metrics JSON for profile consistency).
  2. Uncertainty coverage (target-wise).
  3. Decision impact (DE and US impact summaries).
  4. Stochastic value (EVPI/VSS with dataset-scoped run IDs).
  5. Robustness behavior (infeasibility and regret trends; clearly label dataset scope).

### Required tables/figures
- Forecast metrics table.
- Uncertainty table.
- Decision impact table (DE + US, not DE-only legacy mixed table).
- Stochastic metrics table with two run IDs.
- At least one dispatch/robustness figure.

### Optional details
- Region-by-region interpretation paragraph.

### Needs citation
- None for repository-derived numeric outcomes.

## 10) Metrics Source of Truth and Run IDs
### Required facts
- State exact policy:
  - DE impact from `reports/impact_summary.csv`.
  - US impact from `reports/eia930/impact_summary.csv`.
  - DE stochastic from `reports/research_metrics_de.csv`, run `20260217_165756`.
  - US stochastic from `reports/research_metrics_us.csv`, run `20260217_182305`.
- Explain that this is dataset-scoped latest, not single common-run freezing.

### Required tables/figures
- One compact "metrics provenance" table with file, row filter, run ID, timestamp.

### Optional details
- Include manifest hash/checksum for tamper-evidence.

### Needs citation
- None.

## 11) Claim Traceability
### Required facts
- Link manuscript claims to `paper/claim_matrix.csv` IDs.
- Define statuses: Verified, Conflicting, Unsupported, Needs Citation.
- Explain publication rule: only Verified claims remain in final release manuscript.

### Required tables/figures
- Short claim-status summary table.

### Optional details
- Add counts by status and section.

### Needs citation
- None.

## 12) Discussion
### Required facts
- Why cross-region decision value differs despite same framework.
- What is robustly supported vs what remains uncertain.
- Practical interpretation of positive VSS in both regions under locked runs.

### Required tables/figures
- Optional figure: cost/carbon tradeoff.

### Optional details
- Model-family tradeoffs (GBM vs DL) grounded in your reported metrics.

### Needs citation
- Any broad claim about universal model superiority beyond your datasets.

## 13) Threats to Validity
### Required facts
- Internal validity: metric source drift risk and mitigation via manifest.
- External validity: region scope limits.
- Construct validity: operational proxies (price/carbon assumptions).
- Conclusion validity: sensitivity to run choice and scenario design.

### Required tables/figures
- Optional validity matrix.

### Optional details
- Explicitly enumerate rejected legacy claims.

### Needs citation
- If citing standard validity taxonomy, cite methodology source.

## 14) Operational Safety and Failure Modes
### Required facts
- Failure modes:
  - stale models,
  - drift,
  - missing signals,
  - infeasible optimization edge cases,
  - API degradation.
- Safety controls:
  - health/readiness,
  - drift monitoring,
  - fallback/rollback strategy,
  - human review thresholds.

### Required tables/figures
- Failure mode table: trigger, detection, mitigation, fallback.

### Optional details
- SLA/SLO notes if you include deployment claims.

### Needs citation
- Regulatory claims if mentioning specific policy requirements.

## 15) Limitations
### Required facts
- Limited geographic scope.
- Dependence on available price/carbon proxies.
- Dataset/profile lock may differ from legacy publication tables.
- DOCX sync currently constrained by environment tooling.

### Required tables/figures
- None required.

### Optional details
- Explicit distinction between technical and evidence limitations.

### Needs citation
- Only if external claims are introduced.

## 16) Future Work
### Required facts
- Ranked roadmap:
  1. unify profile and forecast metric generation pipeline,
  2. stronger stochastic scenario design,
  3. automated docx sync pipeline,
  4. expanded regional validation.

### Required tables/figures
- Optional roadmap table.

### Optional details
- Milestones with measurable acceptance gates.

### Needs citation
- None for internal roadmap.

## 17) Conclusion
### Required facts
- Re-state end-to-end contribution.
- Re-state locked DE/US impact and stochastic values.
- Re-state governance contribution: run-scoped traceability and claim validation.

### Required tables/figures
- None.

### Optional details
- One sentence on production-readiness posture.

### Needs citation
- None.

## 18) References
### Required facts
- Include all cited methods and external facts.
- Remove uncited placeholders.

### Required tables/figures
- None.

### Optional details
- Group by methodological domain.

### Needs citation
- This section is itself citation infrastructure.

## 19) Appendices

## Appendix A: Replication Checklist
### Required facts
- Commands, expected outputs, artifact paths.
- Run-ID pinning and validation script command.

### Required tables/figures
- Replication checklist table.

### Optional details
- Runtime/hardware notes.

### Needs citation
- None.

## Appendix B: Artifact Inventory
### Required facts
- File-by-file inventory of data/metrics/report artifacts used in claims.

### Required tables/figures
- Artifact inventory table with path and role.

### Optional details
- SHA256 hashes.

### Needs citation
- None.

## Appendix C: Additional Forecast Diagnostics
### Required facts
- Non-primary diagnostics that support but do not redefine canonical claims.

### Required tables/figures
- Supplemental diagnostic plots/tables.

### Optional details
- Per-target residual analyses.

### Needs citation
- None.

## Appendix D: Additional Robustness Diagnostics
### Required facts
- Stress-test scenarios, regret behavior, infeasibility behavior.

### Required tables/figures
- Robustness supplemental table(s).

### Optional details
- Region-specific perturbation commentary.

### Needs citation
- None.

## Appendix E: Governance and Compliance Notes
### Required facts
- Claim validation process.
- Sync workflow controls.
- Publication release gates.

### Required tables/figures
- Governance gate checklist.

### Optional details
- Who approved metric lock and when.

### Needs citation
- Regulatory citations if policy statements are included.

---

## Canonical Claim Snippets (Copy-Ready)
Use these exact strings in manuscript body to avoid drift:

- "DE cost savings: 7.11%; DE carbon reduction: 0.30%; DE peak shaving: 6.13%."
- "US cost savings: 0.11%; US carbon reduction: 0.13%; US peak shaving: 0.00%."
- "DE stochastic run: 20260217_165756 (EVPI_robust 2.32; EVPI_deterministic -30.40; VSS 2,708.61)."
- "US stochastic run: 20260217_182305 (EVPI_robust 10,279,851.74; EVPI_deterministic 24,915,503.93; VSS 297,092.71)."
- "Dataset-scoped latest policy is used; this manuscript is not a single common-run freeze."

---

## Pre-Submission Editorial Gate
Before any submission export, confirm:
1. `python scripts/validate_paper_claims.py` passes.
2. `paper/paper.tex` has no missing `\input{}` files.
3. No placeholder tokens remain.
4. All external claims in `Needs Citation` list are cited or removed.
5. `paper/PAPER_DRAFT.md`, `paper/paper.tex`, and release DOCX share the same title, abstract metrics, and conclusion metrics.
