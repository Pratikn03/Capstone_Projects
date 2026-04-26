# ORIUS Publication Artifact Surfaces

This directory contains the active tracked publication artifacts that support the
current ORIUS three-domain monograph and review package.

The active program is literal and closed over three rows only:

- Battery = witness row
- AV = defended bounded row
- Healthcare = defended bounded row

The flagship novelty sentence for the promoted ML lane is:

> ORIUS identifies OASG as the degraded-observation release hazard and provides a reliability-aware runtime safety layer across Battery, AV, and Healthcare.

The novelty claim is architectural and runtime-semantic: ORIUS does not claim a new
conformal method, a new robust-optimization primitive, a new universal controller, or a
new conditional-coverage theorem.

The promoted ML center is grouped calibration and runtime safety under degraded observation. This directory does not carry a new conditional-coverage theorem claim.

## Canonical monograph matrices

- `reports/publication/orius_domain_closure_matrix.csv`
- `reports/publication/orius_universal_claim_matrix.csv`
- `reports/publication/orius_monograph_chapter_map.csv`
- `reports/publication/orius_literature_matrix.csv`
- `reports/publication/orius_framework_gap_matrix.csv`
- `reports/publication/orius_maturity_matrix.csv`
- `reports/publication/orius_cross_domain_design_principles.csv`
- `reports/publication/orius_module_claim_crosswalk.csv`
- `reports/publication/orius_publication_artifact_index.csv`

## Canonical ML / Novelty bundle

- `reports/publication/three_domain_ml_benchmark.csv`
- `reports/publication/three_domain_ml_benchmark_summary.json`
- `reports/publication/three_domain_reliability_calibration.csv`
- `reports/publication/three_domain_grouped_coverage.csv`
- `reports/publication/three_domain_grouped_width.csv`
- `reports/publication/three_domain_nonvacuity_checks.json`
- `reports/publication/three_domain_baseline_suite.csv`
- `reports/publication/three_domain_ablation_matrix.csv`
- `reports/publication/three_domain_negative_controls.csv`
- `reports/publication/novelty_separation_matrix.csv`
- `reports/publication/what_orius_is_not_matrix.csv`

The benchmark and grouped-calibration files are the flagship ML evidence bundle. The baseline and ablation tables are explicitly diagnostic cross-domain proxy surfaces and must not be read as battery-witness replacements.

## Live closure status

- `reports/real_data_contract_status.json`
- `docs/BOUNDED_UNIVERSAL_CLOSURE_PROGRAM.md`

## Review and bibliography support

- `reports/publication/orius_annotated_bibliography.csv`
- `reports/publication/orius_reviewer_scorecards.csv`
- `reports/publication/orius_review_global_gap_matrix.csv`
- `reports/publication/orius_artifact_appendix.md`
- `appendices/app_c_flagship_proofs.tex`
- `appendices/app_c_flagship_proofs.pdf`
- `reports/publication/phase3_flagship_v1_proof_book.md`
- `reports/publication/phase3_flagship_v1_proof_book.pdf`
- `reports/publication/final_freeze_release_note.md`
- `reports/publication/orius_review_dossier.pdf`

The bibliography corpora may still mention broader CPS application families as
external prior work. Those references are background context only and are not
active ORIUS defended-domain rows.

The Phase 3 proof book is a repo-normalized implementation book for the
flagship proof rewrite. It is a planning surface, not a theorem-authority
surface.

The standalone Appendix C flagship proof draft is the execution-facing LaTeX
rewrite surface. It is intentionally separate from the current shared manuscript
proof appendix while the monograph include path remains unchanged.

## Build note

Regenerate the generator-owned publication surfaces with:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_orius_monograph_assets.py
```

## Archive policy

Older program-era audit files are historical provenance only. They are not part
of the active monograph control surface, and the preserved PDF inventory is
intentionally limited to the canonical monograph, the canonical review dossier,
and the retained IEEE diagnostic manuscripts. See:

- `reports/legacy_archive/README.md`
