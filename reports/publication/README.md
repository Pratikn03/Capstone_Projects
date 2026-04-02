# ORIUS Publication Artifact Surfaces

This directory contains the active tracked publication artifacts that support the
universal-first ORIUS monograph.

It also anchors the bounded-universal closure program: the repo-level target is
one witness row plus five defended bounded rows under a shared runtime,
benchmark, governance, and provenance gate. Current blockers remain explicit in
the parity and closure matrices rather than being widened by prose.

## Canonical monograph matrices

- `reports/publication/orius_equal_domain_parity_matrix.csv`
- `reports/publication/orius_domain_closure_matrix.csv`
- `reports/publication/orius_universal_claim_matrix.csv`
- `reports/publication/orius_monograph_chapter_map.csv`
- `reports/publication/orius_literature_matrix.csv`
- `reports/publication/orius_framework_gap_matrix.csv`
- `reports/publication/orius_maturity_matrix.csv`
- `reports/publication/orius_cross_domain_design_principles.csv`
- `reports/publication/orius_module_claim_crosswalk.csv`
- `reports/publication/orius_publication_artifact_index.csv`

## Live closure status

- `reports/real_data_contract_status.json`
- `docs/BOUNDED_UNIVERSAL_CLOSURE_PROGRAM.md`

## Review and bibliography support

- `reports/publication/orius_annotated_bibliography.csv`
- `reports/publication/orius_reviewer_scorecards.csv`
- `reports/publication/orius_review_global_gap_matrix.csv`
- `reports/publication/orius_artifact_appendix.md`
- `reports/publication/orius_review_dossier.pdf`

## Build note

Regenerate the generator-owned publication surfaces with:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_orius_monograph_assets.py
```

## Archive policy

Older program-era audit files and frozen package bundles are retained only for
historical provenance. They are not part of the active monograph control
surface. See:

- `reports/legacy_archive/README.md`
