# ORIUS Monograph Authoring Guide

This folder is the manuscript control room for the ORIUS universal-safety monograph
and its submission-facing derivatives. Use it as the authoring guide for:

- writing the canonical book-length manuscript,
- keeping manuscript claims synchronized with tracked artifacts,
- regenerating monograph tables, matrices, and reviewer surfaces,
- compiling the main book and the separate reviewer dossier.

Canonical policy:

- source of truth: `../orius_book.tex`
- mirrored internal controller: `paper/paper.tex`
- legacy archival long-form controller: `../orius_battery_409page_figures_upgraded_main.tex` (non-canonical internal archive only)
- official compiled deliverable: repo-root `paper.pdf`
- review dossier companion: `paper/review/orius_review_dossier.tex`
- flagship IEEE main draft: `paper/ieee/orius_ieee_main.tex`
- flagship IEEE appendix: `paper/ieee/orius_ieee_appendix.tex`
- legacy short derivative: `paper/paper_r1.tex`
- narrative companion: `paper/PAPER_DRAFT.md`

---

## 1. Manuscript roles

| File | Role |
|---|---|
| `../orius_book.tex` | Canonical submission monograph controller |
| `paper.tex` | Mirrored internal monograph controller retained for local chapter-relative editing |
| `ieee/orius_ieee_main.tex` | Flagship IEEE double-column working draft |
| `ieee/orius_ieee_appendix.tex` | Separate IEEE appendix / proofs / benchmark appendix |
| `paper_r1.tex` | Legacy battery-centric short derivative retained for provenance only |
| `PAPER_DRAFT.md` | Non-authoritative narrative companion |
| `../paper.pdf` | Canonical compiled monograph |
| `metrics_manifest.json` | Locked manuscript metrics surface |
| `claim_matrix.csv` | Claim-to-evidence map |
| `manifest.yaml` | Manuscript artifact manifest |
| `sync_rules.md` | Rules for keeping text and artifacts aligned |

---

## 2. What the monograph is

The canonical manuscript is a universal-first ORIUS monograph, not a stitched
article bundle. That means:

1. **clear claim boundary**
   - battery is the witness row with the deepest theorem-to-artifact closure,
   - AV is a defended bounded row under the longitudinal TTC + predictive-entry-barrier contract,
   - healthcare is a defended bounded row under the promoted MIMIC monitoring contract.

2. **book-first structure**
   - universal hazard and claim boundary,
   - runtime architecture and theorem bridge,
   - benchmark and governance protocol,
   - three defended program rows under one template,
   - cross-domain synthesis and explicit non-claims,
   - an archival source register for tracked historical/depth material,
   - appendix-heavy artifact and review traceability.

3. **artifact-grounded writing**
   - all numbers should come from locked manifests or generated tables,
   - figures/tables should be traceable back to `reports/publication/`.

4. **appendix-heavy support**
   - full proofs, reviewer analysis, protocol cards, and artifact crosswalks
     belong in appendices and must be referenced explicitly from the main text.

---

## 3. Manuscript source layout

The canonical long-form story is spread across:

- `../orius_book.tex` for the canonical submission monograph controller,
- `paper.tex` for the mirrored internal controller that shares the same chapter spine,
- `monograph/` for generator-owned universal-first chapter blocks,
- `chapters/` and `chapters_merged/` for tracked historical/depth chapters indexed for provenance,
- `longform/` for section-level extension packets retained as archive sources,
- `appendices/` for proofs, audits, and traceability,
- `review/` for the separate reviewer dossier,
- `reports/publication/` for locked result artifacts and monograph matrices.

The canonical PDF now indexes bounded archive material for provenance, but the
headline defended claim surface remains Battery + AV + Healthcare. In
particular, `../orius_battery_409page_figures_upgraded_main.tex` is still an
internal archival controller, not the submission authority, and archive text
must not be used as evidence for defended three-domain claims unless its content
is first rewritten to the same Battery + AV + Healthcare truth surface.

Important proof/evidence sources:

- `../chapters/ch16_battery_theorem_oasg_existence.tex`
- `../chapters/ch17_battery_theorem_safety_preservation.tex`
- `../appendices/app_c_full_proofs.tex`
- `../appendices/app_m_verified_theorems_and_gap_audit.tex`
- `../appendices/app_s_claim_evidence_registers.tex`
- `../reports/publication/orius_domain_closure_matrix.csv`
- `../reports/publication/orius_universal_claim_matrix.csv`
- `../reports/publication/orius_monograph_chapter_map.csv`

---

## 4. Generator-first workflow

The monograph generator is the source of truth for generated book assets.

Use:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_orius_monograph_assets.py
```

That command owns:

- `paper/monograph/`
- `paper/review/`
- `paper/bibliography/orius_monograph.bib`
- review, closure, and crosswalk matrices in `reports/publication/`

The IEEE support surfaces are generated separately via:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_orius_ieee_assets.py
```

That command owns:

- `reports/publication/orius_top_tier_benchmark_corpus.csv`
- `paper/ieee/generated/`
- `reports/editorial/orius_claim_delta_ledger.csv`
- `reports/editorial/orius_flagship_revision_ledger.csv`

Do not hand-edit generator-owned outputs unless the same change is also made in
`scripts/build_orius_monograph_assets.py`.

---

## 5. Build and verification workflow

Recommended authoring/build loop:

1. update curated static sources such as frontmatter or non-generated depth chapters,
2. regenerate generator-owned monograph assets,
3. run claim/manuscript verification,
4. compile the book and reviewer dossier,
5. only then promote wording into final manuscript-facing prose.

Primary commands:

| Command | Purpose |
|---|---|
| `make orius-monograph-assets` | Regenerate monograph chapters, bibliography, review package, and publication matrices |
| `make paper-verify` | Run manuscript and claim validation |
| `make paper-compile` | Compile the senior-review single-flow monograph from `orius_book.tex` to `paper/paper.pdf` and repo-root `paper.pdf` |
| `make review-compile` | Compile the reviewer dossier PDF |
| `make orius-book` | Verify and compile the main monograph |
| `make orius-review-pack` | Build the reviewer dossier companion |
| `python scripts/validate_paper_claims.py` | Validate claim-to-artifact mappings |
| `make publish-audit` | Run the final submission audit |

---

## 6. Definition of done

The monograph is in a strong final state when:

- the main story is universal-first and three-domain hard-cut,
- every headline result is backed by a tracked artifact, theorem surface, or cited source,
- the bibliography, reviewer package, and publication matrices rebuild cleanly,
- claim validation passes,
- the compiled book remains above the minimum page threshold,
- the reviewer dossier compiles separately,
- no active tracked manuscript/doc/report surface depends on legacy article-lineage framing.
