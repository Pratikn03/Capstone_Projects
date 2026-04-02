# ORIUS Monograph Authoring Guide

This folder is the manuscript control room for the ORIUS universal-safety monograph
and its submission-facing derivatives. Use it as the authoring guide for:

- writing the canonical book-length manuscript,
- keeping manuscript claims synchronized with tracked artifacts,
- regenerating monograph tables, matrices, and reviewer surfaces,
- compiling the main book and the separate reviewer dossier.

Canonical policy:

- source of truth: `paper/paper.tex`
- official compiled deliverable: repo-root `paper.pdf`
- review dossier companion: `paper/review/orius_review_dossier.tex`
- shorter derivative: `paper/paper_r1.tex`
- narrative companion: `paper/PAPER_DRAFT.md`

---

## 1. Manuscript roles

| File | Role |
|---|---|
| `paper.tex` | Canonical book-length manuscript |
| `paper_r1.tex` | Shorter derivative / submission variant |
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
   - AV, industrial, and healthcare are defended bounded rows under the current replay harness,
   - navigation is shadow-synthetic evidence,
   - aerospace is experimental.

2. **book-first structure**
   - universal hazard and claim boundary,
   - runtime architecture and theorem bridge,
   - benchmark and governance protocol,
   - six domain chapters under one template,
   - cross-domain synthesis and explicit non-claims,
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

- `paper.tex` for the compiled monograph controller,
- `monograph/` for generator-owned universal-first chapter blocks,
- `chapters/` for curated depth chapters that remain outside the generator,
- `appendices/` for proofs, audits, and traceability,
- `review/` for the separate reviewer dossier,
- `reports/publication/` for locked result artifacts and monograph matrices.

Older article-lineage or archive-style artifacts may still exist in the repo for
historical provenance, but they are not the reader-facing control surface of the
book.

Important proof/evidence sources:

- `../chapters/ch16_battery_theorem_oasg_existence.tex`
- `../chapters/ch17_battery_theorem_safety_preservation.tex`
- `../appendices/app_c_full_proofs.tex`
- `../appendices/app_m_verified_theorems_and_gap_audit.tex`
- `../appendices/app_s_claim_evidence_registers.tex`
- `../reports/publication/orius_equal_domain_parity_matrix.csv`
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
- parity, review, and crosswalk matrices in `reports/publication/`

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
| `make paper-compile` | Compile the canonical monograph to `paper/paper.pdf` and repo-root `paper.pdf` |
| `make review-compile` | Compile the reviewer dossier PDF |
| `make orius-book` | Verify and compile the main monograph |
| `make orius-review-pack` | Build the reviewer dossier companion |
| `python scripts/validate_paper_claims.py` | Validate claim-to-artifact mappings |
| `make publish-audit` | Run the final submission audit |

---

## 6. Definition of done

The monograph is in a strong final state when:

- the main story is universal-first and parity-gated,
- every headline result is backed by a tracked artifact, theorem surface, or cited source,
- the bibliography, reviewer package, and publication matrices rebuild cleanly,
- claim validation passes,
- the compiled book remains above the minimum page threshold,
- the reviewer dossier compiles separately,
- no active tracked manuscript/doc/report surface depends on legacy article-lineage framing.
