# Documentation Consistency Checklist

Use this checklist before freezing ORIUS manuscript, review, or publication claims.

## 1. Freeze the canonical surfaces

Confirm these remain the active control surfaces:

- `paper/paper.tex`
- `paper/review/orius_review_dossier.tex`
- `paper/metrics_manifest.json`
- `paper/claim_matrix.csv`
- `reports/publication/orius_equal_domain_parity_matrix.csv`
- `reports/publication/orius_domain_closure_matrix.csv`
- `reports/publication/orius_universal_claim_matrix.csv`

## 2. Align the domain posture

Verify every active document says the same thing:

- battery = witness row
- AV = defended bounded row under the TTC plus predictive-entry-barrier contract
- industrial = defended bounded row
- healthcare = defended bounded row
- navigation = shadow-synthetic
- aerospace = experimental

## 3. Update targets

When the claim posture changes, update these first:

- `paper/paper.tex`
- `paper/README.md`
- `frontmatter/titlepage.tex`
- `frontmatter/abstract.tex`
- `frontmatter/acknowledgments.tex`
- `reports/publication/README.md`

## 4. Legacy-language guardrails

Run this check after editing:

```bash
PYTHONPATH=src .venv/bin/pytest --override-ini addopts='' -q tests/test_thesis_package_assets.py
```

The narrative scan in that test file is the canonical guardrail. Any
intentional legacy-language match must live only in an explicitly archived
surface, not in the active monograph control path.

## 5. Verification checks

```bash
PYTHONPATH=src .venv/bin/python scripts/build_orius_monograph_assets.py
PYTHONPATH=src .venv/bin/python scripts/validate_paper_claims.py
make orius-book
make orius-review-pack
```

## 6. Acceptance

The docs are consistent when:

- the book and reviewer package compile,
- the parity matrix, domain-closure matrix, and domain chapters agree,
- active docs use the same ORIUS expansion,
- active tracked surfaces no longer rely on article-lineage framing.
