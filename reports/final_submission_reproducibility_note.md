# Final Submission Reproducibility Note

The ORIUS final submission package is governed by tracked manuscript and artifact
locks rather than by ad hoc draft values.

## Canonical authority

- manuscript authority: `paper/paper.tex`
- narrative companion: `paper/PAPER_DRAFT.md`
- metric lock: `paper/metrics_manifest.json`
- claim register: `paper/claim_matrix.csv`
- release provenance root: `reports/publication/release_manifest.json`

## Verification path

The final package should be reproducible through the repo’s non-mutating
verification and build workflow:

```bash
python scripts/validate_paper_claims.py
make thesis-manuscript
make publish-audit
```

The manuscript claim validator ensures that canonical values, manuscript
authority, and tracked release artifacts remain aligned. The build path then
compiles the canonical PDF and the publish audit checks release-facing artifact
consistency.

## Evidence boundary

All final submission claims are tied to tracked publication artifacts. Ignored
local caches, dashboard snapshots, and untracked generated files are not part
of the authoritative evidence surface.
