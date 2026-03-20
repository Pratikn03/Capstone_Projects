# Paper Sync Rules (LaTeX -> Markdown -> DOCX)

## Purpose
Define a deterministic synchronization contract so manuscript formats do not drift.

## Authority Model
1. **Authoritative source:** `paper/paper.tex`.
2. Narrative companion: `paper/PAPER_DRAFT.md`.
3. Final distribution formats: repo-root `paper.pdf` and `paper/paper.docx`.

If files disagree, `paper/paper.tex` wins.

## Metric and Claim Lock
All numeric and run-ID claims must be sourced from:
- `paper/metrics_manifest.json`
- `paper/claim_matrix.csv`

No claim may bypass the manifest lock.

## Sync Sequence
1. Update `paper/paper.tex`.
2. Run claim validator:
```bash
python scripts/validate_paper_claims.py
```
3. Compile LaTeX and publish repo-root `paper.pdf`.
4. Sync narrative wording into `paper/PAPER_DRAFT.md`.
5. Sync into `paper/paper.docx`.
6. Re-run claim validator (cross-format drift checks).

## Formatting Constraints
- Use ASCII-safe symbols in raw source where possible.
- Keep run IDs exactly as `YYYYMMDD_HHMMSS`.
- Keep percentages rounded to 2 decimals per manifest.
- Avoid placeholder text (`TBD`, `TODO`, `[see latest frozen outputs]`, `To be assigned`).

## LaTeX Constraints
- `paper/paper.tex` must have zero unresolved `\input{}` paths.
- Prefer inline tables over fragile missing external table includes.
- Keep the title, abstract metric lines, and conclusion metric lines aligned with markdown.
- The canonical compiled artifact must exist at repo root as `paper.pdf`.

## DOCX Constraints
- DOCX is synced last and treated as release artifact.
- Any DOCX-only edits must be backported into markdown immediately.

## Environment Prerequisites
Current environment constraints detected:
- System `python3` does not have `python-docx`.
- Project `.venv` Python launcher is currently unusable on this machine state.

### Required for automated DOCX sync
- A working Python interpreter with `python-docx` installed.
- Optional render tooling for visual QA:
  - LibreOffice (`soffice`) for docx->pdf conversion.
  - Poppler (`pdftoppm`) for page image checks.

## Manual Fallback Procedure (When DOCX Automation Is Blocked)
1. Update `paper/paper.tex` first.
2. Validate with `scripts/validate_paper_claims.py`.
3. Compile and verify repo-root `paper.pdf`.
4. Export or edit DOCX manually.
5. Perform a textual drift check:
   - Title
   - Abstract metrics
   - Results key numbers
   - Conclusion key numbers
6. Log any unresolved drift in `paper/accuracy_audit.md`.

## Release Gates
The manuscript is release-ready only when:
1. Claim validator exits with code 0.
2. LaTeX compile succeeds without missing files.
3. Claim matrix has no unresolved `Conflicting` rows in active manuscript sections.
4. `Unsupported` and `Needs Citation` apply only to active manuscript claims; dormant rows must be marked `Inactive`.
5. Historical pre-lock rows must use status `Historical`, not `Conflicting`.
