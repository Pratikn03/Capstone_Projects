# ORIUS Paper Accuracy Audit

## Scope
This audit reconciles manuscript claims across:
- `paper/PAPER_DRAFT.md`
- `paper/paper.tex`
- `paper/paper.docx` (read-only drift check)
- `paper/paper.log`

Canonical policy: **dataset-scoped latest** as of February 17, 2026.

## Source-of-Truth Hierarchy
1. Decision impact:
   - `reports/impact_summary.csv` (DE)
   - `reports/eia930/impact_summary.csv` (US)
2. Stochastic metrics:
   - `reports/research_metrics_de.csv` run `20260217_165756`
   - `reports/research_metrics_us.csv` run `20260217_182305`
3. Dataset/system profile:
   - `data/dashboard/de_stats.json`
   - `data/dashboard/us_stats.json`
   - `data/dashboard/manifest.json`

## Current Consistency Status (Verified February 19, 2026)
1. Markdown and LaTeX claim validation passes.
- Evidence: `python3 scripts/validate_paper_claims.py` returns `PASS: no findings`.

2. LaTeX compilation succeeds (non-fatal typography warnings only).
- Evidence: `cd paper && pdflatex -interaction=nonstopmode -halt-on-error paper.tex` completed successfully and produced `paper/paper.pdf` (18 pages).

3. Historical blockers from earlier drafts are resolved in current `paper/paper.tex`.
- Missing `\input{}` compile failures previously noted in this project state are no longer active blockers in the current compiled file.

4. Cross-format drift has been reconciled for DOCX in the current pass.
- `paper/paper.docx` was regenerated from `paper/PAPER_DRAFT.md` via `pandoc`.
- Spot checks confirm canonical metrics/run IDs are present and legacy patterns (`2.89%`, `0.58%`, `1.35%`, `92,382`) are absent.

## Canonical Values (Locked)
### DE (impact + stochastic)
- Cost savings: **7.11%**
- Carbon reduction: **0.30%**
- Peak shaving: **6.13%**
- EVPI_robust: **2.32**
- EVPI_deterministic: **-30.40**
- VSS: **2,708.61**
- Stochastic run ID: `20260217_165756`

### US (impact + stochastic)
- Cost savings: **0.11%**
- Carbon reduction: **0.13%**
- Peak shaving: **0.00%**
- EVPI_robust: **10,279,851.74**
- EVPI_deterministic: **24,915,503.93**
- VSS: **297,092.71**
- Stochastic run ID: `20260217_182305`

### Dataset profile lock
- DE rows: **17,377**; range `2018-10-07T23:00:00+00:00` to `2020-09-30T23:00:00+00:00`; engineered features **94**.
- US rows: **13,638**; range `2024-07-01T06:00:00+00:00` to `2026-01-20T11:00:00+00:00`; engineered features **114**.

## Claim Status Summary
Detailed rows are in `paper/claim_matrix.csv`.

- `Verified`: locked to artifact-backed values and run IDs in the active release manuscript.
- `Historical`: retained only as a record of superseded pre-lock wording.
- `Inactive`: retained for future work, but absent from the active release manuscript.
- `Conflicting`: active manuscript wording contradicts canonical policy or breaks compilation.
- `Unsupported`: active manuscript wording has no local evidence in repo artifacts.
- `Needs Citation`: active manuscript wording is plausible, but no explicit source is currently linked.

Historical reconciliation rows are tracked explicitly with status `Historical` and should not be interpreted as active blockers in current markdown/LaTeX/DOCX output.

## Unsupported / Citation-Risk Claims
These must be cited or reframed before publication:
- Global storage outlook values (for example, "50 GW by 2030").
- Staffing/workforce claims tied to specific operators.
- Drift-rate and long-horizon degradation claims without explicit report source.
- ML carbon-footprint numbers without reproducible estimate method.

## Remediation Applied in This Implementation
1. Locked canonical values in `paper/metrics_manifest.json`.
2. Added claim traceability matrix in `paper/claim_matrix.csv`.
3. Rebuilt manuscript guidance in `paper/rewrite_pack.md` with dataset-scoped run governance.
4. Added synchronization contract in `paper/sync_rules.md`.
5. Added non-mutating validator in `scripts/validate_paper_claims.py`.
6. Rewrote manuscript files to remove known contradictions and broken table references.
7. Expanded `paper/PAPER_DRAFT.md` into a full thesis-style detailed draft while preserving canonical metrics and run IDs.

## Remaining Environment Constraint
`python-docx` constraints remain, but DOCX synchronization is currently achievable with `pandoc` in this environment.
